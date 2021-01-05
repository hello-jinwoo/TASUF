import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)
from ..registry import HEADS


@HEADS.register_module
class TrackHeadTASUF(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero 
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 with_avg_pool=False,
                 num_fcs = 2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 dynamic=True
                 ):
        super(TrackHeadTASUF, self).__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size) 

        # LSTM:
        #   There was no empirical exploration on 'input_size', 'hidden_size' and 'num_layers'.
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, 
                            num_layers=2, batch_first=False)

        # Convert ROI feature map of (7 x 7 x 256) to input vector for LSTM.
        self.in_fcs = nn.ModuleList()
        # Convert ROI feature map of (7 x 7 x 256) to an vector for matching score computation.
        self.query_fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            self.in_fcs.append(nn.Linear(in_channels, fc_out_channels))
            self.query_fcs.append(nn.Linear(in_channels, fc_out_channels))

        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic=dynamic

    def init_weights(self):
        for fc in self.in_fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)
        for fc in self.query_fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy =  torch.ones(bbox_ious.size(0), 1, 
                device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta),dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert(len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                + self.match_coeff[2] * label_delta
    
    def forward(self, x, ref_x, x_n, ref_x_n, gt_pids_list):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch, 
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        
        batch_size = len(x_n)
        num_ref = len(ref_x) 

        # Just rename the given parameters to match with the implicit attribute.
        ref_x_list = ref_x 
        ref_x_n_list = ref_x_n 

        for ref_x_n in ref_x_n_list:
            assert len(x_n) == len(ref_x_n)

        # Resize tensors to give it as input to FCs
        # (B * #proposals, 256, 7, 7) -> (B * #proposals, 256 * 7 * 7)
        x = x.view(x.size(0), -1) 
        # (seq_len, B * (#objects_1 + #objects_2 + ...), 256, 7, 7)
        #    -> (seq_len, B * (#objects_1 + #objects_2 + ...), 256 * 7 * 7)
        ref_x_list = [ref_x.view(ref_x.size(0), -1) for ref_x in ref_x_list]

        # Convert ROI feature to the query vector for matching score computation.
        # (B * #proposals, 256 * 7 * 7) -> (B * #proposals, 1024)
        for idx, fc in enumerate(self.query_fcs):
            x = fc(x)
            if idx < len(self.query_fcs) - 1:
                x = self.relu(x)

        # Convert ROI feature to the input vector for LSTM
        # (seq_len, B * (#objects_1 + #objects_2 + ...), 256 * 7 * 7) -> (seq_len, B * (#objects_1 + #objects_2 + ...), 1024)
        for idx, fc in enumerate(self.in_fcs):
            ref_x_list = list(map(fc, ref_x_list))
            if idx < len(self.in_fcs) - 1:
                ref_x_list = list(map(self.relu, ref_x_list))
        
        # Split tensors along the batch size (B).
        # (B * #proposals, 1024) -> (B, #proposals, 1024)
        # x_split:
        #       Each element consists of (#proposals=x_n[i], 1024) tensor.
        x_split = torch.split(x, x_n, dim=0) 

        # (seq_len, B * (#objects_1 + #objects_2 + ...), 1024) -> (seq_len, B, sum(#objects_i), 1024)
        ref_x_split_list = [torch.split(ref_x, ref_x_n, dim=0) 
                            for ref_x, ref_x_n, in zip(ref_x_list, ref_x_n_list)]

        # ref_x_dict_list:
        #   Description:
        #       List of ref_x_dict for each batch.
        #   Shape:
        #       (B, #ref_gt_pid, < seq_len, 1024)
        #   
        # ref_x_dict:
        #   Description:
        #       ref_x_dict[gt_pid] is fed as input to the LSTM
        #       to update hidden state corresponding to the specific 'gt_pid', 
        #   Key:
        #       ref_gt_pid : int
        #   Value: 
        #       Sequence of 'ref_x_split's corresponding to the designated 'gt_pid'
        #           : ( < seq_len, 1024)
        ref_x_dict_list = []
        for b in range(batch_size):
            ref_x_dict = dict()
            for i, ref_gt_pids in enumerate(gt_pids_list[:-1]): # Except for gt_pids of 'x' which is the current frame.
                for j, ref_gt_pid in enumerate(ref_gt_pids[b]): 
                    if ref_gt_pid == 0:
                        continue
                    ref_gt_pid = ref_gt_pid.item()
                    if ref_gt_pid in ref_x_dict:
                        ref_x_dict[ref_gt_pid] = torch.cat([ref_x_dict[ref_gt_pid], ref_x_split_list[i][b][j].unsqueeze(0)], dim=0)
                    else: 
                        ref_x_dict[ref_gt_pid] = ref_x_split_list[i][b][j].unsqueeze(0)
            ref_x_dict_list.append(ref_x_dict)

        match_score = []
        for b in range(batch_size):
            # for each ref_gt_pid
            h_t_list = []
            for ref_gt_pid, ref_x_split in sorted(ref_x_dict_list[b].items()): 
                ref_x_split = ref_x_split.unsqueeze(1) # (seq_len, 1024) -> (seq_len, 1, 1024)
                _, (h_t, c_t) = self.lstm(ref_x_split) # h_t: (num_layer=2, batch=1, hidden_size=1024)  
                h_t = h_t.squeeze(1)[-1] # (2, 1, 1024) -> (1024, )
                h_t_list.append(h_t)
            h_t_list = torch.stack(h_t_list, dim=0) # (#objects, 1024)

            prod = torch.mm(x_split[b], torch.transpose(h_t_list, 0, 1))
            m = prod.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())
            prod_ext = torch.cat([dummy, prod], dim=1)
            match_score.append(prod_ext)

        # match_score: (B, #proposals, #ref_gt_pids + 1)
        return match_score

    def loss(self,
             match_score,
             ids,
             id_weights,
             reduce=True):
        losses = dict()
        if self.dynamic:
            n = len(match_score)
            x_n = [s.size(0) for s in match_score]
            ids = torch.split(ids, x_n, dim=0)
            loss_match = torch.tensor([0.], device=torch.cuda.current_device())
            match_acc = 0.
            n_total = 0
            batch_size = len(ids)

            for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
                valid_idx = torch.nonzero(cur_weights).squeeze()
                if len(valid_idx.size()) == 0: continue
                n_valid = valid_idx.size(0)
                n_total += n_valid
                loss_match += weighted_cross_entropy(
                    score, cur_ids, cur_weights, reduce=reduce)
                match_acc += accuracy(torch.index_select(score, 0, valid_idx), 
                                      torch.index_select(cur_ids,0, valid_idx)) * n_valid
            losses['loss_match'] = loss_match / n
            if n_total > 0:
                losses['match_acc'] = match_acc / n_total
            else: 
                losses['match_acc'] = torch.tensor([100.], device=torch.cuda.current_device())
        else:
          if match_score is not None:
              valid_idx = torch.nonzero(cur_weights).squeeze()
              losses['loss_match'] = weighted_cross_entropy(
                  match_score, ids, id_weights, reduce=reduce)
              losses['match_acc'] = accuracy(torch.index_select(match_score, 0, valid_idx), 
                                              torch.index_select(ids, 0, valid_idx))
        return losses

    def forward_test(self, x, ref_x_hidden_states, x_n, ref_x_n):
        '''
        Args:
            ref_x_hidden_states:
                LSTM hidden states for each detected objects
                Shape: (# detected objects, 2, 1024)
                Example:[(h_1, c_1), (h_2, c_2), ... , ]
        '''

        assert len(x_n) == len(ref_x_n)
        batch_size = len(x_n)

        # Resize tensors to give it as input to FCs
        # (#proposals, 256, 7, 7) -> (#proposals, 256 * 7 * 7)
        x = x.view(x.size(0), -1) 

        # Convert ROI feature to the query vector for matching score computation.
        # (#proposals, 256 * 7 * 7) -> (#proposals, 1024)
        for idx, fc in enumerate(self.query_fcs):
            x = fc(x)
            if idx < len(self.query_fcs) - 1:
                x = self.relu(x)

        match_score = []
        prod = []

        # (#objects, hidden_state & cell_state = 2, num_layers=2, 1024) -> (#objects, 1024)
        ref_x_hidden_states = ref_x_hidden_states[:, 0, -1, 0, :] 

        prod = torch.mm(x, torch.transpose(ref_x_hidden_states, 0, 1)) # (#proposals, #objects)
        m = prod.size(0) # #proposals
        dummy = torch.zeros((m, 1), device=torch.cuda.current_device()) # (#proposals, 1)
        prod_ext = torch.cat([dummy, prod], dim=1) # (#proposals, #objects + 1)
        match_score.append(prod_ext)

        # match_score: (B, #proposals, #objects + 1)
        return match_score

    def init_hidden_states(self, det_roi_feats):
        """
        When it is the first time to feed an input vector to LSTM,
        update the hidden states based on the initial value of zeros.
        Therefore, only input vecotrs (det_roi_feats) are given here.
        """

        # det_roi_feats: (#proposals, 256, 7, 7) -> (#proposals, 1024)
        det_roi_feats = det_roi_feats.view(det_roi_feats.size(0), -1) # (#proposals, 256 * 7 * 7))
        for idx, fc in enumerate(self.in_fcs):
                det_roi_feats = fc(det_roi_feats)
                if idx < len(self.in_fcs) - 1:
                    det_roi_feats = self.relu(det_roi_feats)

        det_roi_feats = det_roi_feats.unsqueeze(0) # (#proposals, 1024) -> (1, #proposals, 1024)
        _, (h_t, c_t) = self.lstm(det_roi_feats) # h_t: (num_layers, #proposals, hidden_size)
        h_t = torch.transpose(h_t, 0, 1).unsqueeze(2) # (num_layers, #proposals, hidden_size) -> (#proposals, num_layers, 1, hidden_size)
        c_t = torch.transpose(c_t, 0, 1).unsqueeze(2) 
        hidden_states =  torch.stack([h_t, c_t], dim=1).to(torch.cuda.current_device())
        
        # hidden_states: (#proposals, h&c=2, num_layers=2, batch_size=1, hidden_size)
        return hidden_states 

    def update_hidden_state(self, det_roi_feat, hidden_state):
        """
        Update the hidden states based on the given hidden states
        when given the input vectors and hidden states.
        """

        # det_roi_feat: (256, 7, 7)
        # hidden_state: (h&c=2, num_layers=2, batch_size=1, hidden_size=1024)
        det_roi_feat = det_roi_feat.view(-1).unsqueeze(0) # (256, 7, 7) -> (1, 256 * 7 * 7)
        
        # det_roi_feat: (1, 256 * 7 * 7) -> (1, 1024)
        for idx, fc in enumerate(self.in_fcs):
            det_roi_feat = fc(det_roi_feat)
            if idx < len(self.in_fcs) - 1:
                det_roi_feat = self.relu(det_roi_feat)
        
        det_roi_feat = det_roi_feat.unsqueeze(0) # (batch_size=1, 1024) -> (seq_len=1, batch_size=1, 1024)
        _, hidden_state = self.lstm(det_roi_feat, hidden_state)
        hidden_state = torch.stack(hidden_state, dim=0)

        # hidden_state: (h&c=2, num_layers=2, batch=1, hidden_size=1024)
        return hidden_state 