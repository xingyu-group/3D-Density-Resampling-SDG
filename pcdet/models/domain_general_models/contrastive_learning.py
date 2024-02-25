import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils


class Supervised_Contrastive_Loss(nn.Module):
    def __init__(self, model_cfg, backbone_channels, point_cloud_range, voxel_size,
                temperature=0.07, contrast_mode='all',base_temperature=0.07,
                **kwargs):
        super(Supervised_Contrastive_Loss, self).__init__()

        self.model_cfg = model_cfg
        self.domain_names = self.model_cfg.DOMAIN_NAMES    
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # CL loss settings
        self.temperature = float(self.model_cfg.TEMPERATURE) if self.model_cfg.get('TEMPERATURE', None) is not None else temperature
        self.contrast_mode = self.model_cfg.CONTRAST_MODE if self.model_cfg.get('CONTRAST_MODE', None) is not None else contrast_mode
        self.base_temperature = float(self.model_cfg.BASE_TEMPERATURE) if self.model_cfg.get('BASE_TEMPERATURE', None) is not None else base_temperature

        assert isinstance(self.domain_names, list)
        self.roi_grid_pool_layers = nn.ModuleList()

        if self.model_cfg.get('ROI_GRID_POOL_LINEAR', None) is not None :
            self.pool_cfg = model_cfg.ROI_GRID_POOL_LINEAR
            assert self.model_cfg.FEATURES_SOURCE == self.pool_cfg.FEATURES_SOURCE
            for src_name in self.pool_cfg.FEATURES_SOURCE:
                pool_function = voxelpool_stack_modules.NeighborVoxelLinearPoolModuleMSG(
                    query_ranges=self.pool_cfg.POOL_LAYERS[src_name].QUERY_RANGES,
                    nsamples=self.pool_cfg.POOL_LAYERS[src_name].NSAMPLE,
                    radii=self.pool_cfg.POOL_LAYERS[src_name].POOL_RADIUS,
                    pool_method=self.pool_cfg.POOL_LAYERS[src_name].POOL_METHOD,
                )
                self.roi_grid_pool_layers.append(pool_function)


    def roi_grid_pool(self, batch_dict, feature_selected, sampling_ratio=1.0):
        """
        Args:
            batch_dict:
                batch_size:
                gt_boxes: (B, max_gt, 7 + 1) [x, y, z, dx, dy, dz, heading, gt_label]   
            feature_selected: 'DIR' or 'DSR'
        Returns:

        """
        import random

        rois = batch_dict['gt_boxes']
        # rois = batch_dict[gt_selected]
        # # print(batch_dict['gt_boxes'])

        def filter_boxes_by_label(boxes, ratio=1):
            """
            Args:
                boxes: (B, max_gt, 7 + 1) [x, y, z, dx, dy, dz, heading, gt_label] # sample get_labels within [1,3] 
            Returns:

            """

            batch_size = boxes.shape[0]
            N_gt = boxes.shape[1]
            N_sample = N_gt
            for i_b in range(batch_size):
                N_sample = min(N_sample, torch.logical_and(boxes[i_b, :, 7]>0, boxes[i_b, :, 7]<4).sum().item())
            if ratio != 1:
                N_sample = int(N_sample * ratio)
            
            new_boxes = boxes.new_zeros(batch_size, N_sample, boxes.shape[-1])
            for i_b in range(batch_size):
                boxes_cur_bat = boxes[i_b, torch.logical_and(boxes[i_b, :, 7]>0, boxes[i_b, :, 7]<4), :] # (N, 8)
                index = torch.LongTensor(random.sample(range(boxes_cur_bat.shape[0]), N_sample)).to(boxes_cur_bat.device)
                boxes_cur_bat_sampled = torch.index_select(boxes_cur_bat, 0, index)

                new_boxes[i_b, :, :] = boxes_cur_bat_sampled
            
            return new_boxes, N_sample

        # rois, N_box_per_image = filter_boxes_by_label(rois, sampling_ratio)
        rois, _ = filter_boxes_by_label(rois, sampling_ratio)
        # print('rois')
        # print(type(rois))
        # print(rois.shape)
        # print('rois')

        batch_size = batch_dict['batch_size']
        # with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_dict = {}
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name == 'backbone3d_output':
                cur_stride = batch_dict['encoded_spconv_tensor_stride_' % feature_selected]
                cur_sp_tensors = batch_dict['encoded_spconv_tensor_' % feature_selected]

            else:
                cur_stride = batch_dict['multi_scale_3d_strides_%s' % feature_selected][src_name]
                cur_sp_tensors = batch_dict['multi_scale_3d_features_%s' % feature_selected][src_name]

            # if with_vf_transform:
            #     cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            # else:
            #     cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_dict[src_name]= pooled_features
        
        pooled_features_dict['gt_class_labels'] = rois[:, :, -1].view(-1,1)
       
        return pooled_features_dict
    
    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]
        # print(rois, batch_size_rcnn, grid_size)
        local_roi_grid_points = self.get_dense_grid_points(rois=rois, batch_size_rcnn=batch_size_rcnn, grid_size=grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    def get_dense_grid_points(self, rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict_1, batch_dict_2):
        '''
            batch_dict_*: 
                'encoded_spconv_tensor_stride_%s' % ('DIR' or 'DSR')
                'encoded_spconv_tensor_%s' % ('DIR' or 'DSR')
                'multi_scale_3d_strides_%s' % ('DIR' or 'DSR')
                'multi_scale_3d_features_%s' % ('DIR' or 'DSR')
                'gt_boxes'
                '%s_domain' % ('DIR' or 'DSR')

        '''

        feature_input_name = self.model_cfg.FEATURE_DOMAIN_PROPERTY

        feature_1_dict = self.roi_grid_pool(batch_dict_1, feature_selected=feature_input_name)
        feature_2_dict = self.roi_grid_pool(batch_dict_2, feature_selected=feature_input_name)

        domain_gt_1 = batch_dict_1['%s_domain' % feature_input_name]
        domain_gt_2 = batch_dict_2['%s_domain' % feature_input_name]
        assert (domain_gt_1 in self.domain_names) and (domain_gt_2 in self.domain_names)
        batch_size = batch_dict_1['batch_size']

        cl_losses_multi_src = {}

        for i_src, src_name in enumerate(self.model_cfg.FEATURES_SOURCE):

            feature_1_cur_src = feature_1_dict[src_name] #.clone()
            feature_2_cur_src = feature_2_dict[src_name] #.clone()

            sample_size_1 = feature_1_cur_src.shape[0]
            sample_size_2 = feature_2_cur_src.shape[0]
            total_features = torch.cat([feature_1_cur_src, feature_2_cur_src], dim=0).contiguous() # N_sample, 6x6x6, C

            domain_gt_label = torch.zeros((sample_size_1+sample_size_2), dtype=total_features.dtype, device=total_features.device)
            domain_gt_label[:sample_size_1] = self.domain_names.index(domain_gt_1)
            domain_gt_label[sample_size_1:] = self.domain_names.index(domain_gt_2)

            cl_losses_multi_src[src_name] = self.sup_con_loss(total_features.unsqueeze(1), domain_gt_label) # N_sample, 1, 6x6x6, C
            # print(domain_gt_label)
       
        return cl_losses_multi_src

    def sup_con_loss(self, features, labels):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].       Note that: n_views means different aug-transformations on one image 
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        assert labels is not None 
            
        labels = labels.contiguous().view(-1, 1)
        assert labels.shape[0] == batch_size
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = nn.functional.normalize(contrast_feature, p=2, dim=1) # normalization
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


