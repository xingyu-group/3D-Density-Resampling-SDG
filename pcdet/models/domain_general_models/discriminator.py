import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from collections import OrderedDict
import random
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .gradient_reversal import revgrad

def conv2d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    return m

class Discriminator_BinaryDomain(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.domain_names = self.model_cfg.DOMAIN_NAMES
        self.multi_src_names = self.model_cfg.MULTI_SCALE_FEATURE.FEATURES_SOURCE
        self.multi_src_cfg = self.model_cfg.MULTI_SCALE_FEATURE

        self.MS_domain_classifiers = nn.ModuleList()
        for src in self.multi_src_names:
            classifier = nn.Sequential()

            classifier_conv_blocks = nn.Sequential()
            stride_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['stride']
            maxpool_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['maxpool']
            channel_input_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['channel_input']
            channel_output_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['channel_output']
            assert len(stride_list) == len(maxpool_list) == len(channel_input_list) == len(channel_output_list)
            for i, stides in enumerate(stride_list):
                classifier_conv_blocks.add_module(
                    'conv_block'+str(i+1),
                    self.build_conv2d_block(
                        strides = stides, 
                        use_maxpool = maxpool_list[i], 
                        channel_inputs = channel_input_list[i], 
                        channel_outputs = channel_output_list[i],
                    )
                )

            classifier_linear_blocks = nn.Sequential()
            classifier_linear_blocks.add_module('flatten', nn.Flatten())
            class_head_mlps = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].class_head_mlps
            for i in range(len(class_head_mlps)-1):
                classifier_linear_blocks.add_module('linear_block'+str(i+1), nn.Linear(class_head_mlps[i], class_head_mlps[i+1]))
            # classifier_linear_blocks.add_module('softmax', nn.Softmax(dim=1))

            classifier.add_module('conv_blocks',classifier_conv_blocks)
            classifier.add_module('linear_blocks',classifier_linear_blocks)
            self.MS_domain_classifiers.append(classifier)

        if self.model_cfg.get('FEATURE_CLASSIFIER', None) is not None:
            self.other_src_cfg = self.model_cfg.FEATURE_CLASSIFIER
            self.other_src_classifier = nn.Sequential()

            classifier_conv_blocks = nn.Sequential()
            stride_list = self.other_src_cfg.conv2d_downsample['stride']
            maxpool_list = self.other_src_cfg.conv2d_downsample['maxpool']
            channel_input_list = self.other_src_cfg.conv2d_downsample['channel_input']
            channel_output_list = self.other_src_cfg.conv2d_downsample['channel_output']
            assert len(stride_list) == len(maxpool_list) == len(channel_input_list) == len(channel_output_list)
            for i, stides in enumerate(stride_list):
                classifier_conv_blocks.add_module(
                    'conv_block'+str(i+1),
                    self.build_conv2d_block(
                        strides = stides, 
                        use_maxpool = maxpool_list[i], 
                        channel_inputs = channel_input_list[i], 
                        channel_outputs = channel_output_list[i],
                    )
                )

            classifier_linear_blocks = nn.Sequential()
            classifier_linear_blocks.add_module('flatten', nn.Flatten())
            class_head_mlps = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].class_head_mlps
            for i in range(len(class_head_mlps)-1):
                classifier_linear_blocks.add_module('linear_block'+str(i+1), nn.Linear(class_head_mlps[i], class_head_mlps[i+1]))
            # classifier_linear_blocks.add_module('softmax', nn.Softmax(dim=1))

            self.other_src_classifier.add_module('conv_blocks',classifier_conv_blocks)
            self.other_src_classifier.add_module('linear_blocks',classifier_linear_blocks)

    def build_conv2d_block(self, strides, use_maxpool, channel_inputs, channel_outputs):
        assert isinstance(strides, list)
        m = nn.Sequential()
        for i, stride in enumerate(strides):
            m.add_module(('conv%s' % (i+1)), conv2d_block(in_channels=channel_inputs[i], out_channels=channel_outputs[i], stride=stride))
        if use_maxpool:
            m.add_module(('maxpool%s' % (i+1)), nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        return m


    def forward(self, feature_cm_1: list, feature_cm_2: list, batch_size):
        '''
        Args:
            feature_cm_1 [0-1]: list
                feature_cm_1[0]: dict
                    'encoded_spconv_tensor': sp_tensor
                    'multi_scale_3d_features': dict
                        'x_conv1': sp_tensor
                        'x_conv2': sp_tensor
                        'x_conv3': sp_tensor
                        'x_conv4': sp_tensor
                    'DSR_domain': str

        '''

        if self.training:
            shuffle_feature = True
        else:
            shuffle_feature = False

        assert feature_cm_1[0]['DSR_domain'] == feature_cm_1[1]['DSR_domain']
        assert feature_cm_2[0]['DSR_domain'] == feature_cm_2[1]['DSR_domain']
        domain_gt_1 = feature_cm_1[0]['DSR_domain']
        domain_gt_2 = feature_cm_2[0]['DSR_domain']
        assert domain_gt_1 != domain_gt_2

        assert (domain_gt_1 in self.domain_names) and (domain_gt_2 in self.domain_names)
        
        result_dict = {}
        result_dict['domain_pred'] = {}
        result_dict['domain_gt_label'] = {}

        # multi-scale feature output for domain classification
        for i, src in enumerate(self.multi_src_names):
            # print(src+':')

            x_feature_1 = [feature_dict['multi_scale_3d_features'][src].dense() for feature_dict in feature_cm_1]
            x_feature_1 = torch.cat(x_feature_1, dim=0)
            N, C, D, H, W = x_feature_1.shape
            assert N == 2*batch_size
            x_feature_1 = x_feature_1.view(N, C * D, H, W)
            # print(x_feature_1.shape)

            x_feature_2 = [feature_dict['multi_scale_3d_features'][src].dense() for feature_dict in feature_cm_2]
            x_feature_2 = torch.cat(x_feature_2, dim=0)
            N, C, D, H, W = x_feature_2.shape
            assert N == 2*batch_size
            x_feature_2 = x_feature_2.view(N, C * D, H, W)
            # print(x_feature_2.shape)


            
            order_indicator = random.choice([0,1])
            if shuffle_feature and order_indicator:
                # 'detech' for only training classifiers
                x_features = torch.cat([x_feature_2, x_feature_1], dim=0).detach()
            else:
                x_features = torch.cat([x_feature_1, x_feature_2], dim=0).detach()
            assert x_features.shape[0] == 4*batch_size
            # print(x_features.shape)
            
            x_features = x_features.contiguous()

            ## predicting domain/dataset
            output= self.MS_domain_classifiers[i].conv_blocks(x_features)
            # print(output.requires_grad)
            output= self.MS_domain_classifiers[i].linear_blocks(output)
            result_dict['domain_pred'][src] = output

            ## generating labels of domain
            domain_gt_label = torch.zeros(4*batch_size, device=output.device)
            if shuffle_feature and order_indicator:
                domain_gt_label[:2*batch_size] = self.domain_names.index(domain_gt_2)
                domain_gt_label[2*batch_size:] = self.domain_names.index(domain_gt_1)
            else:
                domain_gt_label[:2*batch_size] = self.domain_names.index(domain_gt_1)
                domain_gt_label[2*batch_size:] = self.domain_names.index(domain_gt_2)
            result_dict['domain_gt_label'][src] = domain_gt_label

        # final feature output for domain classification
        if self.model_cfg.get('FEATURE_CLASSIFIER', None) is not None:
            # print(self.other_src_cfg.SOURCE_NAME)
            feature_scr_name = self.other_src_cfg.FEATURE_NAME
            x_feature_1 = [feature_dict[feature_scr_name].dense().clone() for feature_dict in feature_cm_1]
            x_feature_1 = torch.cat(x_feature_1, dim=0)
            N, C, D, H, W = x_feature_1.shape
            assert N == 2*batch_size
            x_feature_1 = x_feature_1.view(N, C * D, H, W)

            x_feature_2 = [feature_dict[feature_scr_name].dense().clone() for feature_dict in feature_cm_2]
            x_feature_2 = torch.cat(x_feature_2, dim=0)
            N, C, D, H, W = x_feature_2.shape
            assert N == 2*batch_size
            x_feature_2 = x_feature_2.view(N, C * D, H, W)

            order_indicator = random.choice([0,1])
            if shuffle_feature and order_indicator:
                # 'detech' for only training classifiers
                x_features = torch.cat([x_feature_2, x_feature_1], dim=0).detach()
            else:
                x_features = torch.cat([x_feature_1, x_feature_2], dim=0).detach()
            assert x_features.shape[0] == 4*batch_size
            
            x_features = x_features.contiguous()

            ## predicting domain/dataset
            output= self.other_src_classifier.conv_blocks(x_features)
            # print(output.shape)
            output= self.other_src_classifier.linear_blocks(output)
            result_dict['domain_pred'][self.other_src_cfg.SOURCE_NAME] = output

            ## generating labels of domain
            domain_gt_label = torch.zeros(4*batch_size, device=output.device)
            if shuffle_feature and order_indicator:
                domain_gt_label[:2*batch_size] = self.domain_names.index(domain_gt_2)
                domain_gt_label[2*batch_size:] = self.domain_names.index(domain_gt_1)
            else:
                domain_gt_label[:2*batch_size] = self.domain_names.index(domain_gt_1)
                domain_gt_label[2*batch_size:] = self.domain_names.index(domain_gt_2)
            result_dict['domain_gt_label'][self.other_src_cfg.SOURCE_NAME] = domain_gt_label

        return result_dict


    def get_discriminator_loss(self, result_dict):

        discriminator_loss_multi_src = {}
        for src_name in list(result_dict['domain_pred'].keys()):
            discriminator_loss_multi_src[src_name+'_loss'] = F.cross_entropy(result_dict['domain_pred'][src_name], result_dict['domain_gt_label'][src_name].long())

        return discriminator_loss_multi_src


class Discriminator_BinaryDomain_TestTime(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.domain_names = self.model_cfg.DOMAIN_NAMES
        self.multi_src_names = self.model_cfg.MULTI_SCALE_FEATURE.FEATURES_SOURCE
        self.multi_src_cfg = self.model_cfg.MULTI_SCALE_FEATURE

        self.MS_domain_classifiers = nn.ModuleList()
        for src in self.multi_src_names:
            classifier = nn.Sequential()

            classifier_conv_blocks = nn.Sequential()
            stride_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['stride']
            maxpool_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['maxpool']
            channel_input_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['channel_input']
            channel_output_list = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].conv2d_downsample['channel_output']
            assert len(stride_list) == len(maxpool_list) == len(channel_input_list) == len(channel_output_list)
            for i, stides in enumerate(stride_list):
                classifier_conv_blocks.add_module(
                    'conv_block'+str(i+1),
                    self.build_conv2d_block(
                        strides = stides, 
                        use_maxpool = maxpool_list[i], 
                        channel_inputs = channel_input_list[i], 
                        channel_outputs = channel_output_list[i],
                    )
                )

            classifier_linear_blocks = nn.Sequential()
            classifier_linear_blocks.add_module('flatten', nn.Flatten())
            class_head_mlps = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].class_head_mlps
            for i in range(len(class_head_mlps)-1):
                classifier_linear_blocks.add_module('linear_block'+str(i+1), nn.Linear(class_head_mlps[i], class_head_mlps[i+1]))
            # classifier_linear_blocks.add_module('softmax', nn.Softmax(dim=1))

            classifier.add_module('conv_blocks',classifier_conv_blocks)
            classifier.add_module('linear_blocks',classifier_linear_blocks)
            self.MS_domain_classifiers.append(classifier)

        if self.model_cfg.get('FEATURE_CLASSIFIER', None) is not None:
            self.other_src_cfg = self.model_cfg.FEATURE_CLASSIFIER
            self.other_src_classifier = nn.Sequential()

            classifier_conv_blocks = nn.Sequential()
            stride_list = self.other_src_cfg.conv2d_downsample['stride']
            maxpool_list = self.other_src_cfg.conv2d_downsample['maxpool']
            channel_input_list = self.other_src_cfg.conv2d_downsample['channel_input']
            channel_output_list = self.other_src_cfg.conv2d_downsample['channel_output']
            assert len(stride_list) == len(maxpool_list) == len(channel_input_list) == len(channel_output_list)
            for i, stides in enumerate(stride_list):
                classifier_conv_blocks.add_module(
                    'conv_block'+str(i+1),
                    self.build_conv2d_block(
                        strides = stides, 
                        use_maxpool = maxpool_list[i], 
                        channel_inputs = channel_input_list[i], 
                        channel_outputs = channel_output_list[i],
                    )
                )

            classifier_linear_blocks = nn.Sequential()
            classifier_linear_blocks.add_module('flatten', nn.Flatten())
            class_head_mlps = self.multi_src_cfg.CLASSIFICATION_LAYERS[src].class_head_mlps
            for i in range(len(class_head_mlps)-1):
                classifier_linear_blocks.add_module('linear_block'+str(i+1), nn.Linear(class_head_mlps[i], class_head_mlps[i+1]))
            # classifier_linear_blocks.add_module('softmax', nn.Softmax(dim=1))

            self.other_src_classifier.add_module('conv_blocks',classifier_conv_blocks)
            self.other_src_classifier.add_module('linear_blocks',classifier_linear_blocks)

    def build_conv2d_block(self, strides, use_maxpool, channel_inputs, channel_outputs):
        assert isinstance(strides, list)
        m = nn.Sequential()
        for i, stride in enumerate(strides):
            m.add_module(('conv%s' % (i+1)), conv2d_block(in_channels=channel_inputs[i], out_channels=channel_outputs[i], stride=stride))
        if use_maxpool:
            m.add_module(('maxpool%s' % (i+1)), nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        return m


    def forward(self, feature_cm_1: list, feature_cm_2: list, batch_size):
        '''
        Args:
            feature_cm_1 [0-1]: list
                feature_cm_1[0]: dict
                    'encoded_spconv_tensor': sp_tensor
                    'multi_scale_3d_features': dict
                        'x_conv1': sp_tensor
                        'x_conv2': sp_tensor
                        'x_conv3': sp_tensor
                        'x_conv4': sp_tensor
                    'DSR_domain': str

        '''

        if self.training:
            shuffle_feature = True
        else:
            shuffle_feature = False

        assert len(feature_cm_1) == len(feature_cm_2)
        batch_num_per_cm = len(feature_cm_1)
        domain_gt_1 = feature_cm_1[0]['DSR_domain']
        domain_gt_2 = feature_cm_2[0]['DSR_domain']
        assert domain_gt_1 != domain_gt_2

        assert (domain_gt_1 in self.domain_names) and (domain_gt_2 in self.domain_names)
        
        result_dict = {}
        result_dict['domain_pred'] = {}
        result_dict['domain_gt_label'] = {}

        # multi-scale feature output for domain classification
        for i, src in enumerate(self.multi_src_names):
            # print(src+':')

            x_feature_1 = [feature_dict['multi_scale_3d_features'][src].dense() for feature_dict in feature_cm_1]
            x_feature_1 = torch.cat(x_feature_1, dim=0)
            N, C, D, H, W = x_feature_1.shape
            assert N == batch_num_per_cm*batch_size
            x_feature_1 = x_feature_1.view(N, C * D, H, W)
            # print(x_feature_1.shape)

            x_feature_2 = [feature_dict['multi_scale_3d_features'][src].dense() for feature_dict in feature_cm_2]
            x_feature_2 = torch.cat(x_feature_2, dim=0)
            N, C, D, H, W = x_feature_2.shape
            assert N == batch_num_per_cm*batch_size
            x_feature_2 = x_feature_2.view(N, C * D, H, W)
            # print(x_feature_2.shape)

            order_indicator = random.choice([0,1])
            if shuffle_feature and order_indicator:
                # 'detech' for only training classifiers
                x_features = torch.cat([x_feature_2, x_feature_1], dim=0)
            else:
                x_features = torch.cat([x_feature_1, x_feature_2], dim=0)
            assert x_features.shape[0] == 2*batch_num_per_cm*batch_size
            # print(x_features.shape)
            
            x_features = x_features.contiguous()

            ## predicting domain/dataset
            output= self.MS_domain_classifiers[i].conv_blocks(x_features)
            # print(output.requires_grad)
            output= self.MS_domain_classifiers[i].linear_blocks(output)
            result_dict['domain_pred'][src] = output

            ## generating labels of domain
            domain_gt_label = torch.zeros(2*batch_num_per_cm*batch_size, device=output.device)
            if shuffle_feature and order_indicator:
                domain_gt_label[:batch_num_per_cm*batch_size] = self.domain_names.index(domain_gt_2)
                domain_gt_label[batch_num_per_cm*batch_size:] = self.domain_names.index(domain_gt_1)
            else:
                domain_gt_label[:batch_num_per_cm*batch_size] = self.domain_names.index(domain_gt_1)
                domain_gt_label[batch_num_per_cm*batch_size:] = self.domain_names.index(domain_gt_2)
            result_dict['domain_gt_label'][src] = domain_gt_label
            # print(domain_gt_label)

        # final feature output for domain classification
        if self.model_cfg.get('FEATURE_CLASSIFIER', None) is not None:
            # print(self.other_src_cfg.SOURCE_NAME)
            feature_scr_name = self.other_src_cfg.FEATURE_NAME
            x_feature_1 = [feature_dict[feature_scr_name].dense().clone() for feature_dict in feature_cm_1]
            x_feature_1 = torch.cat(x_feature_1, dim=0)
            N, C, D, H, W = x_feature_1.shape
            assert N == batch_num_per_cm*batch_size
            x_feature_1 = x_feature_1.view(N, C * D, H, W)

            x_feature_2 = [feature_dict[feature_scr_name].dense().clone() for feature_dict in feature_cm_2]
            x_feature_2 = torch.cat(x_feature_2, dim=0)
            N, C, D, H, W = x_feature_2.shape
            assert N == batch_num_per_cm*batch_size
            x_feature_2 = x_feature_2.view(N, C * D, H, W)

            order_indicator = random.choice([0,1])
            if shuffle_feature and order_indicator:
                # 'detech' for only training classifiers
                x_features = torch.cat([x_feature_2, x_feature_1], dim=0)
            else:
                x_features = torch.cat([x_feature_1, x_feature_2], dim=0)
            assert x_features.shape[0] == 2*batch_num_per_cm*batch_size
            
            x_features = x_features.contiguous()

            ## predicting domain/dataset
            output= self.other_src_classifier.conv_blocks(x_features)
            # print(output.shape)
            output= self.other_src_classifier.linear_blocks(output)
            result_dict['domain_pred'][self.other_src_cfg.SOURCE_NAME] = output

            ## generating labels of domain
            domain_gt_label = torch.zeros(2*batch_num_per_cm*batch_size, device=output.device)
            if shuffle_feature and order_indicator:
                domain_gt_label[:batch_num_per_cm*batch_size] = self.domain_names.index(domain_gt_2)
                domain_gt_label[batch_num_per_cm*batch_size:] = self.domain_names.index(domain_gt_1)
            else:
                domain_gt_label[:batch_num_per_cm*batch_size] = self.domain_names.index(domain_gt_1)
                domain_gt_label[batch_num_per_cm*batch_size:] = self.domain_names.index(domain_gt_2)
            result_dict['domain_gt_label'][self.other_src_cfg.SOURCE_NAME] = domain_gt_label

        return result_dict


    def get_discriminator_loss(self, result_dict):

        discriminator_loss_multi_src = {}
        for src_name in list(result_dict['domain_pred'].keys()):
            discriminator_loss_multi_src[src_name+'_loss'] = F.cross_entropy(result_dict['domain_pred'][src_name], result_dict['domain_gt_label'][src_name].long())

        return discriminator_loss_multi_src

class Discriminator_Object_BinaryDomain(nn.Module):
    def __init__(self, model_cfg, backbone_channels, point_cloud_range, voxel_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.domain_names = self.model_cfg.DOMAIN_NAMES
        self.classification_cfg = model_cfg.CLASSIFICATION      
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.reverse_grad_feat = model_cfg.REVERSE_GRADIENT_FEAT if self.model_cfg.get('REVERSE_GRADIENT_FEAT', None) is not None else False
        assert isinstance(self.domain_names, list)
        self.roi_grid_pool_layers = nn.ModuleList()

        # Note, only one of ROI_GRID_POOL and ROI_GRID_POOL_LINEAR must be ininitialized in config_yaml 
        if self.model_cfg.get('ROI_GRID_POOL', None) is not None :
            self.pool_cfg = model_cfg.ROI_GRID_POOL
            LAYER_cfg = self.pool_cfg.POOL_LAYERS
            assert self.model_cfg.FEATURES_SOURCE == self.model_cfg.ROI_GRID_POOL.FEATURES_SOURCE == self.model_cfg.CLASSIFICATION.FEATURES_SOURCE
            for src_name in self.pool_cfg.FEATURES_SOURCE:
                mlps = LAYER_cfg[src_name].MLPS
                for k in range(len(mlps)):
                    mlps[k] = [backbone_channels[src_name]] + mlps[k]
                pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                    query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                    nsamples=LAYER_cfg[src_name].NSAMPLE,
                    radii=LAYER_cfg[src_name].POOL_RADIUS,
                    mlps=mlps,
                    pool_method=LAYER_cfg[src_name].POOL_METHOD,
                ) 
                self.roi_grid_pool_layers.append(pool_layer)

        if self.model_cfg.get('ROI_GRID_POOL_LINEAR', None) is not None :
            self.pool_cfg = model_cfg.ROI_GRID_POOL_LINEAR
            assert self.model_cfg.FEATURES_SOURCE == self.pool_cfg.FEATURES_SOURCE == self.model_cfg.CLASSIFICATION.FEATURES_SOURCE
            for src_name in self.pool_cfg.FEATURES_SOURCE:
                pool_function = voxelpool_stack_modules.NeighborVoxelLinearPoolModuleMSG(
                    query_ranges=self.pool_cfg.POOL_LAYERS[src_name].QUERY_RANGES,
                    nsamples=self.pool_cfg.POOL_LAYERS[src_name].NSAMPLE,
                    radii=self.pool_cfg.POOL_LAYERS[src_name].POOL_RADIUS,
                    pool_method=self.pool_cfg.POOL_LAYERS[src_name].POOL_METHOD,
                )
                self.roi_grid_pool_layers.append(pool_function)

        self.classification_layers = nn.ModuleList()
        for src_name in self.classification_cfg.FEATURES_SOURCE:
            class_head_mlps = self.classification_cfg.CLASS_LAYERS[src_name]
            linear_block = nn.Sequential()
            for i in range(len(class_head_mlps)-1):
                linear_block.add_module('linear_layer'+str(i+1), nn.Linear(class_head_mlps[i], class_head_mlps[i+1]))
            # linear_block.add_module('softmax', nn.Softmax(dim=1))

            self.classification_layers.append(linear_block)


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

        domain_pred_multi_src = {}
        domain_gt_label_multi_src = {}
        result_dict = {}
        for i_src, src_name in enumerate(self.model_cfg.FEATURES_SOURCE):
                          
            feature_1_cur_src = feature_1_dict[src_name] #.clone()
            feature_2_cur_src = feature_2_dict[src_name] #.clone()
            # print(feature_1_cur_src.requires_grad)

            if self.reverse_grad_feat:
                feature_1_cur_src = revgrad(feature_1_cur_src, torch.tensor([1.], device=feature_1_cur_src.device))
                feature_2_cur_src = revgrad(feature_2_cur_src, torch.tensor([1.], device=feature_2_cur_src.device))
                # print('*** reversing gradient ***')

            sample_size_1 = feature_1_cur_src.shape[0]
            sample_size_2 = feature_2_cur_src.shape[0]

            total_features = torch.cat([feature_1_cur_src, feature_2_cur_src], dim=0).contiguous() # N_sample, 6x6x6, C

            # shuffle
            idx = torch.randperm(total_features.shape[0])
            total_features = total_features[idx, :].view(total_features.size())

            ## predicting domain/dataset
            output = nn.Flatten()(total_features)
            output = self.classification_layers[i_src](output)
            domain_pred_multi_src[src_name] = output

            domain_gt_label = torch.zeros((sample_size_1+sample_size_2), dtype=output.dtype, device=output.device)
            domain_gt_label[:sample_size_1] = self.domain_names.index(domain_gt_1)
            domain_gt_label[sample_size_1:] = self.domain_names.index(domain_gt_2)

            # shuffle
            domain_gt_label = domain_gt_label[idx].view(domain_gt_label.size())

            domain_gt_label_multi_src[src_name] = domain_gt_label

        result_dict['domain_pred'] = domain_pred_multi_src
        result_dict['domain_gt_label'] = domain_gt_label_multi_src
        # print(torch.nn.Softmax(dim=1)(result_dict['domain_pred']['x_conv2']))
        # print(result_dict['domain_gt_label']['x_conv2'])
        
        return result_dict


    def get_discriminator_loss(self, result_dict):

        discriminator_loss_multi_src = {}
        for src_name in list(result_dict['domain_pred'].keys()):
            discriminator_loss_multi_src[src_name+'_loss'] = F.cross_entropy(result_dict['domain_pred'][src_name], result_dict['domain_gt_label'][src_name].long())

        return discriminator_loss_multi_src

class Orthogonal_DIR_DSR_Object(nn.Module):
    def __init__(self, model_cfg, backbone_channels, point_cloud_range, voxel_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg     
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.roi_grid_pool_layers = nn.ModuleList()

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

    def roi_grid_pool(self, batch_dict, sampling_ratio=1.0):
        """
        Args:
            batch_dict:
                batch_size:
                gt_boxes: (B, max_gt, 7 + 1) [x, y, z, dx, dy, dz, heading, gt_label]   
        Returns:

        """
        import random

        rois = batch_dict['gt_boxes']
        # rois = batch_dict[gt_selected]
        # # print(batch_dict['gt_boxes'])

        def filter_boxes_by_label(boxes, ratio=1.0):
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
            pooled_features_dict[src_name] = {}
            for feature_selected in ['DIR', 'DSR']:
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
                pooled_features_dict[src_name][feature_selected]= pooled_features
        
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



    def forward(self, batch_dict):
        '''
            batch_dict_*: 
                'encoded_spconv_tensor_stride_%s' % ('DIR' or 'DSR')
                'encoded_spconv_tensor_%s' % ('DIR' or 'DSR')
                'multi_scale_3d_strides_%s' % ('DIR' or 'DSR')
                'multi_scale_3d_features_%s' % ('DIR' or 'DSR')
                'gt_boxes'
        '''

        feature_dict = self.roi_grid_pool(batch_dict)
        batch_size = batch_dict['batch_size']

        orth_loss_dict = {}
        for i_src, src_name in enumerate(self.model_cfg.FEATURES_SOURCE):
            # (N_obj, 6x6x6, C)
            feature_dir_cur_src = feature_dict[src_name]['DIR'] 
            feature_dsr_cur_src = feature_dict[src_name]['DSR'] 

            # (N_obj, 6x6x6, C) => (N_obj, C, 6x6x6)
            feature_dir_cur_src = feature_dir_cur_src.permute((0, 2, 1)) 
            feature_dsr_cur_src = feature_dsr_cur_src.permute((0, 2, 1))

            global_feat_dir_cur_src = F.avg_pool1d(feature_dir_cur_src, kernel_size=feature_dir_cur_src.shape[2])[:,:,0]
            global_feat_dsr_cur_src = F.avg_pool1d(feature_dsr_cur_src, kernel_size=feature_dsr_cur_src.shape[2])[:,:,0]

            global_feat_dir_cur_src = F.normalize(global_feat_dir_cur_src, dim=1)
            global_feat_dsr_cur_src = F.normalize(global_feat_dsr_cur_src, dim=1)
            mutual_loss_cur_src = global_feat_dir_cur_src * global_feat_dsr_cur_src
            mutual_loss_cur_src = torch.abs(torch.sum(mutual_loss_cur_src, dim=1))
            mutual_loss_cur_src = torch.mean(mutual_loss_cur_src)

            orth_loss_dict[src_name] = mutual_loss_cur_src

        return orth_loss_dict
