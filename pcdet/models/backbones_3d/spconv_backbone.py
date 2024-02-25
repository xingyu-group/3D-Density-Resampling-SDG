from functools import partial

import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils

from spconv.pytorch import functional as Fsp
from ..domain_general_models.pointnet2_encoder import pointnet2_perceptual_backbone

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        batch_dict['input_sp_tensor'] = input_sp_tensor     
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x_Decoder(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_point_features = 3

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        block = post_act_block
        self.perceptual_module = pointnet2_perceptual_backbone(self.voxel_size, self.point_cloud_range, self.num_point_features) \
            if self.model_cfg.get('PERCEPTUAL_LOSS_CALCULATION', False) else None

        # self.conv_input = spconv.SparseSequential(
        #     spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
        #     norm_fn(16),
        #     nn.ReLU(),
        # )

        # self.conv1 = spconv.SparseSequential(
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        # )

        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        # )

        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        # )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )


        # last_pad = self.model_cfg.get('last_pad', 0)

        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )


        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_m4 = block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_m3 = block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_m2 = block(32, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_m1 = block(16, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            spconv.SubMConv3d(16, 3, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(3),
        )

    def UR_block_forward(self, x_bottom, conv_m, conv_inv):
        x_m = conv_m(x_bottom)
        x = conv_inv(x_m)
        return x


    def forward(self, x_conv4):
        
        # x_conv4 = batch_dict['multi_scale_3d_features']['x_conv4']
        
        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_up4, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_up3, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_up2, self.conv_up_m1, self.conv5)

        return x_up1

    def get_loss(self, upsample_feat_sp, orginal_feat_sp, org_norm=False):
        feat_ups = upsample_feat_sp.features
        if org_norm:
            feat_org = orginal_feat_sp.features 
        else:
            batch_fun = torch.nn.BatchNorm1d(3, affine=False).cuda()
            feat_org = batch_fun(orginal_feat_sp.features)

        # mse loss on original scale 
        loss_recon = {}
        loss_recon['org_recon_loss'] = torch.mean((feat_ups - feat_org)**2)

        if self.model_cfg.get('PERCEPTUAL_LOSS_CALCULATION', False):
            # assert upsample_feat_sp.indices == orginal_feat_sp.indices
            batch_size = orginal_feat_sp.batch_size 
            feats_multi_layer_1, feats_multi_layer_2 = self.perceptual_module(
                feat_1 = feat_ups, 
                feat_2 = feat_org, 
                indices_shared = orginal_feat_sp.indices, 
                batch_size = batch_size
            )
            loss_recon['pcp_recon_loss'] = self.perceptual_module.get_loss(feats_multi_layer_1, feats_multi_layer_2)
        
        return loss_recon


class VoxelBackBone8x_DIR(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor_DIR': out,
            'encoded_spconv_tensor_stride_DIR': 8
        })
        batch_dict.update({
            'multi_scale_3d_features_DIR': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides_DIR': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        # print("forarding DIR")

        return batch_dict

class VoxelBackBone8x_DSR(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.domain_name = str(model_cfg.SRC_DOMAIN)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor_DSR': out,
            'encoded_spconv_tensor_stride_DSR': 8
        })
        batch_dict.update({
            'multi_scale_3d_features_DSR': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides_DSR': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        batch_dict.update({
            'DSR_domain':self.domain_name if self.domain_name != 'None' else batch_dict['dataset_domain']
        })
        # print("forarding DSR %s " % self.domain_name)

        return batch_dict

class VoxelBackBone8x_All2D(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
            'x_conv_out': 128,
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor_ALL': out,
            'encoded_spconv_tensor_stride_ALL': 8
        })
        batch_dict.update({
            'multi_scale_3d_features_ALL': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides_ALL': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        # print("forwarding ALL (DSR + DIR)")

        return batch_dict


class VoxelBackBone8x_DSR_SubGen(nn.Module):
    def __init__(self, model_cfg, backbone_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone_channels = backbone_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        block = post_act_block

        self.conv2 = spconv.SparseSequential(
            block(self.backbone_channels['x_conv2'], self.backbone_channels['x_conv2'], 3, norm_fn=norm_fn, padding=1, indice_key='spconv2_dir', conv_type='spconv'),
            block(self.backbone_channels['x_conv2'], self.backbone_channels['x_conv2'], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_dir'),
            # block(self.backbone_channels['x_conv2'], self.backbone_channels['x_conv2'], 3, norm_fn=norm_fn, padding=1, indice_key='subm2_dir'),
        )

        self.conv3 = spconv.SparseSequential(
            block(self.backbone_channels['x_conv3'], self.backbone_channels['x_conv3'], 3, norm_fn=norm_fn, padding=1, indice_key='spconv3_dir', conv_type='spconv'),
            block(self.backbone_channels['x_conv3'], self.backbone_channels['x_conv3'], 3, norm_fn=norm_fn, padding=1, indice_key='subm3_dir'),
            # block(self.backbone_channels['x_conv3'], self.backbone_channels['x_conv3'], 3, norm_fn=norm_fn, padding=1, indice_key='subm3_dir'),
        )

        self.conv4 = spconv.SparseSequential(
            block(self.backbone_channels['x_conv4'], self.backbone_channels['x_conv4'], 3, norm_fn=norm_fn, padding=1, indice_key='spconv4_dir', conv_type='spconv'),
            block(self.backbone_channels['x_conv4'], self.backbone_channels['x_conv4'], 3, norm_fn=norm_fn, padding=1, indice_key='subm4_dir'),
            # block(self.backbone_channels['x_conv4'], self.backbone_channels['x_conv4'], 3, norm_fn=norm_fn, padding=1, indice_key='subm4_dir'),
        )

        self.conv_out = spconv.SparseSequential(
            block(self.backbone_channels['x_conv_out'], self.backbone_channels['x_conv_out'], 3, norm_fn=norm_fn, padding=1, indice_key='spconv5_dir', conv_type='spconv'),
            block(self.backbone_channels['x_conv_out'], self.backbone_channels['x_conv_out'], 3, norm_fn=norm_fn, padding=1, indice_key='subm5_dir'),
            # block(self.backbone_channels['x_conv_out'], self.backbone_channels['x_conv_out'], 3, norm_fn=norm_fn, padding=1, indice_key='subm5_dir'),
        )

    def sub_spconv_a_b(self, spconv_a, spconv_b):
        neg_spconv_b = replace_feature(spconv_b, -1*spconv_b.features)
        return Fsp.sparse_add(spconv_a, neg_spconv_b)
  
    def forward(self, batch_dict):
        
        # extract DIR 
        x_conv2_dir = self.conv2(batch_dict['multi_scale_3d_features_ALL']['x_conv2'])
        x_conv3_dir = self.conv3(batch_dict['multi_scale_3d_features_ALL']['x_conv3'])
        x_conv4_dir = self.conv4(batch_dict['multi_scale_3d_features_ALL']['x_conv4'])
        x_conv_output_dir = self.conv_out(batch_dict['encoded_spconv_tensor_ALL'])

        batch_dict.update({
            'encoded_spconv_tensor_DIR': x_conv_output_dir,
            'encoded_spconv_tensor_stride_DIR': batch_dict['encoded_spconv_tensor_stride_ALL']
        })
        batch_dict.update({
            'multi_scale_3d_features_DIR': {
                # 'x_conv1': x_conv1_dir,
                'x_conv2': x_conv2_dir,
                'x_conv3': x_conv3_dir,
                'x_conv4': x_conv4_dir,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides_DIR': {
                # 'x_conv1': batch_dict['multi_scale_3d_strides_ALL']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_ALL']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_ALL']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_ALL']['x_conv4'],
            }
        })
        batch_dict.update({
            'DIR_domain': batch_dict['dataset_domain']
        })


        # extract DSR
        x_conv2_dsr = self.sub_spconv_a_b(batch_dict['multi_scale_3d_features_ALL']['x_conv2'], x_conv2_dir)
        x_conv3_dsr = self.sub_spconv_a_b(batch_dict['multi_scale_3d_features_ALL']['x_conv3'], x_conv3_dir)
        x_conv4_dsr = self.sub_spconv_a_b(batch_dict['multi_scale_3d_features_ALL']['x_conv4'], x_conv4_dir)
        x_conv_output_dsr = self.sub_spconv_a_b(batch_dict['encoded_spconv_tensor_ALL'], x_conv_output_dir)

        batch_dict.update({
            'encoded_spconv_tensor_DSR': x_conv_output_dsr,
            'encoded_spconv_tensor_stride_DSR': batch_dict['encoded_spconv_tensor_stride_ALL']
        })
        batch_dict.update({
            'multi_scale_3d_features_DSR': {
                # 'x_conv1': x_conv1_dsr,
                'x_conv2': x_conv2_dsr,
                'x_conv3': x_conv3_dsr,
                'x_conv4': x_conv4_dsr,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides_DSR': {
                # 'x_conv1': batch_dict['multi_scale_3d_strides_ALL']['x_conv1'],
                'x_conv2': batch_dict['multi_scale_3d_strides_ALL']['x_conv2'],
                'x_conv3': batch_dict['multi_scale_3d_strides_ALL']['x_conv3'],
                'x_conv4': batch_dict['multi_scale_3d_strides_ALL']['x_conv4'],
            }
        })

        batch_dict.update({
            'DSR_domain': batch_dict['dataset_domain']
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict


# ------------------------------------------------------- # 
# ------------------New 3D Backbone---------------------- #
# ------------------------------------------------------- # 
class VoxelWideResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.input_channels = model_cfg.IN_CHANNELS
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.wide_factor = model_cfg.WIDE_FACTOR
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, 16*self.wide_factor, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16*self.wide_factor),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16*self.wide_factor, 16*self.wide_factor, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16*self.wide_factor, 16*self.wide_factor, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*self.wide_factor, 32*self.wide_factor, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32*self.wide_factor, 32*self.wide_factor, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32*self.wide_factor, 32*self.wide_factor, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*self.wide_factor, 64*self.wide_factor, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64*self.wide_factor, 64*self.wide_factor, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64*self.wide_factor, 64*self.wide_factor, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*self.wide_factor, 128*self.wide_factor, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128*self.wide_factor, 128*self.wide_factor, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128*self.wide_factor, 128*self.wide_factor, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128*self.wide_factor, 128*self.wide_factor, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128*self.wide_factor),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16*self.wide_factor,
            'x_conv2': 32*self.wide_factor,
            'x_conv3': 64*self.wide_factor,
            'x_conv4': 128*self.wide_factor
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features_after_scn'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict


class VoxelWideResBackBone_L8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.wide_factor = model_cfg.WIDE_FACTOR
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16*self.wide_factor, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16*self.wide_factor),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16*self.wide_factor, 16*self.wide_factor, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16*self.wide_factor, 16*self.wide_factor, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16*self.wide_factor, 16*self.wide_factor, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16*self.wide_factor, 16*self.wide_factor, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*self.wide_factor, 32*self.wide_factor, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32*self.wide_factor, 32*self.wide_factor, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32*self.wide_factor, 32*self.wide_factor, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32*self.wide_factor, 32*self.wide_factor, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32*self.wide_factor, 32*self.wide_factor, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*self.wide_factor, 64*self.wide_factor, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64*self.wide_factor, 64*self.wide_factor, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64*self.wide_factor, 64*self.wide_factor, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64*self.wide_factor, 64*self.wide_factor, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64*self.wide_factor, 64*self.wide_factor, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*self.wide_factor, 128*self.wide_factor, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128*self.wide_factor, 128*self.wide_factor, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128*self.wide_factor, 128*self.wide_factor, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128*self.wide_factor, 128*self.wide_factor, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128*self.wide_factor, 128*self.wide_factor, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128*self.wide_factor, 128*self.wide_factor, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128*self.wide_factor),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16*self.wide_factor,
            'x_conv2': 32*self.wide_factor,
            'x_conv3': 64*self.wide_factor,
            'x_conv4': 128*self.wide_factor
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict