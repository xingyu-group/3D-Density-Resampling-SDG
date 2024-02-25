import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from pcdet.utils import common_utils


class pointnet2_perceptual_backbone(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel):
        super(pointnet2_perceptual_backbone, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.4, 0.8, 1.2], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.8, 1.2, 2.4], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

    def trans_feat_from_sp_to_pts(self, feat_1, feat_2, indices_shared, batch_size):
        '''
        Input:
            indices_shared: input points position data, [N, (batch_idx + x + y + z)]
            feat_*: input points data, [N, num_channels]
        Return:
            feat_*_trans: points data, [B, n_pts, num_channels]
            indice_trans: coor_indice, [B, n_pts, 3]
        '''

        feat_1_multi_img = []
        feat_2_multi_img = []
        indice_multi_img = [] 
        n_points_max = 0

        for i in range(batch_size):
            batch_mark = indices_shared[:,0] == i

            n_points = batch_mark.sum().item()
            if n_points > n_points_max:
                n_points_max = n_points

            feat_1_multi_img.append(feat_1[batch_mark])
            feat_2_multi_img.append(feat_2[batch_mark])
            indice_multi_img.append(indices_shared[batch_mark, 1:])

        feat_1_multi_img_aligned = []
        feat_2_multi_img_aligned = []
        indice_multi_img_aligned = [] 

        for feat_item in feat_1_multi_img:
            if feat_item.shape[0] < n_points_max:
                zeros_filled = torch.zeros((n_points_max - feat_item.shape[0], feat_item.shape[1]), dtype=feat_item.dtype, device=feat_item.device)
                feat_item = torch.cat((feat_item, zeros_filled), dim=0)
            feat_1_multi_img_aligned.append(feat_item.view(1, feat_item.shape[0], feat_item.shape[1]))
        
        for feat_item in feat_2_multi_img:
            if feat_item.shape[0] < n_points_max:
                zeros_filled = torch.zeros((n_points_max - feat_item.shape[0], feat_item.shape[1]), dtype=feat_item.dtype, device=feat_item.device)
                feat_item = torch.cat((feat_item, zeros_filled), dim=0)
            feat_2_multi_img_aligned.append(feat_item.view(1, feat_item.shape[0], feat_item.shape[1]))
        
        for coor_item in indice_multi_img:
            coor_item = common_utils.get_voxel_centers(
                coor_item,
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            if coor_item.shape[0] < n_points_max:
                zeros_filled = torch.zeros((n_points_max - coor_item.shape[0], coor_item.shape[1]), dtype=coor_item.dtype, device=feat_item.device)
                coor_item = torch.cat((coor_item, zeros_filled), dim=0)
            indice_multi_img_aligned.append(coor_item.view(1, coor_item.shape[0], coor_item.shape[1]))

        feat_1_trans = torch.cat(feat_1_multi_img_aligned, dim=0)
        feat_2_trans = torch.cat(feat_2_multi_img_aligned, dim=0)
        coord_trans = torch.cat(indice_multi_img_aligned, dim=0)

        return feat_1_trans, feat_2_trans, coord_trans


    def forward(self, feat_1, feat_2, indices_shared, batch_size):
        '''
        Input:
            indices_shared: input points position data, [N, (batch_idx + x + y + z)]
            feat_*: input points data, [N, 3]
        '''
        assert feat_1.shape[1] == feat_2.shape[1] == 3


        point_feat_1, point_feat_2, xyz = self.trans_feat_from_sp_to_pts(feat_1, feat_2, indices_shared, batch_size)
        
        point_feat_1 = point_feat_1.permute(0, 2, 1) # [B, D, N]
        point_feat_2 = point_feat_2.permute(0, 2, 1) # [B, D, N]
        xyz = xyz.permute(0, 2, 1) # [B, 3, N]

        # print('maximum and minimum coords on 3 dimensions are', torch.max(xyz[0], dim=1)[0], torch.min(xyz[0], dim=1)[0])
        

        '''
        Input:
            xyz: input points position data, [B, C, N]
            point_feat_*: input points data, [B, D, N]
        '''

        layer1_xyz_1, layer1_points_1 = self.sa1(xyz, point_feat_1)
        layer2_xyz_1, layer2_points_1 = self.sa2(layer1_xyz_1, layer1_points_1)
        layer3_xyz_1, layer3_points_1 = self.sa3(layer2_xyz_1, layer2_points_1)

        layer1_xyz_2, layer1_points_2 = self.sa1(xyz, point_feat_2, layer1_xyz_1)
        layer2_xyz_2, layer2_points_2 = self.sa2(layer1_xyz_2, layer1_points_2, layer2_xyz_1)
        layer3_xyz_2, layer3_points_2 = self.sa3(layer2_xyz_2, layer2_points_2)

        feats_multi_layer_1 = {
            'layer1_out': layer1_points_1,
            'layer2_out': layer2_points_1,
            'layer3_out': layer3_points_1,
        }

        feats_multi_layer_2 = {
            'layer1_out': layer1_points_2,
            'layer2_out': layer2_points_2,
            'layer3_out': layer3_points_2,
        }

        return feats_multi_layer_1, feats_multi_layer_2


    def get_loss(self, multi_feat_1, multi_feat_2):

        total_loss = 0
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

        for key in list(multi_feat_1.keys()):
            # multi_feat_1[key]: [B, D', S]
            total_loss += (1 - cos_sim(multi_feat_1[key], multi_feat_2[key])).mean()
        return total_loss


