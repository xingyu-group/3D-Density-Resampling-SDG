from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
import torch 
from torch import nn


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    # print('******* fps_idx',fps_idx.shape)
    assert fps_idx.shape[0] ==1 
    fps_data = data[:,fps_idx.squeeze(0).long(),:]
    # print('******* fps_data',fps_data.shape)
    # fps_data = pointnet2_utils.grouping_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # # normalize
        # neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

def knn_mask(points, prob, batch_size, center_num):
    '''
        input: N 4 (batch_idx + xyz)
        ---------------------------
        output: N_mask 4 (batch_idx + xyz)
    '''
    points_new = []
    for i_bat in range(batch_size):
        pts_cur_bat = points[points[:,0]==i_bat, 1:] # N x 3
        pts_cur_bat = pts_cur_bat.unsqueeze(0)# 1 x N x 3
        pts_num = pts_cur_bat.shape[1]
        group_size = int(pts_num // center_num)
        group_operator = Group(num_group=center_num, group_size=group_size)
        pts_grouped, _ = group_operator.forward(pts_cur_bat) # 1 x G x M x 3
        assert pts_grouped.shape[1] == center_num
        sampled_pos = torch.rand(center_num).to(pts_cur_bat.device)
        save_mask = sampled_pos < prob
        save_idx = torch.arange(center_num).to(pts_cur_bat.device)[save_mask]
        pts_sampled = pts_grouped[:,save_idx,:,:].view(-1, 3) # N_sampled x 3
        filled_i_bat = torch.zeros((pts_sampled.shape[0], 1), dtype=pts_sampled.dtype, device=pts_sampled.device) + i_bat
        points_new.append(torch.cat((filled_i_bat, pts_sampled), dim=1))

    return torch.cat(points_new, dim=0)




