import numpy as np
from .utils.scene_noise_utils import car2sph_pc, sph2car_pc
from .utils.scene_density_utils import car2sph_pc as car2sph_pc_with_int
from .utils.scene_density_utils import sph2car_pc as sph2car_pc_with_int
from .utils.object_common_utils.point_extract import Lidar_to_Max2, Max2_to_Lidar, normalize, flag_in_boxe
from .utils.object_common_utils.point_extract import normalize_gt as normalize2
from .utils.scene_density_utils import get_N_bins
import cupy as cp
import numba

'''
Add Gaussian noise to point cloud radially (sensor-based)
'''
def gaussian_noise_radial_scene(pointcloud, severity, gt_boxes=None):
    N, C = pointcloud.shape
    c = [0.04, 0.06, 0.08, 0.10, 0.12][severity-1]
    jitter = np.random.normal(size=N) * c
    new_pc = car2sph_pc(pointcloud)
    new_pc[:, 0] += jitter * np.sqrt(3)
    pointcloud[:, :3] = sph2car_pc(new_pc)
    pointcloud = pointcloud.astype('float32')
    return pointcloud


''' Cutout several part in the point cloud with reflectivities '''
def cutout_scene(pointcloud, severity, gt_boxes=None):
    N, _ = pointcloud.shape
    c = [(N//200,20), (N//150,20), (N//100,20), (N//80,20), (N//60,20)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i,:3]
        dist = np.sum((pointcloud[:,:3] - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    return pointcloud


'''
Sensor-based beam missing, globally (delete beams)
'''
def beam_del_scene(pointcloud, severity, gt_boxes=None):
    N, _ = pointcloud.shape
    c = [N//30, N//10, N//5, N//3, N//2][severity-1]
    idx_del = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx_del, axis=0)
    
    return pointcloud

''' locally decrease point density on object-level pointcloud'''
def density_dec_obj(pointcloud, severity, gt_boxes):        
    N, C = pointcloud.shape
    c = [(1,30), (2,30), (3,30), (4,30), (5,30)][severity-1]
    flags_obj = flag_in_boxe(pointcloud, gt_boxes)
    idx_add = np.zeros(pointcloud.shape[0], dtype=bool)
    pts_add = []
    for i in range(gt_boxes.shape[0]):
        idx_pts = flags_obj[i]
        if sum(idx_pts) > (c[0]*int((3/4) * c[1])+5):
            pts_obj = pointcloud[idx_pts]
            # convert to max-2
            pts_obj_max2 = Lidar_to_Max2(pts_obj, gt_boxes[i])
            points = pts_obj_max2            
            ## density decreasing on object                  
            for _ in range(c[0]):
                i_pts = np.random.choice(points.shape[0],1)
                picked = points[i_pts, :3]
                dist = np.sum((points[:, :3] - picked)**2, axis=1, keepdims=True)
                N_near = min(c[1], points.shape[0]-1)
                idx = np.argpartition(dist, N_near, axis=0)[:N_near]                
                idx_2 = np.random.choice(N_near,int((3/4) * c[1]),replace=False)
                idx = idx[idx_2]
                points = np.delete(points, idx.squeeze(), axis=0)    
            pts_obj_max2_crp = points   
            # convert to Lidar
            pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, gt_boxes[i])
            idx_add = idx_add | np.array(idx_pts)
            pts_add.append(pts_obj_crp)
        else:
            idx_add = idx_add | np.array(idx_pts)
    pointcloud = np.delete(pointcloud, idx_add, axis=0)       
    if len(pts_add)>0: # if pointcloud is indeed processed        
        pointcloud = np.concatenate((pointcloud, np.concatenate(pts_add, axis=0)), axis=0)
    return pointcloud


''' Shear x and y, except z'''
def shear_obj(pointcloud, severity, gt_boxes):
    N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    flags_obj = flag_in_boxe(pointcloud, gt_boxes)
    # print(gt_boxes)

    for i in range(gt_boxes.shape[0]):
        idx_pts = flags_obj[i]
        pts_obj = pointcloud[idx_pts]
        if sum(idx_pts) == 0:
            continue
        # convert to max-2
        # print(pts_obj.shape, gt_boxes[i].shape)
        pts_obj_max2 = Lidar_to_Max2(pts_obj, gt_boxes[i])
        ## shear        
        a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        matrix = np.array([1, a, 0,
                           b, 1, 0,
                           d, e, 1]).reshape(3,3)      
        new_pc = np.matmul(pts_obj_max2[:,:3], matrix).astype('float32')
        
        pts_obj_max2[:,:3] = new_pc
        pts_obj_max2_crp = normalize2(pts_obj_max2, gt_boxes[i][3:6])    
        pts_obj_crp = Max2_to_Lidar(pts_obj_max2_crp, gt_boxes[i])
        pointcloud[idx_pts]=pts_obj_crp
                
    return pointcloud


'''
Sensor-based layer increasing, globally
'''
def layer_interp_scene(pointcloud, severity, gt_boxes=None, bin_num=64, shift_origin=[0, 0, -1.7], phi_thresh=0.03, random_choice_layer_with_step=False):
    N, _ = pointcloud.shape
    c = [1, 2, 3, 4, 5][severity-1]

    new_pointcloud_list = []

    pointcloud[:,:3] += np.array(shift_origin)
    pointcloud_sph=car2sph_pc_with_int(pointcloud)
    bins=get_N_bins(pointcloud_sph[:,2], bin_num)
    
    for i in range(bin_num - 1):
        layer_mask_cur = (pointcloud_sph[:,2]>=bins[i][0]) & (pointcloud_sph[:,2]<=bins[i][1]) 
        layer_mask_nxt = (pointcloud_sph[:,2]>=bins[i+1][0]) & (pointcloud_sph[:,2]<=bins[i+1][1]) 

        if layer_mask_cur.sum()>0 and layer_mask_nxt.sum()>0:
            phi_del = np.abs(pointcloud_sph[layer_mask_cur,1].reshape((-1,1)) - pointcloud_sph[layer_mask_nxt,1].reshape((1,-1)))
            idx_adj = np.argmin(phi_del,1)
            phi_del_min = np.min(phi_del,1)
            mask = phi_del_min < phi_thresh
            if mask.sum()>0:
                pc_cur = pointcloud_sph[layer_mask_cur][mask]
                pc_nxt = pointcloud_sph[layer_mask_nxt][idx_adj[mask]]
                if c ==1:
                    new_pc = (pc_cur + pc_nxt)/2
                elif c ==2: 
                    new_pc_1 = pc_cur/3 + pc_nxt*2/3 
                    new_pc_2 = pc_cur*2/3 + pc_nxt/3 
                    new_pc = np.concatenate((new_pc_1, new_pc_2),0)
                else:
                    raise NotImplementedError('Severity not defined.')

                new_pc = sph2car_pc_with_int(new_pc)
                new_pc[:,:3] -= np.array(shift_origin)
                new_pointcloud_list.append(new_pc) 
    
    pointcloud[:,:3] -= np.array(shift_origin)
    new_pointcloud_list = [pointcloud] + new_pointcloud_list
    pointcloud = np.concatenate(new_pointcloud_list,0)
    return pointcloud

'''
Sensor-based layer increasing, globally
'''
def layer_interp_scene_x2(pointcloud, severity, gt_boxes=None, bin_num=64, shift_origin=[0, 0, -1.7], phi_thresh=0.03, random_choice_layer_with_step=False):
    N, _ = pointcloud.shape
    c = [1, 2, 3, 4, 5][severity-1]

    new_pointcloud_list = []

    pointcloud[:,:3] += np.array(shift_origin)
    pointcloud_sph=car2sph_pc_with_int(pointcloud)
    bins=get_N_bins(pointcloud_sph[:,2], bin_num)
    
    for i in range(bin_num - 1):
        layer_mask_cur = (pointcloud_sph[:,2]>=bins[i][0]) & (pointcloud_sph[:,2]<=bins[i][1]) 
        layer_mask_nxt = (pointcloud_sph[:,2]>=bins[i+1][0]) & (pointcloud_sph[:,2]<=bins[i+1][1]) 

        if layer_mask_cur.sum()>0 and layer_mask_nxt.sum()>0:
            phi_del = np.abs(pointcloud_sph[layer_mask_cur,1].reshape((-1,1)) - pointcloud_sph[layer_mask_nxt,1].reshape((1,-1)))
            idx_adj = np.argmin(phi_del,1)
            phi_del_min = np.min(phi_del,1)
            mask = phi_del_min < phi_thresh
            if mask.sum()>0:
                pc_cur = pointcloud_sph[layer_mask_cur][mask]
                pc_nxt = pointcloud_sph[layer_mask_nxt][idx_adj[mask]]

                new_pc = (pc_cur + pc_nxt)/2

                new_pc = sph2car_pc_with_int(new_pc)
                new_pc[:,:3] -= np.array(shift_origin)
                new_pointcloud_list.append(new_pc) 
    
    pointcloud[:,:3] -= np.array(shift_origin)
    new_pointcloud_list = [pointcloud] + new_pointcloud_list
    pointcloud = np.concatenate(new_pointcloud_list,0)
    return pointcloud


'''
Sensor-based layer missing, globally
'''
# @numba.njit(parallel=True)
def layer_del_scene(pointcloud, severity, gt_boxes=None, bin_num=64, shift_origin=[0, 0, -1.7], random_choice_layer_with_step=False, random_dropout_prob=0.99):
    N, _ = pointcloud.shape
    c = [1, 2, 3, 4, 6, 8][severity-1]

    pointcloud_shifted = pointcloud[:,:3] + np.array(shift_origin)
    pointcloud_sph=car2sph_pc_local(pointcloud_shifted)
    # pointcloud_sph[:,1:3]=pointcloud_sph[:,1:3]/np.pi*180
    bins=get_N_bins_local(pointcloud_sph[:,2], bin_num)
    
    n_pts_sampled = 0
    n_trial = 5
    while (n_pts_sampled<500) and (n_trial!=0):
        idx_save = np.zeros(N, dtype=np.bool_)
        start_layer_idx = np.random.choice(np.arange(c)) if random_choice_layer_with_step else 0
        for i in range(start_layer_idx, bin_num, c):
            save_i = i
            if random_choice_layer_with_step:
                save_i = i+ np.random.choice(np.arange(c))
                save_i = (bin_num-1) if save_i>(bin_num-1) else save_i         
            temp_idx= (pointcloud_sph[:,2]>bins[save_i][0]) & (pointcloud_sph[:,2]<bins[save_i][1])
            idx_save = idx_save|temp_idx
        pointcloud = pointcloud[idx_save]
        n_trial -= 1
        n_pts_sampled = pointcloud.shape[0]
    pointcloud = pointcloud[np.random.choice(pointcloud.shape[0],int(pointcloud.shape[0]*random_dropout_prob),replace=False),:]
    return pointcloud


# @numba.jit()
def car2sph_pc_local(pointcloud):
    '''
    args:
        pointcloud: N x (3 + c) : x, y, and z
    return:
        pointcloud: N x (3 + c) : r, phi, and theta
    '''
    r_sph = np.linalg.norm(pointcloud[:,:3], axis=1)
    phi = np.arctan2(pointcloud[:,1],pointcloud[:,0])
    the = np.arccos(pointcloud[:,2]/r_sph)

    pointcloud_trans = np.hstack((r_sph.reshape(-1,1), phi.reshape(-1,1), the.reshape(-1,1)))
    return pointcloud_trans
    



# @numba.jit()
def get_N_bins_local(data, N_bin, thr_bin=5):    
    std_data = np.std(data)
    mean_data = np.mean(data)
    # remove outliers
    range_filtered = np.array([(mean_data - 3.1*std_data), (mean_data + 3.1*std_data)])
    data_filted = data[(data>range_filtered[0]) & (data<range_filtered[1])]
    
    ##get bins
    arr_bin = np.linspace(np.min(data_filted), np.max(data_filted), N_bin+1)
    bins = np.zeros((N_bin, 2), dtype=arr_bin.dtype)
    for i in range(N_bin):
        bins[i, 0], bins[i, 1] = arr_bin[i], arr_bin[i+1]   
    return bins