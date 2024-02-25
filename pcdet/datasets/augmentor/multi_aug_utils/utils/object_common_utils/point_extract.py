import numpy as np
# from pathlib import Path
from .utils import box_utils


def Lidar_to_Max2(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7 
    Returns:
        points normalized to max-2 unit square box: N x 3+C
    """
    # shift
    points[:,:3]=points[:,:3]-gt_boxes_lidar[:3]
    # normalize to 2 units 
    points[:,:3]=points[:,:3]/np.max(gt_boxes_lidar[3:6])*2
    # reversely rotate 
    points=rotate_pts_along_z(points, -gt_boxes_lidar[6])
    
    return points

def Max2_to_Lidar(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7 
    Returns:
        points denormalized to lidar coordinates
    """
       
    # rotate 
    points=rotate_pts_along_z(points, gt_boxes_lidar[6])
    # denormalize to lidar
    points[:,:3]=points[:,:3]*np.max(gt_boxes_lidar[3:6])/2
    # shift
    points[:,:3]=points[:,:3]+gt_boxes_lidar[:3]
    
    return points
    

def rotate_pts_along_z(points, angle):
    """
    Args:
        points: (N x 3 + C) narray
        angle: angle along z-axis, angle increases x ==> y
    Returns:

    """
    N, C = points.shape
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa,  sina, 0.0,
         -sina, cosa, 0.0,
         0.0,   0.0,  1.0]).reshape(3, 3)
    points_rot = np.matmul(points[ :, 0:3], rot_matrix)
    points_rot = np.hstack((points_rot, points[ :, 3:].reshape(N,-1)))
    
    return points_rot

def flag_in_boxe(pointcloud, gt_boxes):
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes)
    flags = []
    for k in range(gt_boxes.shape[0]):
        flag = box_utils.in_hull(pointcloud[:, :3], corners_lidar[k])
        flags += [flag]
    return flags

# def process_1_scan(root_path, sample_idx):
#     root_split_path=Path(root_path)
#     info = {}    
#     calib = get_calib(root_split_path, sample_idx)    
#     obj_list = get_label(root_split_path, sample_idx)
#     info['name'] = np.array([obj.cls_type for obj in obj_list])    
#     num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
#     info['num_objs'] = num_objects
#     loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)[:num_objects]
#     dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])[:num_objects]
#     rots = np.array([obj.ry for obj in obj_list])[:num_objects]
#     loc_lidar = calib.rect_to_lidar(loc)
#     l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
#     loc_lidar[:, 2] += h[:, 0] / 2
#     gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
#     info['gt_boxes_lidar'] = gt_boxes_lidar    
#     points = get_lidar(root_split_path, sample_idx)    
#     pts_fov = points
#     corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
#     pts_in_gt = []
#     for k in range(num_objects):
#         flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
#         pts_in_gt += [flag]
#     info['pts_in_gt'] = pts_in_gt
#     return info

# constrained to max2 with the origin ratio
def normalize(points):
    """
    Args:
        points: N x 3+C 
    Returns:
        limit points to max-2 unit square box: N x 3+C
    """
    if points.shape[0] != 0:
        indicator = np.max(np.abs(points[:,:3]))
        if indicator>1:
            points[:,:3] = points[:,:3]/indicator
    return points

# constrained to gt
def normalize_gt(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3 
    Returns:
        limit points to gt: N x 3+C
    """    
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        for i in range(3):
            indicator = np.max(np.abs(points[:,i])) / box_boundary_normalized[i]
            if indicator > 1: 
                points[:,i] /= indicator 
    return points

# constrained to gt with the origin ratio
def normalize_max2(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3 
    Returns:
        limit points to gt with the origin ratio: N x 3+C
    """    
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        indicators = np.zeros(3)
        for i in range(3):
            indicators[i] = np.max(np.abs(points[:,i]))/box_boundary_normalized[i] if np.max(np.abs(points[:,i])) > box_boundary_normalized[i] else 1.0
        points[:,:3]/=np.max(indicators)  

    return points



# constrained to gt with the origin ratio
def normalize_max2_2(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C 
    Returns:
        limit points to gt with the origin ratio: N x 3+C
        scalling factor
    """
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        indicators = np.zeros(3)
        for i in range(3):
            indicators[i] = np.max(np.abs(points[:,i]))/box_boundary_normalized[i] if np.max(np.abs(points[:,i])) > box_boundary_normalized[i] else 1.0
        indicator_rat = np.max(indicators)
        points[:,:3]/=indicator_rat
        # points touch the ground (almost)
        points[:,2]=points[:,2]-box_boundary_normalized[2]*(1.0-1.0/indicator_rat)
        return points, indicator_rat
    else:
        return points, 1.