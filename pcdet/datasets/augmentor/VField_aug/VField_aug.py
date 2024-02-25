import torch
import numpy as np
import random
from sklearn.cluster import KMeans
from pcdet.utils import box_utils
from math import atan2
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class VField():
    def __init__(self, classes_interest, VF_file):
        # default box
        assert isinstance(classes_interest, list), 'classes of objects should in a list.'
        self.affecting_objects =  classes_interest
        self.step = 0.2
        self.w_B0 = 1.8 + 0.00001
        self.h_B0 = 1.6 + 0.00001
        self.l_B0 = 4.6 + 0.00001
        self.N_w= int(self.w_B0 / self.step)
        self.N_h= int(self.h_B0 / self.step)
        self.N_l= int(self.l_B0 / self.step)
        self.init_unifrom = 0.01
        # variability of perturbations
        self.N = 6          # number of vectors for each G
        self.G = 12         # groups of rotation
        # other settings
        self.k = 2          # nearest vectors for perturbing one point
        # print(self.N_w, self.N_h, self.N_l)
        vectors_init = torch.load(VF_file).detach().cpu().numpy()
        self.vectors = vectors_init
        self.boundary_vector = 0.3
        # self.cluster_ori = self.orientation_clustering('angle_list.npy')
        
    def generate_vector_coors_in_scene(self, gt_boxes_lidar):
        l = gt_boxes_lidar[3]
        w = gt_boxes_lidar[4]
        h = gt_boxes_lidar[5]
        vector_coors_origin = self.rescale_box(w, h, l)
        # vector_coors_shape = vector_coors_origin.shape
        vector_coors_origin = vector_coors_origin.reshape(-1,3)
        vector_coors = self.transform_coordinate_to_scene(vector_coors_origin, gt_boxes_lidar[6], gt_boxes_lidar[0:3])
        return vector_coors

    
    def rescale_box(self, w, h, l):
        step_w = w / self.N_w 
        step_h = h / self.N_h 
        step_l = l / self.N_l 
        w_lin = np.linspace(step_w/2, w-step_w/2, self.N_w)
        h_lin = np.linspace(step_h/2, h-step_h/2, self.N_h)
        l_lin = np.linspace(step_l/2, l-step_l/2, self.N_l)
        w_coors, h_coors, l_coors = np.meshgrid(w_lin, h_lin, l_lin, indexing='ij')
        vector_coors = np.stack((l_coors - l/2, w_coors - w/2, h_coors - h/2), axis=3)
        return vector_coors

    def transform_coordinate_to_scene(self, points,  angle, loc):
        """
        Args:
            points: N x 3 (torch.tensor)
            gt_boxes_lidar: 7 
        Returns:
            points with lidar coordinates
        """       
        # rotate 
        points = self.rotate_pts_along_z(points, angle)
        # shift
        points = points + loc
        return points

    def rotate_pts_along_z(self, points, angle):
        """
        Args:
            points: N x 3
            angle: angle along z-axis, angle increases x ==> y
        Returns:
        """
        cosa = np.cos(angle)
        sina = np.sin(angle)
        rot_matrix = np.array(
            [cosa,  sina, 0.0,
            -sina, cosa, 0.0,
            0.0,   0.0,  1.0], dtype=points.dtype).reshape(3, 3)
        # print('Start dot...')
        points_rot = np.dot(points, rot_matrix) 
        # print('Succeed!')   
        return points_rot


    def identify_orientation_group(self, direction, loc_x, loc_y):
        orientation = atan2(loc_y, loc_x)
        relative_orientation = orientation - direction
        assert (-2*np.pi <= relative_orientation <= 2*np.pi)
        if relative_orientation < 0:
            relative_orientation += 2 * np.pi
        id_angle_group = int(relative_orientation/(2*np.pi/self.G))
        return id_angle_group
    
    def perturbation(self, points, boxes_gt, names_gt):
        batch_size = boxes_gt.shape[0]
        gt_boxes = boxes_gt
        N_obj = gt_boxes.shape[0]
        gt_boxes_np = gt_boxes[:,:7]
        gt_box_corners = box_utils.boxes_to_corners_3d(gt_boxes_np)

        for i_obj in range(N_obj):
            if names_gt[i_obj] in self.affecting_objects: 
                # preparing vectors
                vector_coors = self.generate_vector_coors_in_scene(gt_boxes[i_obj,:]) # 1656 x 3
                n = random.choice(range(self.N))
                g = self.identify_orientation_group(gt_boxes[i_obj, 6], gt_boxes[i_obj, 0], gt_boxes[i_obj, 1])
                vectors = self.vectors[self.affecting_objects.index(names_gt[i_obj]), g, n, :, :, :, :].reshape(-1,3) # 1656 x 3
                # print(vectors)

                # preparing points in the box
                flag = box_utils.in_hull(points[:, 0:3], gt_box_corners[i_obj])
                points_in_box = points[flag, :]  
                # print(points_in_box)   

                if sum(flag)>0:
                    # for each point of points in the box
                    for i_pt in range(sum(flag)):
                        pt = points_in_box[i_pt, 0:3]
                        # obtain k nearest vectors
                        dis_ =  np.linalg.norm(vector_coors - pt, axis=1)  
                        idx = np.argsort(dis_)  
                        idx = idx[:self.k]  
                        dis_conscending = dis_[idx]
                        # dis_reciprocal_sum = (1/dis_conscending).sum()
                        # essential part: perturbing point
                        pert = 0
                        for i_k in range(self.k):
                            pert += np.dot(vectors[idx[i_k],:], pt) / np.dot(pt, pt) * pt / self.k  # * (1 / dis_conscending[i_k]) / dis_reciprocal_sum
                        pt += pert
                        points_in_box[i_pt, 0:3] = pt 
                # print(points_in_box)
                points[flag, :] = points_in_box
        return points


