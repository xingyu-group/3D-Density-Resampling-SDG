import numpy as np
from numpy import random
import numba

### tools ##
''' Fill vacant intesity of new pc if new points exist， by KNN-distance clustering '''
def fill_intensity(pc, pc_cor_xyz, n_b=5):
    # pc：N x 4(xyz+intensity)
    # pc_cor_xyz: N+d x 3(xyz)
    N, _ = pc.shape
    N_all, _ = pc_cor_xyz.shape
    if N == N_all:
        return np.hstack((pc_cor_xyz, pc[:, 3].reshape([-1,1])))
    else:
        pc_cor = np.hstack((pc_cor_xyz, np.vstack((pc[:,3].reshape([-1,1]), np.zeros((N_all-N,1))))))
        for i in range(N, N_all):     
            dist = np.sum((pc[:,:3] - pc_cor[i,:3])**2, axis=1, keepdims=True)
            idx = np.argpartition(dist, n_b, axis=0)[:n_b]
            pc_cor[i,3] = np.sum(pc_cor[idx,3])/n_b
        return pc_cor
    
''' Delete outliers by zscore'''
def del_outlier_axis(data, num=10):
    # by samples
    z_abs = np.abs(data - data.mean()) / (data.std())
    list_ = []
    for _ in range(num):
        index_max = np.argmax(z_abs)
        list_.append(index_max)
        z_abs[index_max] = 0
    return np.delete(data, list_)


'''convert cartesian coordinates into spherical ones'''
@numba.jit()
def car2sph_pc(pointcloud):
    '''
    args:
        points: N x 3 : x, y, and z
    return:
        points: N x 3 : r, phi, and theta
    '''
    r_sph = np.sqrt(pointcloud[:,0]**2 + pointcloud[:,1]**2 + pointcloud[:,2]**2)
    phi = np.arctan2(pointcloud[:,1],pointcloud[:,0])
    the = the = np.arccos(pointcloud[:,2]/r_sph)
    return np.hstack((r_sph.reshape(-1,1), phi.reshape(-1,1), the.reshape(-1,1)))

'''convert spherical coordinates into cartesian ones'''
@numba.jit()
def sph2car_pc(pointcloud):
    '''
    args:
        points: N x 3 : r, phi, and theta
    return:
        points: N x 3 : x, y, and z
    '''
    x = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.cos(pointcloud[:,1])
    y = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.sin(pointcloud[:,1])
    z = pointcloud[:,0]*np.cos(pointcloud[:,2])
    return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

def mean_fill_except_xyz(pc_ref, points):
    '''
    args:
        pc_ref: N x (xyz + C)
        points: N x (xyz + C)
    return:
        points: N x (xyz + C)
    '''
    points[:,3:] = np.mean(pc_ref[:,3:], axis=0).reshape(1,-1)
    return points
