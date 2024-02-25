import numpy as np
from numpy import random
import numba

# quadratic-polynomial fitting
def interp_3D(points, inp_rate):
    ''' 
    Args: 
        origin points: N x (3+C) 
        inp_rate: float
    Return:
        new points
    '''
    N, C =points.shape
    # select dimension with lowest variance as the target
    z_idx = np.argmin( np.max(points[:,:3],axis=0) - np.min(points[:,:3],axis=0))
    if z_idx ==0:
        x_idx, y_idx = 1, 2
    elif z_idx ==1:
        x_idx, y_idx = 2, 0
    else:
        x_idx, y_idx = 0, 1
    
    def poly_2D(x, y):
        ''' 
        Input: 
            N x 2: (x, y)
        Output:
            N x 5: (x, y, x^2, y^2, xy)        
        '''
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1),   # x, y
                          (x**2).reshape(-1,1), (y**2).reshape(-1,1),   # x^2, y^2 
                          (x*y).reshape(-1,1)))   # xy 
        
    X_=np.hstack((np.ones((N,1)), poly_2D(points[:,x_idx].reshape(-1,1), points[:,y_idx].reshape(-1,1))))
    Y_=points[:,z_idx].reshape(-1,1)
    W =np.linalg.inv(X_.T @ X_)@X_.T@Y_
    
    N_new = int(N * inp_rate)
    x_new = np.random.randn(N_new) * np.std(points[:,x_idx]) + np.mean(points[:,x_idx])
    y_new = np.random.randn(N_new) * np.std(points[:,y_idx]) + np.mean(points[:,y_idx])
    
    X_new = np.hstack((np.ones((N_new,1)), poly_2D(x_new.reshape(-1,1), y_new.reshape(-1,1))))
    z_new = X_new @ W.reshape(-1,1)
    
    if z_idx ==0:
        points_new = np.hstack((z_new.reshape(-1,1), x_new.reshape(-1,1), y_new.reshape(-1,1)))
    elif z_idx ==1:
        points_new = np.hstack((y_new.reshape(-1,1), z_new.reshape(-1,1), x_new.reshape(-1,1)))
    else:
        points_new = np.hstack((x_new.reshape(-1,1), y_new.reshape(-1,1), z_new.reshape(-1,1)))
    
    # fill up the reflectivity of new points with that of the nearest point
    r_new = np.zeros(N_new*(C-3)).reshape(-1, (C-3))
    for i in range(N_new):
        dist = np.sum((points[:,:3] - points_new[i,:3])**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, 0, axis=0)[0] 
        r_new[i,:] = points[idx.squeeze(),3:]
    points_new = np.hstack((points_new, r_new))
    
    return points_new

# @numba.jit()
def car2sph_pc(pointcloud):
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
    if pointcloud.shape[1] > 3:
        pointcloud_trans = np.hstack((pointcloud_trans, pointcloud[:,3:].reshape(-1,(pointcloud.shape[1]-3))))
    return pointcloud_trans
    
# @numba.jit()
def sph2car_pc(pointcloud):
    '''
    args:
        pointcloud: N x (3 + c) : r, phi, and theta
    return:
        pointcloud: N x (3 + c) : x, y, and z
    '''
    x = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.cos(pointcloud[:,1])
    y = pointcloud[:,0]*np.sin(pointcloud[:,2])*np.sin(pointcloud[:,1])
    z = pointcloud[:,0]*np.cos(pointcloud[:,2])

    pointcloud_trans = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
    if pointcloud.shape[1] > 3:
        pointcloud_trans = np.hstack((pointcloud_trans, pointcloud[:,3:].reshape(-1,(pointcloud.shape[1]-3))))
    return pointcloud_trans

def get_64bins(data):    
    ## filter first
    N_b=1000
    array_den, array_bin = np.histogram(data, bins=N_b, density=True)
    bin_head = array_bin[:-1]
    bin_end = array_bin[1:]
    bin_step = np.sum(bin_head-bin_end)/N_b
    thr_bin = 1/N_b/4 # The 1/N_b is the average density  
    bin_head_filtered = bin_head[array_den>thr_bin]
    bin_end_filtered = bin_end[array_den>thr_bin]
    range_filtered = [np.min(bin_head_filtered), np.max(bin_end_filtered)]
    #filtered points
    data = data[(range_filtered[0]<data) & (data<range_filtered[1])]
    
    ##get bins
    arr_bin = np.linspace(np.min(data), np.max(data), 65)
    bins = []
    for i in range(64):
        bins.append([arr_bin[i], arr_bin[i+1]])    
    return bins 

# @numba.jit()
def get_N_bins(data, N_bin, thr_bin=5):    
    #filtered points
    std_data = np.std(data)
    mean_data = np.mean(data)
    range_filtered = [(mean_data - 3.1*std_data), (mean_data + 3.1*std_data)] 
    data_filted = data[(data>range_filtered[0]) & (data<range_filtered[1])]
    
    ##get bins
    arr_bin = np.linspace(np.min(data_filted), np.max(data_filted), N_bin+1)
    bins = []
    for i in range(N_bin):
        bins.append([arr_bin[i], arr_bin[i+1]])    
    return bins