import numpy as np

''' interpolation by Least-Square '''
def interp_3D(points, inp_rate):
    ''' 
    Args: 
        origin points: N x (3+C) 
        inp_rate: float
    Return:
        new points
    '''
    N,_=points.shape
    z_idx = np.argmin( np.max(points[:,:3],axis=0) - np.min(points[:,:3],axis=0))
    if z_idx ==0:
        x_idx, y_idx = 1, 2
    elif z_idx ==1:
        x_idx, y_idx = 2, 0
    else:
        x_idx, y_idx = 0, 1
        
    X_=np.hstack((np.ones((N,1)), points[:,x_idx].reshape(-1,1), points[:,y_idx].reshape(-1,1)))
    Y_=points[:,z_idx].reshape(-1,1)
    
    # Least-Square closed-form solution
    W =np.linalg.inv(X_.T @ X_)@X_.T@Y_
    
    N_new = int(N * inp_rate)
    x_new = np.random.randn(N_new) * np.std(points[:,x_idx]) + np.mean(points[:,x_idx])
    y_new = np.random.randn(N_new) * np.std(points[:,y_idx]) + np.mean(points[:,y_idx])
    r_mean = np.mean(points[:,3])
    
    X_new = np.hstack((np.ones((N_new,1)), x_new.reshape(-1,1), y_new.reshape(-1,1)))
    z_new = X_new @ W.reshape(-1,1)
    
    if z_idx ==0:
        points_new = np.hstack((z_new.reshape(-1,1), x_new.reshape(-1,1), y_new.reshape(-1,1)))
    elif z_idx ==1:
        points_new = np.hstack((y_new.reshape(-1,1), z_new.reshape(-1,1), x_new.reshape(-1,1)))
    else:
        points_new = np.hstack((x_new.reshape(-1,1), y_new.reshape(-1,1), z_new.reshape(-1,1)))
    
    points_new = np.hstack((points_new, np.zeros(N_new).reshape(-1, 1)+r_mean))
    return points_new
    