import numpy as np

def icp(src, dst):
    """
    This is the icp implementation that will be used pose estimation
    from LIDAR data.

    Parameters:
    src: numpy array
         The point cloud at timestep t
    
    dst: numpy array
         The point cloud at timestep t+1


    Returns:
    C: numpy array
       The rotation matrix

    t: numpy array
       The translation vector

    Note: Using those matrices we can see how the body frame moved from t to t+1
          and we can use this to estimate the pose of the vehicle (or any state).
    """

    # Find the centroid of both sets of points
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)

    # Subtract the centroids from the point clouds
    src_demean = src - src_centroid
    dst_demean = dst - dst_centroid

    # Compute the covariance matrix
    H = np.dot(src_demean.T, dst_demean)

    # Compute the Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)

    # Proper rotation without reflection
    I = np.eye(U.shape[0])
    I[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)

    # Compute the rotation matrix
    R = np.dot(np.dot(Vt, I), U)

    # Compute the translation vector
    t = dst_centroid - np.dot(R, src_centroid)

    return R, t