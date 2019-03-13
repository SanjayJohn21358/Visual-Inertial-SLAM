import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrinsic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu

def hat_map(x):
    '''
    make skew symmetric matrix
    '''
    return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])


def imu_EKF_prediction(car,tau,lv,av):
    '''
    Implements EKF prediction of localization given IMU data
    Input:
        car - Car object
        tau - time step in seconds
        lv - linear velocity from IMU
        av - angular velocity from IMU
    Output:
        None
    '''
    # allocate noise
    W = 50*np.identity(6)

    # setup u_hat, u_cov
    u_hat = np.vstack((np.hstack((hat_map(av), lv.reshape(3,1))),np.array([0,0,0,0])))
    u_cov = np.vstack((np.hstack((hat_map(av), hat_map(lv))),np.hstack((np.zeros((3,3)),hat_map(av)))))

    # calculate covariance and mean of car's position using EKF prediction
    car.cov = expm(-float(tau[0])*u_cov)*car.cov*np.transpose(expm(-float(tau[0])*u_cov)) + W
    car.mean = np.dot(expm(-float(tau[0])*u_hat),car.mean)


def derivative_projection_func(q):
    '''
    Given q, find derivative projection function (for Jacobian)
    '''
    return (1/q[2])*np.array([[1,0,-q[0]/q[2],0],[0,1,-q[1]/q[2],0],[0,0,0,0],[0,0,-q[3]/q[2],1]])


def world_T_car(meen):
    '''
    Converts position of car to the world frame        
    '''
    R = meen[0:3,0:3]
    T_inv = np.vstack((np.hstack((np.transpose(R), -np.dot(np.transpose(R),meen[0:3,3].reshape(3,1)))),np.array([0,0,0,1])))
    return T_inv



def landmark_EKF_update(car,Landmarks,tau,old_features,features,K,b,cam_T_imu):
    '''
    Implements EKF update of landmarks given stereo data
    Input:
        car - Car object containing mean of IMU position
        Landmarks - Landmarks object containing mean and covariance position of each landmark
        tau - time step
        old_features - features from last time step (to evaluate if new)
        features - features from current time step
        K - camera calibration matrix (left camera)
        b - stereo baseline
        cam_T_imu - extrinsic matrix from IMU to (left)camera, in SE(3)
    Output:
        None

    '''
    # setup number of landmarks
    n = features.shape[1]
    # setup M matrix (stereo calibration matrix, mirror of K)
    M = np.hstack((np.vstack((K[0:2,0:3],K[0:2,0:3])),np.array([0,0,-K[0,0]*b,0]).reshape(4,-1)))
    # setup dilation matrix
    D = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])

    # setup noise parameter
    V = 3000

    # for each landmark 
    # (you can do for loop since we are not updating IMU, no correlation between landmarks)
    for i in range(n):

        # check if feature is present
        if features[:,i][0] == -1:
            continue
        # check if feature is new
        # if new, mean corresponding is simply the feature itself, no update
        if old_features[:,i][0] == -1 and features[:,i][0] != -1 and Landmarks.mean[i,:][0] == 0 and Landmarks.mean[i,:][1] == 0 and Landmarks.mean[i,:][2] == 0:
            Landmarks.mean[i,:] = np.dot(world_T_car(car.mean),np.dot(np.linalg.inv(cam_T_imu),np.hstack((K[0,0]*b*np.dot(np.linalg.inv(K),np.hstack((features[:,i][0:2],1)))/(features[:,i][0] - features[:,i][2]),1))))
            continue

        # if feature is valid and seen before, update
        # create z_hat(pixel value) from landmark mean
        lmk = np.dot(cam_T_imu,np.dot(car.mean,Landmarks.mean[i,:]))
        z_hat = np.dot(M,(lmk/lmk[2]))

        # create H matrix using derivative of projection function (to enact Jacobian)
        H = np.dot(M,np.dot(derivative_projection_func(lmk),np.dot(cam_T_imu,np.dot(car.mean,D))))

        # calculate Kalman gain and update landmark mean and covariance
        Kay = np.dot(Landmarks.cov[i,:,:],np.dot(np.transpose(H),np.linalg.inv(np.dot(H,np.dot(Landmarks.cov[i,:,:],np.transpose(H))) + V*np.identity(4)[0:5,0:4])))
        Landmarks.mean[i,:] = Landmarks.mean[i,:] + np.dot(D,np.dot(Kay,(features[:,i] - z_hat)))
        Landmarks.cov[i,:,:] = np.dot((np.identity(3) - np.dot(Kay,H)),Landmarks.cov[i,:,:])
        


def imu_EKF_update():
    '''
    Implements EKF update of IMU given IMU prediction and landmark update
    Input:

    '''


def visualize_trajectory_2d(pose,landmark,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  ax.plot(landmark[:,0],landmark[:,1],'b.',label='landmarks')
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/50)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax
