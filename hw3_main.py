import numpy as np
from utils import *

class Car(object):
	def __init__(self):
		self.mean = np.identity(4) #+ 0.0005*np.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]]) # initialize mu as [[R p] [0 0]] where R is identity(3) and p is 0,0,0
		self.cov = 0.005*np.identity(6)

class Landmarks(object):
	def __init__(self,M):
		self.mean = np.zeros((M,4))
		self.cov = np.zeros((M,3,3))
		for i in range(M):
			self.cov[i,:,:] = np.identity(3)*0.005


if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	# Initialize car and landmarks
	Lambo = Car()
	Landmarks = Landmarks(features.shape[1])

	# Initialize pose trajectory
	pose_trajectory = np.zeros((4,4,t.shape[1]))

	for i,t_n in enumerate(t[0][1:],1):

		#get time step
		tau = abs(t_n - t[:,i-1])
		# (a) IMU Localization via EKF Prediction
		imu_EKF_prediction(Lambo,tau,linear_velocity[:,i],rotational_velocity[:,i])

		# add inverse pose to trajectory array
		pose_trajectory[:,:,i] = world_T_car(Lambo.mean)

		# (b) Landmark Mapping via EKF Update
		landmark_EKF_update(Lambo,Landmarks,tau,features[:,:,i-1],features[:,:,i],K,b,cam_T_imu)

		# (c) Visual-Inertial SLAM (Extra Credit)
		#updated_imu = imu_EKF_update()


	# Visualize the robot pose over time
	visualize_trajectory_2d(pose_trajectory,Landmarks.mean,show_ori=True)



