# Visual Inertial SLAM

## Project Description
Program that takes in stereo image data, and IMU uses SLAM to create a video map of environment. We are given a robot's sensor data as it moves along an environment; our task is to create a 2D map of this environment.

## Methodology
Uses Extended Kalman Filter to make appropriate changes based on map

## Dataset 
Stereo camera data is used for mapping. IMU data is used for localization.

## Files
utils.py: Contains all helper functions used to employ SLAM.

hw3_main.py: Runs the SLAM algorithm. Run this file to see results!

## Results

results/last20.png: final timestep image of dataset 20

results/last27.png: final timestep image of dataset 27

results/last42.png: final timestep image of dataset 42

results/w-----.png: final timestep of dataset with corresponding noise w =, v=



Red is trajectory, blue is landmarks in scene.

![alt text](results/last27.png?raw=True 'Visual Inertial SLAM of dataset 27')

![alt text](results/last42.png?raw=True 'Visual Inertial SLAM of dataset 42') 

