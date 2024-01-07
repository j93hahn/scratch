import numpy as np


"""
A simple class to generate arbitrary camera poses from which to render images. Great introduction found
at https://wandb.ai/wandb/nerf-jax/reports/Implementing-NeRF-in-JAX--VmlldzoxODA2NDk2#rendering-the-scene

TODO: Look at Intro to Computer Vision Homework 2

The camera pose is defined by radius, theta, and phi. We begin by placing the camera at the origin such
that it looks down the negative z-axis.
at the origin. The camera is then rotated by theta and phi
degrees around the x and z axes, respectively. Finally, the camera is translated
by radius units along the z-axis.
- Radius is the distance from the camera to the origin
- Theta is the angle between the x-axis and the camera
- Phi is the angle between the z-axis and the camera
"""
class SinglePoseCalculator():
    def __init__(self, radius, theta, phi, is_radians=False):
        self.radius = radius
        if not is_radians:  # theta and phi should be converted to radians
            self.theta = theta / 180.0 * np.pi
            self.phi = phi / 180.0 * np.pi
        else:
            self.theta = theta
            self.phi = phi

    def get_translation_matrix(self):
        T = np.eye(4)
        T[2, 3] = self.radius
        return T

    def get_rotation_matrix_phi(self):
        R = np.eye(4)
        R[1, 1] = np.cos(self.phi)
        R[1, 2] = -np.sin(self.phi)
        R[2, 1] = np.sin(self.phi)
        R[2, 2] = np.cos(self.phi)
        return R

    def get_rotation_matrix_theta(self):
        R = np.eye(4)
        R[0, 0] = np.cos(self.theta)
        R[0, 2] = np.sin(self.theta)
        R[2, 0] = -np.sin(self.theta)
        R[2, 2] = np.cos(self.theta)
        return R

    def get_c2w_matrix(self):
        c2w = self.get_translation_matrix()
        c2w = self.get_rotation_matrix_phi() @ c2w
        c2w = self.get_rotation_matrix_theta() @ c2w
        c2w = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]) @ c2w
        return c2w


"""
This RayGenerator class is used to generate rays for a given camera pose. The focal length is the distance
between the camera and the image plane. The height and width are the dimensions of the image plane.
"""
class RayGenerator():
    def __init__(self, focal_length, height, width):
        self.focal_length = focal_length
        self.height = height
        self.width = width
