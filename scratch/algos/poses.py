"""
Focal length is the distance between the camera lens and the image plane. It is
measured in pixels. The focal length is determined by the camera and is fixed
for a given camera. The focal length is usually represented in millimeters (mm).
The focal length is inversely proportional to the field of view. A longer focal
length results in a narrower field of view. A shorter focal length results in a
wider field of view.

The focal length is calculated using the following formula:

focal_length = (image_width * distance_to_object) / object_width

"""




import numpy as np
import torch
import json
from pathlib import Path
from warnings import warn

try:
    import open3d as o3d
except ImportError:
    o3d = None

class PoseCalculator():
    def __init__(self, K, R, t):
        self.K = K
        self.R = R
        self.t = t

    def visualize(self, pcd):
        if o3d is None:
            warn("open3d not found. Visualization not possible.")
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    def calculate_focal_length(self, ):
        return


if __name__ == '__main__':
    with open(Path(__file__).parent / 'transforms.json', 'r') as f:
        x = json.load(f)

    # x = np.load('save.npy')[0]
    # x2 = np.load('save2.npy')[0]
    # print(f"x: {x}")
    # print(f"x2: {x2}")
    # pcd = o3d.geometry.PointCloud()
    # pcd2 = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(x)
    # pcd2.points = o3d.utility.Vector3dVector(x2)
    # pcd.paint_uniform_color([0.0, 1.0, 0.0])
    # pcd2.paint_uniform_color([1.0, 0.0, 0.0])
    # o3d.visualization.draw_geometries([pcd2, pcd])

    sigma_high_voxel = np.load('big_xyz_locs_pose0.npy')
    sigma_low_voxel = np.load('xyz_locs_pose0.npy')


    pcd = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sigma_high_voxel)
    pcd2.points = o3d.utility.Vector3dVector(sigma_low_voxel)
    pcd.paint_uniform_color([0.0, 1.0, 0.0])
    pcd2.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd2])
