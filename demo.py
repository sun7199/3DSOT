import json
import os

import yaml
from pyquaternion import Quaternion

from datasets.data_classes import Box
import open3d as o3d
import torch
import pytorch_lightning as pl
from models.m2track import M2TRACK
from datasets import points_utils
import numpy as np
from models.base_model import MotionBaseModel


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def evaluate_one_sample(self, data_dict, ref_box):
    end_points = self(data_dict)

    estimation_box = end_points['estimation_boxes']
    estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

    if len(estimation_box.shape) == 3:
        best_box_idx = estimation_box_cpu[:, 4].argmax()
        estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

    candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=self.config.degrees,
                                             use_z=self.config.use_z,
                                             limit_box=self.config.limit_box)
    return candidate_box


def build_bbox(center, size, rotation):
    box_center_velo = np.array([center["x"], center["y"], center["z"]])
    # transform bb from camera coordinate into velo coordinates
    size = [size["width"], size["length"], size["height"]]
    rotation = rotation["z"]
    orientation = Quaternion(
        axis=[0, 0, 1], radians=rotation)
    bb = Box(box_center_velo, size, orientation)
    return bb


def get_3d_box(center, dimensions, rotation):
    w, l, h = dimensions[0], dimensions[1], dimensions[2]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])
    # 转换为弧度
    rx, ry, rz = rotation[0], rotation[1], rotation[2]
    # 旋转矩阵
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))
    corners_rotated = np.dot(R, corners).T

    # 平移到中心点
    if isinstance(center, list):
        corners_rotated += np.array(center)
    else:
        corners_rotated += np.array([center[0], center[1], center[2]])

    return corners_rotated


def singleFrame(frame, prev_frame, this_frame, json_path, result_bbs):
    # visualization
    pcd = o3d.io.read_point_cloud(prev_frame)
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口，设置窗口标题
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([255, 255, 255])
    # 设置渲染点的大小
    opt.point_size = 3.0
    # 添加点云
    vis.add_geometry(pcd)

    with open(json_path, "rb") as file:
        json_data = json.load(file)
    checkpoint_path = 'pretrained_models/mmtrack_kitti_car.ckpt'
    model = M2TRACK.load_from_checkpoint(checkpoint_path)
    model.eval()
    objects = json_data["objects"]

    for bb in result_bbs:
        input_dict, ref_box = model.build_input_dict_own(frame, prev_frame, this_frame, bb)
        print(bb)
        # Perform the forward pass
        output_dict = model.evaluate_one_sample(input_dict, ref_box)

        center = output_dict.center

        dimensions = output_dict.wlh
        rotation = output_dict.orientation.radians * output_dict.orientation.axis

        box_corners = get_3d_box(center, dimensions, rotation)

        # two different box points sequence
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        # lines_box = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
        #                       [0, 4], [1, 5], [2, 6], [3, 7]])
        # 设置点与点之间线段的颜色
        colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
        # 创建Bbox候选框对象
        line_set = o3d.geometry.LineSet()
        # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        # 设置每条线段的颜色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # 把八个顶点的空间信息转换成o3d可以使用的数据类型
        line_set.points = o3d.utility.Vector3dVector(box_corners)
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)
    vis.run()


if __name__ == "__main__":
    folder_path = "/media/yueming/local disk/pc18_7(1)/pc18-7"
    frame = 1
    result_bbs = []
    for root, dirs, files in os.walk(folder_path):
        if files:
            files = sorted(files)
            if files[0].endswith("json"):
                json_path = os.path.join(root, files[0])
                with open(json_path, "rb") as file:
                    json_data = json.load(file)
                objects = json_data["objects"]

                for obj in objects:
                    center = obj["box3d"]["center"]
                    size = obj["box3d"]["dimensions"]
                    rotation = obj["box3d"]["rotation"]
                    bbox = build_bbox(center, size, rotation)
                    result_bb = []
                    result_bb.append(bbox)
                    result_bbs.append(result_bb)

                for i in range(2, len(files), 2):
                    file1 = files[i - 2]
                    file2 = files[i + 1] if i + 1 < len(files) else None
                    file3 = files[i - 1]
                    json_path = os.path.join(root, file1)
                    json_path = json_path.replace("\\", "/")
                    prev_frame = os.path.join(root, file3) if file3 else None
                    this_frame = os.path.join(root, file2) if file2 else None

                    root = root.replace("\\", "/")
                    singleFrame(frame, prev_frame, this_frame, json_path, result_bbs)
                    print("frame:", frame)
                    frame += 1
