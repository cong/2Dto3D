#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2020-07-16 09:01
# @Author  : WangCong
# @Email   : iwangcong@outlook.com

import numpy as np
import cv2

camera_parameter = {
    # R
    "R": [[-0.91536173, 0.40180837, 0.02574754],
          [0.05154812, 0.18037357, -0.98224649],
          [-0.39931903, -0.89778361, -0.18581953]],
    # T
    "T": [1841.10702775, 4955.28462345, 1563.4453959],
    # f/dx, f/dy
    "f": [1145.04940459, 1143.78109572],
    # center point
    "c": [512.54150496, 515.45148698]
}


def pixel_to_world(camera_intrinsics, r, t, img_points):

    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points


if __name__ == '__main__':
    f = camera_parameter["f"]
    c = camera_parameter["c"]
    camera_intrinsic = np.mat(np.zeros((3, 3), dtype=np.float64))
    camera_intrinsic[0, 0] = f[0]
    camera_intrinsic[1, 1] = f[1]
    camera_intrinsic[0, 2] = c[0]
    camera_intrinsic[1, 2] = c[1]
    camera_intrinsic[2, 2] = np.float64(1)
    r = camera_parameter["R"]
    t = np.asmatrix(camera_parameter["T"]).T
    # img_points = [[100, 200],
    #               [150, 300]]
    img_points = np.array(([100, 200],
                           [150, 300]), dtype=np.double)
    result = pixel_to_world(camera_intrinsic, r, t, img_points)
    print(result)
    print('----')

    axis = np.float32([[7700, 73407, 0], [-66029, -605036, 0]])
    r2 = np.asmatrix(camera_parameter["R"])
    result2, _ = cv2.projectPoints(axis, r2, t, camera_intrinsic, 0)
    print(result2)



