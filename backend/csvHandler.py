from utils import read_csv, plot
from breakshapes import get_break_shapes
from demo import smooth_curve
import numpy as np
from math import atan2, atan
from circle_detection import Circle, check_if_circle
import os
pointsRC = []
def normalize_shape(path_XYs, angle_threshold=0.3, eps=10, min_samples=1):
    normalized_shapes = []
    for shape in path_XYs:
        [broken_shape,pts] = get_break_shapes(shape, angle_threshold, eps, min_samples)
        for p in pts:
            pointsRC.append(p)
        for s in broken_shape:
            normalized_shapes.append(s)
    return normalized_shapes
    
def merge_paths2(pathsXY, threshold=10, epsilon=0.4):
    merged_pathsXY = []
    visited = [False] * len(pathsXY)
    for i in range(len(pathsXY)):
        if visited[i]:
            continue
        visited[i] = True
        tempPath = pathsXY[i]
        while True:
            found = False
            for j in range(len(pathsXY)):
                if visited[j]:
                    continue
                if (np.linalg.norm(np.mean(tempPath[:4], axis=0) - np.mean(pathsXY[j][-4:], axis=0)) < threshold):
                    print("Case-1",i,j)
                    avg_slope_1 = 0
                    for k in range(1, 4):
                        pts = tempPath[k]
                        pts_next = tempPath[k + 1]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_1 += slope
                    avg_slope_1 /= 3
                    avg_slope_2 = 0
                    for k in range(1, 4):
                        pts = pathsXY[j][-k - 1]
                        pts_next = pathsXY[j][-k]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_2 += slope
                    avg_slope_2 /= 3
                    if abs(avg_slope_1 - avg_slope_2) < epsilon:
                        tempPath = np.concatenate((pathsXY[j], tempPath))
                        visited[j] = True
                        found = True
                elif (np.linalg.norm(np.mean(tempPath[-4:], axis=0) - np.mean(pathsXY[j][:4], axis=0)) < threshold):
                    print("Case-2",i,j)
                    avg_slope_1 = 0
                    for k in range(1, 4):
                        pts = tempPath[-k - 1]
                        pts_next = tempPath[-k]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_1 += slope
                    avg_slope_1 /= 3
                    avg_slope_2 = 0
                    for k in range(1, 4):
                        pts = pathsXY[j][k]
                        pts_next = pathsXY[j][k + 1]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_2 += slope
                    avg_slope_2 /= 3
                    if abs(avg_slope_1 - avg_slope_2) < epsilon:
                        tempPath = np.concatenate((tempPath, pathsXY[j]))
                        visited[j] = True
                        found = True
                elif (np.linalg.norm(np.mean(tempPath[:4], axis=0) - np.mean(pathsXY[j][:4], axis=0)) < threshold):
                    print("Case-3",i,j)
                    avg_slope_1 = 0
                    for k in range(1, 4):
                        pts = tempPath[k]
                        pts_next = tempPath[k + 1]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_1 += slope
                    avg_slope_1 /= 3
                    avg_slope_1 = avg_slope_1 * -1
                    avg_slope_2 = 0
                    for k in range(1, 4):
                        pts = pathsXY[j][k]
                        pts_next = pathsXY[j][k + 1]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_2 += slope
                    avg_slope_2 /= 3
                    if abs(avg_slope_1 - avg_slope_2) < epsilon:
                        tempPath = np.concatenate((np.flip(pathsXY[j], 0), tempPath))
                        visited[j] = True
                        found = True
                elif (np.linalg.norm(np.mean(tempPath[-4:], axis=0) - np.mean(pathsXY[j][-4:], axis=0)) < threshold):
                    print("Case-4",i,j)
                    avg_slope_1 = 0
                    for k in range(1, 4):
                        pts = tempPath[-k - 1]
                        pts_next = tempPath[-k]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_1 += slope
                    avg_slope_1 /= 3
                    avg_slope_1 = avg_slope_1 * -1
                    avg_slope_2 = 0
                    for k in range(1, 4):
                        pts = pathsXY[j][-k - 1]
                        pts_next = pathsXY[j][-k]
                        slope = atan2((pts_next[0] - pts[0]), (pts_next[1] - pts[1]))
                        avg_slope_2 += slope
                    avg_slope_2 /= 3
                    print(avg_slope_1,avg_slope_2)
                    if abs(avg_slope_1 - avg_slope_2) < epsilon:
                        tempPath = np.concatenate((tempPath, np.flip(pathsXY[j], 0)))
                        visited[j] = True
                        found = True
            if not found:
                break
        merged_pathsXY.append(tempPath)
    return merged_pathsXY


def merge_Straignth(pathsXY, threshold=10):
    merged_pathsXY = []
    visited = [False] * len(pathsXY)
    for i in range(len(pathsXY)):
        if visited[i]:
            continue
        visited[i] = True
        tempPath = pathsXY[i]
        while True:
            found = False
            for j in range(len(pathsXY)):
                if visited[j]:
                    continue
                if (np.linalg.norm(np.mean(tempPath[:4], axis=0) - np.mean(pathsXY[j][-4:], axis=0)) < threshold):
                    tempPath = np.concatenate((pathsXY[j], tempPath))
                    visited[j] = True
                    found = True
                elif (np.linalg.norm(np.mean(tempPath[-4:], axis=0) - np.mean(pathsXY[j][:4], axis=0)) < threshold):
                    tempPath = np.concatenate((tempPath, pathsXY[j]))
                    visited[j] = True
                    found = True
                elif (np.linalg.norm(np.mean(tempPath[:4], axis=0) - np.mean(pathsXY[j][:4], axis=0)) < threshold):
                    tempPath = np.concatenate((np.flip(pathsXY[j], 0), tempPath))
                    visited[j] = True
                    found = True
                elif (np.linalg.norm(np.mean(tempPath[-4:], axis=0) - np.mean(pathsXY[j][-4:], axis=0)) < threshold):
                    tempPath = np.concatenate((tempPath, np.flip(pathsXY[j], 0)))
                    visited[j] = True
                    found = True
            if not found:
                break
        merged_pathsXY.append(tempPath)
    return merged_pathsXY

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    num_points = len(points)
    interpolated_x = np.linspace(x[0], x[-1], num_points)
    interpolated_y = np.linspace(y[0], y[-1], num_points)
    fitted_line = np.column_stack((interpolated_x, interpolated_y))
    return fitted_line

def generate_uniform_circle_points(circle):
    [center_x, center_y] = circle.center
    radius = circle.radius
    num_points = int(2 * np.pi * radius)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    circle_points = np.column_stack((x, y))
    return circle_points
pathsXY = []

def fix_line(points, threshold=10.0):
    angles = np.degrees(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])))
    angles = np.round(angles).astype(int)
    angles = np.abs(angles)
    num_unique_elements = np.unique(angles)
    variance = np.var(num_unique_elements)
    if(variance<200):
        return fit_line(points)
    else:
        return points


def process_csv(input_csv):
    for s in read_csv(input_csv):
        for j in s:
                if len(j) < 50:
                    pts = np.array(smooth_curve(j))
                else:
                    pts = np.array(j)            
                crl = check_if_circle(pts)
                if crl:
                    circle_pts = generate_uniform_circle_points(crl)
                    pathsXY.append(circle_pts)
                else:
                    pathsXY.append(pts)

    # print((plotExtend()))
    pathsXY2 = []
    curvesS = []
    striaght_lines = []
    for i in normalize_shape(pathsXY):
        pathsXY2.append(fix_line(i))
        if all(np.array_equal(x, y) for x, y in zip(i, fix_line(i))):
            curvesS.append(i)
        else:
            striaght_lines.append(fix_line(i))    
    striaght_lines = merge_Straignth(striaght_lines)
    curvesS = merge_paths2(curvesS)
    pathsXY4 = []
    for i in curvesS:
        if(check_if_circle(i)):
            pathsXY4.append(generate_uniform_circle_points(check_if_circle(i)))
        else:
            pathsXY4.append(i)
    for striaght_line in striaght_lines:
        pathsXY4.append(striaght_line)
        
    plot(pathsXY4)

    process_csv('new.csv')