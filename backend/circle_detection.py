import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sympy import symbols, Eq, solve
from math import atan2, tan
import uuid

def findCOCircle(points):
    # Fit a spline to the points
    tck, u = splprep(points.T, s=0)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)
    # Select key points on the curve (e.g., 25%, 50%, 75% of the curve)
    indices_of_interest = [len(unew) // 4, len(unew) // 2, 3 * len(unew) // 4]
    # Calculate first derivatives at these key points
    dx1 = np.array([splev(unew[i], tck, der=1)[0] for i in indices_of_interest])
    dy1 = np.array([splev(unew[i], tck, der=1)[1] for i in indices_of_interest])
    # Calculate points of interest
    points_of_interest = np.array([out[0][indices_of_interest], out[1][indices_of_interest]]).T
    # Calculate unit normal vectors at these key points
    unit_tangent_vectors = np.vstack((dx1, dy1)).T
    tangent_magnitudes = np.linalg.norm(unit_tangent_vectors, axis=1)
    unit_normal_vectors = np.vstack((-unit_tangent_vectors[:, 1], unit_tangent_vectors[:, 0])).T / tangent_magnitudes[:, np.newaxis]
    # Define symbols for the intersection calculation
    x, y = symbols('x y')
    # List to store equations of normals
    normal_equations = []
    # Calculate normal lines at the key points
    for point, normal in zip(points_of_interest, unit_normal_vectors):
        px, py = point
        nx, ny = normal
        normal_equation = Eq(ny * (x - px) - nx * (y - py), 0)
        normal_equations.append(normal_equation)
    # Solve for the intersection of normals
    intersection = solve(normal_equations[:2], (x, y))
    # Convert intersection to numerical values
    center_of_curvature = np.array([float(intersection[x]), float(intersection[y])])
    return center_of_curvature

def findROCircle(points):
    # Fit a spline to the points
    tck, u = splprep(points.T, s=0)
    unew = np.linspace(0, 1, 100)
    out = splev(unew, tck)
    # Calculate first and second derivatives
    dx1, dy1 = splev(unew, tck, der=1)
    dx2, dy2 = splev(unew, tck, der=2)
    # Calculate curvature at each point
    curvature = np.abs(dx1 * dy2 - dy1 * dx2) / np.power(dx1**2 + dy1**2, 3/2)
    # Calculate radius of curvature at each point
    radius_of_curvature = 1 / curvature
    # Average the radius of curvature
    average_roc = np.mean(radius_of_curvature)
    return average_roc

def mode(lst):
    freq = {}
    for i in lst:
        freq.setdefault(i, 0)
        freq[i] += 1
    hf = max(freq.values())
    hflst = []
    for i, j in freq.items():
        if j == hf:
            hflst.append(i)
    return hflst[0]

class CollectionPoints:
    def __init__(self, points):
        self.points = points

class Curves(CollectionPoints):
    def __init__(self, points, ROC, COC):
        super().__init__(points)
        self.ROC = ROC
        self.COC = COC

    def __str__(self):
        return f'Curve(Radius of Curvature: {self.ROC}, Center of Curvature: {self.COC})'

class StraightLine(CollectionPoints):
    def __init__(self, points, slope):
        super().__init__(points)
        self.slope = slope

    def __str__(self):
        return f'Straight Line(Slope: {self.slope})'

class Circle(CollectionPoints):
    def __init__(self, points, center, radius):
        self.points = points
        self.center = center
        self.radius = radius

    def __str__(self):
        return f'Circle(Center: {self.center}, Radius: {self.radius})'

def fit_circle(points):
    A = np.zeros((len(points), 3))
    A[:, 0] = points[:, 0]
    A[:, 1] = points[:, 1]
    A[:, 2] = 1
    B = np.zeros((len(points), 1))
    B[:, 0] = points[:, 0]**2 + points[:, 1]**2
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc = 0.5 * C[0][0]
    yc = 0.5 * C[1][0]
    r = np.sqrt(C[2][0] + xc**2 + yc**2)
    return (xc, yc), r


def check_if_circle(points, tolerance=0.20, percentage_threshold=0.9,angle_tolerance=150):
    center, radius = fit_circle(points)

    # Calculate distances of all points from the center
    distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
    
    # Calculate mean and standard deviation of distances
    mean_distance = np.mean(distances)
    
    # Determine how many points fall within the acceptable range
    deviation_range = tolerance * mean_distance
    within_range = np.sum(np.abs(distances - mean_distance) <= deviation_range)
    
    # Check if more than percentage_threshold of points fall within the acceptable range
    if (within_range / len(points)) < percentage_threshold:
        return None
    
    # Additional check: Ensure points are uniformly distributed around the center
    angles = np.degrees(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])))
    angles = np.round(angles).astype(int)
    angles = np.abs(angles)
    num_unique_elements = len(np.unique(angles))
    if(num_unique_elements < angle_tolerance):
        return None
    return Circle(points, center, radius)

def classify_paths(path_XYs):
    classified_paths = []
    for j, XYs in enumerate(path_XYs):
        print(str(j) + ' ')
        XY = XYs[0]
        # Compute slopes for straight line classification
        slopes = []
        for i in range(len(XY) - 1):

            pts = XY[i].tolist()
            pts_next = XY[i + 1].tolist()
            slope = atan2((pts_next[1] - pts[1]), (pts_next[0] - pts[0]))
            slopes.append(slope)
        avg_slope_ang = mode(np.round(slopes, 2).tolist())
        slope_modified = []
        for angle in slopes:
            if abs(angle - avg_slope_ang) <= 0.25:
                slope_modified.append(angle)

        # Check if the path is a straight line
        if len(slope_modified) >= 0.8 * len(slopes):
            classified_paths.append(StraightLine(XY, tan(avg_slope_ang)))
        else:
            # print(str(i) + ' ')
            # Check if the path is a circle
            circle = check_if_circle(XY)
            if circle is not None:
                classified_paths.append(circle)
                print(f'Circle found with center: {circle.center} and radius: {circle.radius}')
            classified_paths.append(Curves(XY, findROCircle(XY), findCOCircle(XY)))
    return classified_paths


