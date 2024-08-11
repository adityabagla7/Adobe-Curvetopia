import numpy as np
from sklearn.cluster import DBSCAN

def break_shape(shape, representative_corners):
    break_lines = []
    start = 0
    for corner in representative_corners:
        end = np.where(shape == corner)[0][0] + 1
        if end - start > 5:
            break_lines.append(shape[start:end])
            start = end
    if start < len(shape)-5:
        rpc = find_representative_corners(shape[start:], angle_threshold=0.3, eps=5, min_samples=1)
        if len(rpc)==1:
            end = np.where(shape == rpc[0])[0][0] + 1
            if end - start > 5:
                break_lines.append(shape[start:end])
            else:
                break_lines.append(shape[start:])
        else:
            break_lines.append(shape[start:])
    return break_lines

def find_representative_corners(points, angle_threshold=0.3, eps=5, min_samples=1):
    def calculate_curvature(points):
        angles = []
        if len(points) < 3:
            return np.array([])
        
        for i in range(1, len(points) - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            
            # Vectors
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Compute angle between vectors
            angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
            angles.append(np.abs(angle))
        
        return np.array(angles)

    def detect_potential_corners(points, angle_threshold):
        angles = calculate_curvature(points)
        if angles.size == 0:
            return np.array([])
        
        potential_corners = []
        
        for i in range(1, len(points) - 1):
            if angles[i - 1] > angle_threshold:
                potential_corners.append(points[i])
        
        return np.array(potential_corners)

    def cluster_corners(corners, eps, min_samples):
        if corners.size == 0:
            return []
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(corners)
            labels = db.labels_
            
            clusters = []
            for label in set(labels):
                if label != -1:  # Ignore noise points
                    cluster_points = corners[labels == label]
                    clusters.append(cluster_points)
            return clusters
        except Exception as e:
            print(f"An error occurred during clustering: {e}")
            return []

    def find_representative_corners(clusters):
        if not clusters:
            return np.array([])
        representative_corners = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            # Choose the point closest to the centroid of the cluster
            centroid = np.mean(cluster, axis=0)
            distances = np.linalg.norm(cluster - centroid, axis=1)
            representative_index = np.argmin(distances)
            representative_corners.append(cluster[representative_index])
        
        return np.array(representative_corners)

    # Detect potential corners
    potential_corners = detect_potential_corners(points, angle_threshold)

    # Handle case where no potential corners are found
    if potential_corners.size == 0:
        print("No potential corners found.")
        return np.array([])
    else:
        # Cluster the corners
        clusters = cluster_corners(potential_corners, eps, min_samples)
        # Find representative corners for each cluster
        representative_corners = find_representative_corners(clusters)
        if len(representative_corners) > 1:
            # Check if first point and any representative points are the same
            if np.linalg.norm(points[0] - representative_corners[0]) < eps:
                representative_corners = representative_corners[1:]
            # Check if last point and any representative points are the same
            if np.linalg.norm(points[-1] - representative_corners[-1]) < eps:
                representative_corners = representative_corners[:-1]
        return representative_corners

def get_break_shapes(points, angle_threshold=0.2, eps=5, min_samples=1):
    newArr = find_representative_corners(points)
    if newArr.size > 0:
        return break_shape(points, newArr),newArr
    else:
        return [points],newArr
    



