def iou_3d(box1, box2):
    # Unpack the coordinates
    _, _, _, x1_max, x1_min, y1_max, y1_min, z1_max, z1_min = box1
    _, _, _, x2_max, x2_min, y2_max, y2_min, z2_max, z2_min = box2

    # Calculate the coordinates of the intersection box
    x_max = min(x1_max, x2_max)
    x_min = max(x1_min, x2_min)
    y_max = min(y1_max, y2_max)
    y_min = max(y1_min, y2_min)
    z_max = min(z1_max, z2_max)
    z_min = max(z1_min, z2_min)

    # Check if there is an intersection
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return 0.0

    # Calculate volumes
    intersection_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    box1_volume = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    box2_volume = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)

    # Calculate union
    union_volume = box1_volume + box2_volume - intersection_volume

    # Calculate IOU
    return intersection_volume / union_volume

def feature_similarity(feature1, feature2):
    #todo
    pass


import numpy as np

def remove_outliers(points):
    """
    Removes the 10% most distant points from a set based on their Euclidean distance to the centroid.
    
    Parameters:
        points (numpy.ndarray): An array of points where each row represents a point and columns represent dimensions.
    
    Returns:
        numpy.ndarray: An array containing the filtered points, excluding the 10% outliers.
    """
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate distances from the centroid
    distances = np.linalg.norm(points - centroid, axis=1)

    # Determine the 90th percentile distance
    threshold_distance = np.percentile(distances, 90)

    # Filter out points beyond the 90th percentile distance
    filtered_points = points[distances <= threshold_distance]

    return filtered_points