def iou_3d(box1, box2):
    # Unpack the coordinates
    xcenter1,ycenter1,zcenter1,xlen1,ylen1,zlen1 = box1
    xcenter2,ycenter2,zcenter2,xlen2,ylen2,zlen2 = box2


    # Calculate the coordinates of the intersection box
    x_max = min(xcenter1+xlen1/2, xcenter2+xlen2/2)
    x_min = max(xcenter1-xlen1/2, xcenter2-xlen2/2)
    y_max = min(ycenter1+ylen1/2, ycenter2+ylen2/2)
    y_min = max(ycenter1-ylen1/2, ycenter2-ylen2/2)
    z_max = min(zcenter1+zlen1/2, zcenter2+zlen2/2)
    z_min = max(zcenter1-zlen1/2, zcenter2-zlen2/2)

    # Check if there is an intersection
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return 0.0

    # Calculate volumes
    intersection_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    box1_volume = xlen1 * ylen1 * zlen1
    box2_volume = xlen2 * ylen2 * zlen2

    # Calculate union
    union_volume = box1_volume + box2_volume - intersection_volume

    # Calculate IOU
    # range 0-1
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