import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
#images: list of filenames for checkboard calibration files
#checkerboard: dimension of the inner corners
#dW: corner refinement window size. should be smaller for lower resolution images

X_CAPT = np.float32([])
Y_CAPT = np.float32([])
def computeExtrinsic(img_path, mtx, dist, dX, dY):
    global X_CAPT, Y_CAPT
    X_CAPT = np.float32([])
    Y_CAPT = np.float32([])
    color_img = cv2.imread(img_path)
    I = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)
    def capture_click(event, x_click, y_click, flags, params):
        global X_CAPT, Y_CAPT
        if event == cv2.EVENT_LBUTTONDOWN:
            xy_click = np.float32([x_click, y_click])
            print(xy_click)
            xy_click = xy_click.reshape(-1, 1, 2)
            #print(xy_click)
            refined_xy = cv2.cornerSubPix(I, xy_click, (11, 11), (-1, -1), criteria)
            #print(refined_xy)
            X_CAPT = np.append(X_CAPT, refined_xy[0, 0, 0])
            Y_CAPT = np.append(Y_CAPT, refined_xy[0, 0, 1])
            cv2.drawMarker(color_img, (int(X_CAPT[-1]), int(Y_CAPT[-1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30)
    #plt.imshow(color_img[:,:,::-1])
    compute_name = 'Define Extrinsic'
    cv2.namedWindow(compute_name)
    cv2.setMouseCallback(compute_name, capture_click)
    print("Click on the four corners of the rectangular pattern, starting from the bottom-left and proceeding counter-clockwise.")
    #draw user selected features
    while True:
        cv2.imshow(compute_name, color_img)
        key = cv2.waitKey(1)
        if key == ord("q") or len(X_CAPT) == 4:
            break
    x = X_CAPT
    y = Y_CAPT
    #print("click input: ", x, y)
    #sort corners
    #x_v = x - x.mean()
    #y_v = y - y.mean()
    #theta = np.arctan2(-y_v, x_v)
    #ind = np.argsort(np.mod(theta-theta[0], 2*np.pi))
    #ind = ind[::-1]
    #x = x[ind]
    #y = y[ind]
    #print ("sorted corners: ", x, y)
    #xy_corners_undist = cv2.undistortPoints(, mtx, dist )
    #project grid points
    #M = cv2.findhomography()
    #distort projected points
    img_points = np.vstack((x, y)).T.reshape(-1, 1, 2)
    obj_points = np.array([[0, dX, dX, 0], [0, 0, dY, dY], [0, 0, 0, 0]]).T.reshape(-1, 1, 3)
    #print("image points: ", img_points)
    #print("object points: ", obj_points)
    result = cv2.solvePnPRansac(obj_points, img_points, mtx, dist)
    #print(result)
    rvec = result[1]
    tvec = result[2]
    rmat = cv2.Rodrigues(rvec)[0]
    #display axis
    #origin, x (red), y (green), z (blue)
    axis = np.float32([[0,0,0], [dX,0,0], [0,dY,0], [0,0,min(dX, dY)]]).reshape(-1,3)
    axis_img = cv2.projectPoints(axis, rvec, tvec, mtx, dist)[0]
    #print("projected points: ", axis_img)
    axis_img = axis_img.astype(int)
    #cv2.namedWindow('Result')
    cv2.putText(color_img, 'X', (axis_img[1,0,0], axis_img[1,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[1,0,0], axis_img[1,0,1]), (0,0,255), 2) #x
    cv2.putText(color_img, 'Y', (axis_img[2,0,0], axis_img[2,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[2,0,0], axis_img[2,0,1]), (0,255,0), 2) #y
    cv2.putText(color_img, 'Z', (axis_img[3,0,0], axis_img[3,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[3,0,0], axis_img[3,0,1]), (255,0,0), 2) #z
    cv2.imshow(compute_name, color_img)
    print("Done! Press any key to exit")
    cv2.waitKey(0)
    #visualize camera relative to calibration plane
    image_corners = np.float32([[0,0], [I.shape[1], 0], [0, I.shape[0]], [I.shape[1], I.shape[0]]])
    corner_colors = [(0,0,0), (1,0,1), (0,0,1), (0,0,1)]
    print("image corners: ", image_corners, image_corners.shape)
    corner_rays = np.matmul(rmat.T, np.squeeze(200*pixel2ray(image_corners, mtx, dist)).T)
    print(corner_rays)
    fig = plt.figure("Projected camera view")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([0, dX, dX, 0, 0], [0, 0, dY, dY, 0])
    C = np.matmul(-rmat.T, tvec)
    ax.scatter(C[0], C[1], C[2], s=10, marker="s")
    ax.quiver(C[0], C[1], C[2], corner_rays[0,:], corner_rays[1,:], corner_rays[2,:], color=corner_colors)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()
    return tvec, rmat
#points: nx2 np.float32 array
#mtx: camera matrix
#dist: distortion values
#rays: nx1x3 np.float32 array
def pixel2ray(points, mtx, dist):
    undist_points = cv2.undistortPoints(points, mtx, dist)
    rays = cv2.convertPointsToHomogeneous(undist_points)
    #print("rays: ", rays)
    norm = np.sum(rays**2, axis = -1)**.5
    #print("norm: ", norm)
    rays = rays/norm.reshape(-1, 1, 1)
    return rays


#code for setting axes to be equal in matplotlib3D
#taken from https://stackoverflow.com/questions/13685386/63625222#63625222
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.
    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.                             ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files                       ####
##################################################################################################


import pyrealsense2 as rs
import numpy as np

"""
  _   _        _                      _____                     _    _                    
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
                 |_|                                                                      
"""


class Device:
    def __init__(self, pipeline, pipeline_profile, product_line):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
        self.product_line = product_line

def enumerate_connected_devices(context):
    """
    Enumerate the connected Intel RealSense devices

    Parameters:
    -----------
    context 	   : rs.context()
                     The context created for using the realsense library

    Return:
    -----------
    connect_device : array
                     Array of (serial, product-line) tuples of devices which are connected to the PC

    """
    connect_device = []

    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            device_info = (serial, product_line) # (serial_number, product_line)
            connect_device.append( device_info )
    return connect_device


def post_process_depth_frame(depth_frame, decimation_magnitude=1.0, spatial_magnitude=2.0, spatial_smooth_alpha=0.5,
                             spatial_smooth_delta=20, temporal_smooth_alpha=0.4, temporal_smooth_delta=20):
    """
    Filter the depth frame acquired using the Intel RealSense device

    Parameters:
    -----------
    depth_frame          : rs.frame()
                           The depth frame to be post-processed
    decimation_magnitude : double
                           The magnitude of the decimation filter
    spatial_magnitude    : double
                           The magnitude of the spatial filter
    spatial_smooth_alpha : double
                           The alpha value for spatial filter based smoothening
    spatial_smooth_delta : double
                           The delta value for spatial filter based smoothening
    temporal_smooth_alpha: double
                           The alpha value for temporal filter based smoothening
    temporal_smooth_delta: double
                           The delta value for temporal filter based smoothening

    Return:
    ----------
    filtered_frame : rs.frame()
                     The post-processed depth frame
    """

    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)

    return filtered_frame


def projectPixelTo3dRay(mtx, uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)

        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        x = (uv[0] - mtx[0,2]) / mtx[0,0]
        y = (uv[1] - mtx[1,2]) / mtx[1,1]
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return np.array([x, y, z]).reshape((3,1))
