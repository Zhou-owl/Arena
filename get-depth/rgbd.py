import os
import time

import cv2  # pip install opencv-python
import imageio
import numpy as np  # pip install numpy
import PySimpleGUI as sg  # pip install pysimplegui
from dt_apriltags import Detector
from PIL import Image
from ultralytics import YOLO
from oc_sort import OCSort
from util import *
from examples import *
import numpy as np

# frames = []
# # Load a model
model_tripod = YOLO('./runs/detect/train/weights/best.pt')  
model_basic = YOLO('./yolov8l-seg.pt')  
at_detector = Detector(families='tag36h11')

tracker = OCSort(det_thresh=0.7,iou_threshold=0.5)
capture_root = './capture'


# init cameras 

pipeline = rs.pipeline()
config = rs.config()
device_list = rs.context()
device_serials = [i.get_info(rs.camera_info.serial_number) for i in device_list.devices]
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)


def init_params():
    device_manager = DeviceManager(device_list, config)
    device_manager.enable_all_devices()
    for k in range(150):
        frames = device_manager.poll_frames()
    device_manager.enable_emitter(True)
    device_extrinsics = device_manager.get_depth_to_color_extrinsics(frames)
    device_intrinsics = device_manager.get_device_intrinsics(frames)

    device_names = device_extrinsics.keys()

    for cam in device_names:
        E_rot =  np.array(device_extrinsics[cam].rotation).reshape((3,3))
        E_trans = np.array(device_extrinsics[cam].translation).reshape((3,1))
        E_dr = np.concatenate((E_rot,E_trans),axis=1)
        # print(device_intrinsics[cam].keys())

        color_stream = device_intrinsics[cam].popitem()[1]
        depth_stream = device_intrinsics[cam].popitem()[1]
        I_d_fx = depth_stream.fx
        I_d_fy = depth_stream.fy
        I_d_ppx = depth_stream.ppx
        I_d_ppy = depth_stream.ppy
        I_d_dist = np.array(depth_stream.coeffs)
        I_d_mtx = np.array([[I_d_fx,0,I_d_ppx],[0,I_d_fy,I_d_ppy],[0,0,1]])

        I_c_fx = color_stream.fx
        I_c_fy = color_stream.fy
        I_c_ppx = color_stream.ppx
        I_c_ppy = color_stream.ppy
        I_c_dist = np.array(color_stream.coeffs)
        I_c_mtx = np.array([[I_c_fx,0,I_c_ppx],[0,I_c_fy,I_c_ppy],[0,0,1]])

        np.savez(cam+".npz", E_dr=E_dr, I_d_mtx=I_d_mtx, I_d_dist=I_d_dist,I_c_dist=I_c_dist,I_c_mtx=I_c_mtx)


#init_params()

def load_multicam_params():
    matrix_dict = {}
    for dev in device_serials:
        with np.load(dev+".npz") as X:
            E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx = [X[i] for i in ('E_dr', 'I_d_mtx', 'I_d_dist', 'I_c_dist', 'I_c_mtx')]
            matrix_dict[dev] = [E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx]
    return matrix_dict

def load_cam_params(cam_id):
    with np.load(cam_id+".npz") as X:
        E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx = [X[i] for i in ('E_dr', 'I_d_mtx', 'I_d_dist', 'I_c_dist', 'I_c_mtx')]
    return E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx



# init gui
layout = [
    [sg.Image(filename='', key='color',size=(640,360))],
    [sg.Text('fps = 0',  key="text")],
    [sg.Image(filename='', key='depth',size=(640,360))],
    [sg.Checkbox('Tripod', key='tripod')],
    [sg.Checkbox('FindTag', key='tag')],
    [sg.Checkbox('Chair (check Tag and uncheck Calib)', key='chair')],
    [sg.Checkbox('tracker', key='tracker')],
    [sg.Checkbox('Test', key='test')],
    [sg.Button('Capture')],
    [sg.Button('Exit')]
]
window = sg.Window('camera',
            layout,
            location=(1000, 500),
            resizable=True,
            element_justification='c',
            font=("Arial Bold",20),
            finalize=True)




cam1_name = '241122304996'
cam2_name = '225322300120'
params = load_multicam_params()
# start steaming
single_cam = pipeline.start(config)

while True:
    event, values = window.read(timeout=0, timeout_key='timeout')
    # Wait for a coherent pair of frames: depth and color
    frame = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frame)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()


    if not depth_frame or not color_frame:
        continue
    
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    #print(np.mean(depth_image))
    color_image = np.asanyarray(color_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx = load_cam_params(cam1_name)


    # model prediction
    if values['tripod']:
        pred = model_tripod(color_image, conf=0.4, iou=0.1)
    else:
        pred = model_basic(color_image,iou=0.2)

    marked = pred[0].plot()  
    boxes = pred[0].boxes.xyxy.tolist()
    classes = pred[0].boxes.cls.tolist()
    confidences = pred[0].boxes.conf.tolist()
    fps = 1000/pred[0].speed["inference"]
    masks = pred[0].masks.xy
    window['text'].update("fps = {:.2f}".format(fps))
    gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)

    fx_c,fy_c,px_c,py_c = I_c_mtx[0][0],I_c_mtx[1][1],I_c_mtx[0][2], I_c_mtx[1][2]
    cam_params_c = [fx_c,fy_c,px_c,py_c]
    names = model_basic.names

    tags = at_detector.detect(gray,estimate_tag_pose=True,camera_params=cam_params_c,tag_size=0.15)

    if len(tags)>0:
        rotm_cw = tags[0].pose_R
        trasm_cw = tags[0].pose_t
        homo = tags[0].homography

    if values['test']:
        center_x = 650
        center_y = 450
        center_x_min = 600
        center_x_max = 700

        # center_y_min = int(box[1] + y_length/4)
        # center_y_max = int(box[1] + 3*y_length/4)

        center_y_min = 400
        center_y_max = 500

        cv2.rectangle(depth_colormap, (center_x_min, center_y_min), (center_x_max, center_y_max), (255, 255, 255), 2)


        point = [[center_x,center_y]]
        point = np.array(point,dtype='float')
        #dir_cam = projectPixelTo3dRay(I_d_mtx,point)
        dir_cam = pixel2ray(point,I_d_mtx,I_d_dist).reshape((3,1))
    
        dir_w = np.dot(rotm_cw.T, dir_cam)    
        
        depth = depth_image[center_y_min:center_y_max,center_x_min:center_x_max].astype(float)
        depth_scale = single_cam.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale

        depth2 = depth_frame.get_distance(center_x,center_y)
        
        valid_num = np.count_nonzero(depth)
        t_dist = np.sum(depth/valid_num)

        print('t:',t_dist)

    if values['tracker']:
        np_box = np.array(boxes)
        np_score = np.array(confidences).reshape((-1,1))
        np_class = np.array(classes).reshape((-1,1))
        detected_objects = np.concatenate((np_box,np_score,np_class), axis=1)
        tracked_objects = tracker.update(detected_objects)
        for idx,obj in enumerate(tracked_objects):
            x1,y1,x2,y2,tracked_id,cla,conf =  obj
            cv2.rectangle(depth_colormap,(int(x1),int(y1)),(int(x2),int(y2)),color=(255,0,0),thickness=3)
            cv2.putText(depth_colormap,str(tracked_id),(int(x2),int(y2)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(0,0,255))

            

            

    if values['chair']:
        for box, label, conf, mask in zip(boxes, classes, confidences, masks):
            if label == 56:
                if conf>=0.8:
                    # center_y = box[1] + (box[3] - box[1])*3/4 # row from up left
                    mask_points = np.int32([mask])
                    cv2.fillPoly(marked, mask_points, (255,0,0))
                    x_length = box[2] - box[0]
                    y_length = box[3] - box[1]

                    center_x_1 = box[0] + x_length /2 # col from up left
                    center_y_1 = box[3]

                    # colored image
                    cv2.circle(marked, (int(center_x_1),int(center_y_1)), 6, (0, 255, 255), 3)
                    point_1 = [[center_x_1,center_y_1]]
                    point_1 = np.array(point_1,dtype='float')
                    dir_cam_1 = pixel2ray(point_1, I_c_mtx, I_c_dist).reshape((3,1))

                    origin_w = -np.dot(rotm_cw.T, (np.array([[0],[0],[0]])- trasm_cw))
                    dir_w_1 = np.dot(rotm_cw.T, dir_cam_1)    
                    normal = np.array([[0,0,1]])
                    plane_x0 = np.array([[1,0,0]]).T

                    t1 = (np.dot(normal, origin_w)-np.dot(normal,plane_x0))/np.dot(normal,dir_w_1)
                    print('t1:',t1)
                    intersection_w_1 = (origin_w - t1 * dir_w_1)
                    intersection_w_1 = np.around(intersection_w_1,decimals=2)

                    text_str = str(intersection_w_1.tolist())
                    cv2.putText(marked,text_str,(int(center_x_1),int(center_y_1)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255))

                    ###########

                    seg_mask = np.zeros(depth_image.shape, dtype=bool)
                    cols, rows = zip(*mask_points[0])
                    seg_mask[rows,cols] = True
                    depth_scale = single_cam.get_device().first_depth_sensor().get_depth_scale()

                    masked_depth_image = np.where(seg_mask, depth_image, 0).astype(float) * depth_scale

                    # center
                    center_x = int(center_x_1)
                    center_y = int(box[1] + y_length/2)
                    #center_y = int(center_y_1 - y_length/2)
                    # center_x_min = int(box[0] + 3*x_length /8)
                    # center_x_max = int(box[0] + 5 * x_length /8)

                    # center_y_min = int(box[1] + y_length/4)
                    # center_y_max = int(box[1] + 3*y_length/4)

                    # center_y_min = int(center_y_1 - 5*y_length/8)
                    # center_y_max = int(center_y_1- 3*y_length/8)

                    # cv2.rectangle(depth_colormap, (center_x_min, center_y_min), (center_x_max, center_y_max), (255, 255, 255), 2)
                    cv2.rectangle(depth_colormap, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
                    cv2.circle(depth_colormap, (int(center_x),int(center_y)), 6, (0, 255, 255), 3)


                    # point = [[center_x,center_y]]
                    # point = np.array(point,dtype='float')
                    # #dir_cam = projectPixelTo3dRay(I_d_mtx,point)
                    # dir_cam = pixel2ray(point,I_d_mtx,I_d_dist).reshape((3,1))

                    point = np.array(mask_points,dtype='float')
                    # #dir_cam = projectPixelTo3dRay(I_d_mtx,point)
                    dir_cam = pixel2ray(point,I_d_mtx,I_d_dist).transpose(2,0,1).squeeze()
                    dir_cam = np.mean(dir_cam,axis=1,keepdims=True)
                    dir_w = np.dot(rotm_cw.T, dir_cam) 
                    
                    # depth = depth_image[center_y_min:center_y_max,center_x_min:center_x_max].astype(float)
                    # depth = depth * depth_scale


                    non_zero_masked_depth_image = masked_depth_image[masked_depth_image > 0]
                    low_percentile = np.percentile(non_zero_masked_depth_image, 10)
                    high_percentile = np.percentile(non_zero_masked_depth_image, 90)
                    filtered_depth_image = np.where(
                        (masked_depth_image >= low_percentile) & (masked_depth_image <= high_percentile),
                        masked_depth_image,
                        0
                    )

                    valid_num = np.count_nonzero(masked_depth_image)
                    t_dist = np.sum(filtered_depth_image) / valid_num

                    print('t:',t_dist)

                    intersection_w = origin_w - t_dist * dir_w
                    intersection_w = np.around(intersection_w,decimals=2)
    
                    text_str_temp = str(intersection_w.tolist())
                    cv2.putText(depth_colormap,text_str_temp,(int(center_x),int(center_y)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(255,255,255))

                    with open('value_rgbd.txt', 'w') as file:
                        file.write(str(text_str_temp))



        
    
    if values['tag']:

        
        for tag in tags:

            # if tag.families == "":
            #     origin = tag
            rotm = tag.pose_R
            trasm = tag.pose_t
            homo = tag.homography
            cv2.circle(marked, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2) # left-top
            cv2.circle(marked, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2) # right-top

            cv2.circle(marked, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2) # right-bottom
            cv2.circle(marked, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2) # left-bottom

            cv2.circle(marked, tuple(tag.center.astype(int)), 4, (255, 0, 0), 4) #标记apriltag码中心点
            # M, e1, e2 = at_detector.detection_pose(tag, cam_params)
 

            P = np.concatenate((rotm, trasm), axis=1) #相机投影矩阵

            P = np.matmul(I_c_mtx,P)
            x = np.matmul(P,np.array([[-1],[0],[0],[1]]))  
            x = x / x[2]
            y = np.matmul(P,np.array([[0],[-1],[0],[1]]))
            y = y / y[2]
            z = np.matmul(P,np.array([[0],[0],[-1],[1]]))
            z = z / z[2]
            cv2.line(marked, tuple(tag.center.astype(int)), tuple(x[:2].reshape(-1).astype(int)), (0,0,255), 2) #x轴红色
            cv2.line(marked, tuple(tag.center.astype(int)), tuple(y[:2].reshape(-1).astype(int)), (0,255,0), 2) #y轴绿色
            cv2.line(marked, tuple(tag.center.astype(int)), tuple(z[:2].reshape(-1).astype(int)), (255,0,0), 2) #z轴蓝色



    # GUI update
    rawbytes = cv2.imencode('.png', marked[::2,::2,:])[1].tobytes()
    window['color'].update(data=rawbytes)
    predbytes = cv2.imencode('.png', depth_colormap[::2,::2,:])[1].tobytes()
    window['depth'].update(data=predbytes)


    if event == 'Capture':
        cur_time = time.localtime()
        imgname = time.strftime("%H-%M%S.png",cur_time)
        cv2.imwrite(os.path.join(capture_root,imgname),marked)
    if event == 'Exit':
        break


    

pipeline.stop()
window.close()
