import os
import time
import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
import PySimpleGUI as sg  # pip install pysimplegui
from dt_apriltags import Detector
from PIL import Image
from ultralytics import YOLO
from oc_sort import OCSort
from util import *
from examples import *


is_first_run = False
conf_thred = 0.5

# define model
model_tripod = YOLO('./runs/detect/train/weights/best.pt')  
model_basic = YOLO('./yolov8l-seg.pt')  

# init tag detector
at_detector = Detector(families='tag36h11')

# init cameras 
device_list = rs.context()
device_serials = [i.get_info(rs.camera_info.serial_number) for i in device_list.devices]


# init camera intrinsic params
def init_params():
    default_config = rs.config()
    device_manager = DeviceManager(device_list, default_config)
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

if is_first_run:
    init_params()

def load_multicam_params():
    matrix_dict = {}
    for dev in device_serials:
        with np.load(dev+".npz") as X:
            E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx = [X[i] for i in ('E_dr', 'I_d_mtx', 'I_d_dist', 'I_c_dist', 'I_c_mtx')]
            matrix_dict[dev] = [E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx]
    return matrix_dict

cam_params = load_multicam_params()


# init gui
layout = [
    [sg.Image(filename='', key='color0',size=(640,360)),sg.Image(filename='', key='color6',size=(640,360))],
    [sg.Text('fps = 0',  key="text0"),sg.Text('fps = 0',  key="text6")],
    [sg.Image(filename='', key='depth0',size=(640,360)),sg.Image(filename='', key='depth6',size=(640,360))],
    [sg.Checkbox('Tripod', key='tripod')],
    [sg.Checkbox('FindTag', key='tag')],
    [sg.Checkbox('Detect Objects', key='detect')],
    [sg.Checkbox('Chair only', key='chair')],
    [sg.Checkbox('Single tracker', key='tracker')],
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

# init oc-sort tracker
tracker = {}
for cam_id in device_serials:
    tracker[cam_id] = OCSort(det_thresh=0.7,iou_threshold=0.5)


# start streaming
pipelines = {}
cam_profiles = {}
config = {}
rotm_cw = {}
trasm_cw = {}
homo = {}

for cam_id in device_serials:
    pipelines[cam_id] = rs.pipeline()
    config[cam_id] = rs.config()
    config[cam_id].enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config[cam_id].enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    config[cam_id].enable_device(cam_id)
    cam_profiles[cam_id] = pipelines[cam_id].start(config[cam_id])

while True:
    event, values = window.read(timeout=0, timeout_key='timeout')

    # init frames, tags, preds for all cams
    color_frames = {}
    depth_frames = {}
    depth_colormaps = {}
    cam_tags = {}
    preds = {}
    markeds = {}
    for cam_id, pipe in pipelines.items():
        frame = pipe.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frame = align.process(frame)
        color_frame = aligned_frame.get_color_frame()
        depth_frame = aligned_frame.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        #print(np.mean(depth_image))
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        color_frames[cam_id] = color_image
        depth_frames[cam_id] = depth_image
        depth_colormaps[cam_id] = depth_colormap

        # model to use
        if values['tripod']:
            pred = model_tripod(color_image, conf=0.4, iou=0.1)
        else:
            pred = model_basic(color_image,iou=0.2)

        preds[cam_id] = pred
        marked = pred[0].plot()  
        markeds[cam_id] = marked

        gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)

        [E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx] = cam_params[cam_id]

        fx_c,fy_c,px_c,py_c = I_c_mtx[0][0],I_c_mtx[1][1],I_c_mtx[0][2], I_c_mtx[1][2]
        cam_params_c = [fx_c,fy_c,px_c,py_c]

        tag = at_detector.detect(gray,estimate_tag_pose=True,camera_params=cam_params_c,tag_size=0.15)
        cam_tags[cam_id] = tag


    # no valid camera frame
    if len(depth_colormaps) == 0 or len(color_frames) == 0:
        continue

    depth_colormap_dim = list(depth_colormaps.values())[0].shape
    color_colormap_dim = list(color_frames.values())[0].shape

    # init extrinsic params using tag

    if values['tag']:
        # only refer to the first detected tag 
        for cam_id, tags in cam_tags.items():
            if len(tags) > 0:
                tag = tags[0]
            else:
                continue

            rotm_cw[cam_id] = tag.pose_R
            trasm_cw[cam_id] = tag.pose_t
            homo[cam_id] = tag.homography

            pred = preds[cam_id]
            marked = markeds[cam_id]
            cv2.circle(marked, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2) # left-top
            cv2.circle(marked, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2) # right-top

            cv2.circle(marked, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2) # right-bottom
            cv2.circle(marked, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2) # left-bottom

            cv2.circle(marked, tuple(tag.center.astype(int)), 4, (255, 0, 0), 4) #标记apriltag码中心点
            # M, e1, e2 = at_detector.detection_pose(tag, cam_params)
 
            [E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx] = cam_params[cam_id]
            P = np.concatenate((rotm_cw[cam_id], trasm_cw[cam_id]), axis=1) #相机投影矩阵

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
 
    if values['detect']:
        for cam_id, pred in preds.items():
            # only process detection when tag is detected
            if cam_id not in rotm_cw.keys():
                continue
            
            c_mtx = cam_params[cam_id][4]
            c_dist = cam_params[cam_id][3]
            d_mtx = cam_params[cam_id][1]
            d_dist = cam_params[cam_id][2]
            rot = rotm_cw[cam_id]
            tras = trasm_cw[cam_id]

            marked = markeds[cam_id]
            boxes = pred[0].boxes.xyxy.tolist() # bounding box of all detected objects
            classes = pred[0].boxes.cls.tolist()    # class number of all detected objects
            confidences = pred[0].boxes.conf.tolist()
            fps = 1000/pred[0].speed["inference"]
            masks = pred[0].masks.xy    # segmentation points of all detected objects
            
            window['text'+cam_id[-1]].update("fps = {:.2f}".format(fps))

            for box, label, conf, mask in zip(boxes, classes, confidences, masks):
                if values['chair']:
                    if label != 56:
                        continue

                # todo: change conf to slide bar
                if conf>=conf_thred:
                    # center_y = box[1] + (box[3] - box[1])*3/4 # row from`` up left
                    mask_points = np.int32([mask])
                    cv2.fillPoly(marked, mask_points, (255,0,0))
                    x_length = box[2] - box[0]
                    y_length = box[3] - box[1]

                    center_x_1 = box[0] + x_length /2 # col from up left
                    center_y_1 = box[3]


                    cv2.circle(marked, (int(center_x_1),int(center_y_1)), 6, (0, 255, 255), 3)
                    point_1 = [[center_x_1,center_y_1]]
                    point_1 = np.array(point_1,dtype='float')
                    dir_cam_1 = pixel2ray(point_1, c_mtx, c_dist).reshape((3,1))

                    origin_w = -np.dot(rot.T, (np.array([[0],[0],[0]])- tras))
                    dir_w_1 = np.dot(rot.T, dir_cam_1)    
                    normal = np.array([[0,0,1]])
                    plane_x0 = np.array([[1,0,0]]).T

                    t1 = (np.dot(normal, origin_w)-np.dot(normal,plane_x0))/np.dot(normal,dir_w_1)
                    print('t1:',t1)
                    intersection_w_1 = (origin_w - t1 * dir_w_1)
                    intersection_w_1 = np.around(intersection_w_1,decimals=2)

                    text_str = str(intersection_w_1.tolist())
                    cv2.putText(marked,text_str,(int(center_x_1),int(center_y_1)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255))

                    ###########

                    seg_mask = np.zeros(depth_frames[cam_id].shape, dtype=bool)
                    cols, rows = zip(*mask_points[0])
                    seg_mask[rows,cols] = True
                    depth_scale = cam_profiles[cam_id].get_device().first_depth_sensor().get_depth_scale()

                    masked_depth_image = np.where(seg_mask, depth_frames[cam_id], 0).astype(float) * depth_scale


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
                    cv2.rectangle(depth_colormaps[cam_id], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 2)
                    cv2.circle(depth_colormaps[cam_id], (int(center_x),int(center_y)), 6, (0, 255, 255), 3)


                    # point = [[center_x,center_y]]
                    # point = np.array(point,dtype='float')
                    # #dir_cam = projectPixelTo3dRay(I_d_mtx,point)
                    # dir_cam = pixel2ray(point,I_d_mtx,I_d_dist).reshape((3,1))

                    point = np.array(mask_points,dtype='float')
                    # #dir_cam = projectPixelTo3dRay(I_d_mtx,point)
                    dir_cam = pixel2ray(point,d_mtx,d_dist).transpose(2,0,1).squeeze()
                    dir_cam = np.mean(dir_cam,axis=1,keepdims=True)
                    dir_w = np.dot(rot.T, dir_cam) 
                    
                    # depth = depth_image[center_y_min:center_y_max,center_x_min:center_x_max].astype(float)
                    # depth = depth * depth_scale
                    
                    valid_num = np.count_nonzero(masked_depth_image)
                    t_dist = np.sum(masked_depth_image/valid_num)

                    print('t:',t_dist)

                    intersection_w = origin_w - t_dist * dir_w
                    intersection_w = np.around(intersection_w,decimals=2)
    
                    text_str_temp = str(intersection_w.tolist())
                    cv2.putText(depth_colormaps[cam_id],text_str_temp,(int(center_x),int(center_y)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(255,255,255))

                    with open(cam_id+'_detect.txt', 'w') as file:
                        file.write(str(text_str_temp))



        

    if values['tracker']:
        for cam_id, pred in preds.items():
            boxes = pred[0].boxes.xyxy.tolist() # bounding box of all detected objects
            classes = pred[0].boxes.cls.tolist()    # class number of all detected objects
            confidences = pred[0].boxes.conf.tolist()
            np_box = np.array(boxes)
            np_score = np.array(confidences).reshape((-1,1))
            np_class = np.array(classes).reshape((-1,1))
            detected_objects = np.concatenate((np_box,np_score,np_class), axis=1)
            tracked_objects = tracker[cam_id].update(detected_objects)
            for idx,obj in enumerate(tracked_objects):
                x1,y1,x2,y2,tracked_id,cla,conf =  obj
                cv2.rectangle(depth_colormaps[cam_id],(int(x1),int(y1)),(int(x2),int(y2)),color=(255,0,0),thickness=3)
                cv2.putText(depth_colormaps[cam_id],str(tracked_id),(int(x2),int(y2)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(0,0,255))

            

            



    # GUI update
    for cam_id in markeds.keys():
        rawbytes = cv2.imencode('.png', markeds[cam_id][::2,::2,:])[1].tobytes()
        window['color'+cam_id[-1]].update(data=rawbytes)
        predbytes = cv2.imencode('.png', depth_colormaps[cam_id][::2,::2,:])[1].tobytes()
        window['depth'+cam_id[-1]].update(data=predbytes)

    capture_root = './capture'

    if event == 'Capture':
        cur_time = time.localtime()
        imgname = time.strftime("%H-%M%S.png",cur_time)
        cv2.imwrite(os.path.join(capture_root,imgname),marked)
    if event == 'Exit':
        break


    
for cam_id, pipe in pipelines.items():
    pipe.stop()
window.close()
