import time
import torch
import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
import PySimpleGUI as sg  # pip install pysimplegui
from dt_apriltags import Detector
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from util import *
from mmdet.apis import init_detector, inference_detector
from examples import *

# with np.load("./intrinsic_calib.npz") as X:
#     mtx, dist = [X[i] for i in ('mtx', 'dist')]

def bbox_2D_prediction(input_frame, thred = 0.3):
    # change this function to swith bbox detector
    # input_frame: color image
    # return: bbox - [[cla, conf,[xmin, ymin, xmax ymax]],]
    config_file = './yolox_l_8x8_300e_coco.py'
    checkpoint_file = './yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda')
    pred = inference_detector(model,input_frame)
    for i,cla in enumerate(pred):
        if len(cla):
            for each in cla[:]:
                if each[-1] > thred:
                    pred[i].remove(each)
    return model, pred
def obj_segmentation(pred):
    # change this function to swith bbox segmentation
def find_devices(id=0):
    device_list = rs.context()
    device_serials = [i.get_info(rs.camera_info.serial_number) for i in device_list.devices]
    for i,dev in enumerate(device_serials):
        print("found device: ", i)
        print("  --serials: ", dev)

    return device_serials[id]

# init camera intrinsic params
def init_params():
    device_list = rs.context()
    device_serials = [i.get_info(rs.camera_info.serial_number) for i in device_list.devices]

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

def get_cam_params(cam_id, is_first_run):
    if is_first_run:
        init_params()
    with np.load(cam_id+".npz") as X:
        I_d_mtx, I_d_dist, I_c_dist, I_c_mtx = [X[i] for i in ('I_d_mtx', 'I_d_dist', 'I_c_dist', 'I_c_mtx')]
        matrix_dict = [I_d_mtx, I_d_dist, I_c_dist, I_c_mtx]
        fx_c,fy_c,px_c,py_c = I_c_mtx[0][0],I_c_mtx[1][1],I_c_mtx[0][2], I_c_mtx[1][2]
        cam_params_c = [fx_c,fy_c,px_c,py_c]
        return cam_params_c, matrix_dict

def load_multicam_params():
    # init cameras 
    device_list = rs.context()
    device_serials = [i.get_info(rs.camera_info.serial_number) for i in device_list.devices]
    matrix_dict = {}
    for dev in device_serials:
        with np.load(dev+".npz") as X:
            E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx = [X[i] for i in ('E_dr', 'I_d_mtx', 'I_d_dist', 'I_c_dist', 'I_c_mtx')]
            matrix_dict[dev] = [E_dr, I_d_mtx, I_d_dist, I_c_dist, I_c_mtx]
    return matrix_dict

# cam_params = load_multicam_params()

def init_gui():
    layout = [
        [sg.Image(filename='', key='color0',size=(640,360)),sg.Image(filename='', key='color6',size=(640,360))],
        [sg.Text('fps = 0',  key="text0"),sg.Text('fps = 0',  key="text6")],
        [sg.Image(filename='', key='depth0',size=(640,360)),sg.Image(filename='', key='depth6',size=(640,360))],
        [sg.Checkbox('Tripod', key='tripod')],
        [sg.Checkbox('FindTag', key='tag')],
        [sg.Checkbox('2d bbox detection', key='detect')],
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
    return layout, window

def init_tag_detector(gray, cam_params_c):
    at_detector = Detector(families='tag36h11')
    tags = at_detector.detect(gray,estimate_tag_pose=True,camera_params=cam_params_c,tag_size=0.15)
    if len(tags) > 0:
        return tags[0]
    return None

def init_segment_predictor():

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)



config_file = './yolox_l_8x8_300e_coco.py'
checkpoint_file = './yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
model = init_detector(config_file, checkpoint_file, device='cuda')


at_detector = Detector(families='tag36h11')

capture_root = './capture'
layout = [
    [sg.Image(filename='', key='raw',size=(500,400))],
    [sg.Text('fps = 0',  key="text")],
    [sg.Image(filename='', key='pred')],
    [sg.Radio('None', 'Radio', True, size=(10, 1))],
    [sg.Checkbox('Tripod', key='tripod')],
    [sg.Checkbox('Calibration', key='calib')],
    [sg.Checkbox('FindTag', key='tag')],
    [sg.Checkbox('Chair (check Tag and uncheck Calib)', key='chair')],
    [sg.Button('Capture', size=(20, 3))],
    [sg.Button('Exit', size=(20, 3))]
]

window = sg.Window('camera',
            layout,
            location=(1000, 500),
            resizable=True,
            element_justification='c',
            font=("Arial Bold",20),
            finalize=True)

cap = cv2.VideoCapture("/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_455_Intel_R__RealSense_TM__Depth_Camera_455_247523061064-video-index0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  

count=1

def bbox_predict(frame):
    dst = frame
    bounding_boxes = []
    bboxes_3d = []
    # model prediction

    pred = inference_detector(model,dst)
    for cla in pred:
        if len(cla):
            for each in cla:
                if each[-1] > conf_thred:
                    bounding_boxes.append(each[:4])
    #model.show_result(dst, pred, wait_time=1)
    img = model.show_result(dst, pred,wait_time=0,score_thr=0.4)
    input_boxes = torch.tensor(bounding_boxes, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])

    masks = []
    if len(transformed_boxes):
        image = predictor.set_image(img,"RGB")

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        for mask in masks:
            mask = mask.cpu().numpy()
            color = np.array([30, 144, 255])
            h, w = mask.shape[-2:]
            maskcolor = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            masked_image = np.where(maskcolor==0,img, maskcolor * 0.6 + img * 0.4)
            print(np.max(masked_image))
            img = masked_image
    return img, masks

while count:
    event, values = window.read(timeout=0, timeout_key='timeout')
    #read from cam
    ret, frame = cap.read()

    
    img, masks = bbox_predict(frame)
    try:
        img = img * masks
    except:
        pass
    rawbytes = cv2.imencode('.png', frame[::2,::2,:])[1].tobytes()
    window['raw'].update(data=rawbytes)
    predbytes = cv2.imencode('.png', img)[1].tobytes()
    window['pred'].update(data=predbytes)
cap.release()

#     marked = pred[0].plot()  
#     boxes = pred[0].boxes.xyxy.tolist()
#     classes = pred[0].boxes.cls.tolist()
#     confidences = pred[0].boxes.conf.tolist()
#     fps = 1000/pred[0].speed["inference"]
#     window['text'].update("fps = {:.2f}".format(fps))
#     gray = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)
#     fx,fy,cx,cy = mtx[0][0],mtx[1][1],mtx[0][2], mtx[1][2]
#     cam_params = [fx, fy, cx, cy]
#     tags = at_detector.detect(gray,estimate_tag_pose=True,camera_params=cam_params,tag_size=0.15)

#     if len(tags)>0:
#         rotm = tags[0].pose_R
#         trasm = tags[0].pose_t
#         homo = tags[0].homography

#     if values['chair']:
#         for box, label, conf in zip(boxes, classes, confidences):
#             if label == 56:
#                 if conf>=0.8:
#                     center_x = box[0] + (box[2] - box[0]) /2 # col from up left
#                     # center_y = box[1] + (box[3] - box[1])*3/4 # row from up left
#                     center_y = box[3]
#                     cv2.circle(marked, (int(center_x),int(center_y)), 6, (0, 255, 255), 3)
#                     point = [[center_x,center_y]]
#                     point = np.array(point,dtype='float')
#                     dir_cam = pixel2ray(point, mtx, dist).reshape((3,1))
#                     origin_w = -np.dot(rotm.T, (np.array([[0],[0],[0]])- trasm))
#                     dir_w = np.dot(rotm.T, dir_cam)
#                     normal = np.array([[0,0,1]])
#                     plane_x0 = np.array([[1,0,0]]).T
#                     t = (np.dot(normal, origin_w)-np.dot(normal,plane_x0))/np.dot(normal,dir_w)
#                     intersection_w = (origin_w - t * dir_w)
#                     intersection_w = np.around(intersection_w,decimals=2)
    
#                     text_str = str(intersection_w.tolist())
#                     cv2.putText(marked,text_str,(int(center_x),int(center_y)),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255))
#                     with open('value1.txt', 'w') as file:
#                         file.write(str(text_str))
#                 else:
#                     with open('value1.txt', 'w') as file:
#                         file.write("no valid chairs")




    
#     if values['tag']:

        
#         for tag in tags:

#             # if tag.families == "":
#             #     origin = tag
#             rotm = tag.pose_R
#             trasm = tag.pose_t
#             homo = tag.homography
#             cv2.circle(marked, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2) # left-top
#             cv2.circle(marked, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2) # right-top

#             cv2.circle(marked, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2) # right-bottom
#             cv2.circle(marked, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2) # left-bottom

#             # intersection_c_list = []

#             # for c in tag.corners:
#             #     point = c
#             #     point = np.array(point,dtype='float')
#             #     dir_cam = pixel2ray(point, mtx, dist).reshape((3,1))
#             #     origin_w = -np.dot(rotm.T, (np.array([[0],[0],[0]])- trasm))
#             #     dir_w = np.dot(rotm.T, dir_cam)
#             #     normal = np.array([[0,0,1]])
#             #     plane_x0 = np.array([[1,0,0]]).T
#             #     t = (np.dot(normal, origin_w)-np.dot(normal,plane_x0))/np.dot(normal,dir_w)
#             #     intersection_w = (origin_w - t * dir_w)
#             #     intersection_w = np.around(intersection_w,decimals=5)
#             #     intersection_c_list.append(intersection_w)

#             # scale_x = (abs(2/(intersection_c_list[1][0] - intersection_c_list[0][0])) + abs(2/(intersection_c_list[3][0] - intersection_c_list[2][0])))/2
#             # scale_y = (abs(2/(intersection_c_list[3][1] - intersection_c_list[0][1])) + abs(2/(intersection_c_list[2][1] - intersection_c_list[3][1])))/2


#             cv2.circle(marked, tuple(tag.center.astype(int)), 4, (255, 0, 0), 4) #标记apriltag码中心点
#             # M, e1, e2 = at_detector.detection_pose(tag, cam_params)
 

#             P = np.concatenate((rotm, trasm), axis=1) #相机投影矩阵

#             P = np.matmul(mtx,P)
#             x = np.matmul(P,np.array([[-1],[0],[0],[1]]))  
#             x = x / x[2]
#             y = np.matmul(P,np.array([[0],[-1],[0],[1]]))
#             y = y / y[2]
#             z = np.matmul(P,np.array([[0],[0],[-1],[1]]))
#             z = z / z[2]
#             cv2.line(marked, tuple(tag.center.astype(int)), tuple(x[:2].reshape(-1).astype(int)), (0,0,255), 2) #x轴红色
#             cv2.line(marked, tuple(tag.center.astype(int)), tuple(y[:2].reshape(-1).astype(int)), (0,255,0), 2) #y轴绿色
#             cv2.line(marked, tuple(tag.center.astype(int)), tuple(z[:2].reshape(-1).astype(int)), (255,0,0), 2) #z轴蓝色



    # GUI update
    

#     if event == 'Capture':
#         cur_time = time.localtime()
#         imgname = time.strftime("%H-%M%S.png",cur_time)
#         cv2.imwrite(os.path.join(capture_root,imgname),marked)
#     if event == 'Exit':
#         break


    

# window.close()


