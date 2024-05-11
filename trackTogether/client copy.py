import socket
import cv2
import numpy as np
import pickle
import torch
import rgbd_multicam
import datetime
from modules import *

# config
is_first_run = False
device_id = 0
boxes_conf_thred = 0.4
target_object_list = [0,5,39,56,58,67,73]

#init client 
tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = "192.168.1.113"
server_port = 8088
tcp_client_socket.connect((server_ip, server_port))

# init params
cam_id = find_devices(device_id)
cam_params_c, cam_params_dict = get_cam_params(cam_id,is_first_run)

# start streaming (single cam)
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
config.enable_device(cam_id)
cam_profiles = pipe.start(config)

# init gui
layout, window = init_gui()


# frame loop
try:
    while True:
        event, values = window.read(timeout=0, timeout_key='timeout')
        
        # get frames
        frame = pipe.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frame = align.process(frame)
        color_frame = aligned_frame.get_color_frame()
        depth_frame = aligned_frame.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # tobe changed
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # extrinsic mat
        tag = init_tag_detector(gray=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY), cam_params_c=cam_params_c)
        if tag:
            rotm_cw = tag.pose_R
            trasm_cw = tag.pose_t
            homo = tag.homography
        else:
            continue

        if values['detect']:
            # pred bbox [[cla, conf,[xmin, ymin, xmax ymax]],]
            bbox_model, pred, formated_bounding_boxes = bbox_2D_prediction(color_image, thred = boxes_conf_thred)
            [d_mtx, d_dist, c_dist, c_mtx] = cam_params_dict
            for count,fbb in enumerate(formated_bounding_boxes):
                label = fbb[0]
                conf = fbb[1]
                box = fbb[2]









        current_datetime = datetime.datetime.now()
        current_timestamp = current_datetime.timestamp()

        bboxes, depth, color = rgbd_multicam.generate_bbox()

        # seg = rgbd_multicam.genarate_seg()
        track = rgbd_multicam.generate_track()
        if not success:
            print('Video Reading Finished!')
            break

        # 在这里进行图像处理和嵌入

        # tobe changed
        p_enc_3d = PositionalEncodingPermute3D(11)

        embedding_vector = p_enc_3d(depth,color)
        concat_data = modules.dataconcat(bboxes,depth, color, track)
        output = clusterNet()

        # embedding_vector = np.random.rand(100)

        embedded_data = pickle.dumps(embedding_vector)

        tcp_client_socket.send(embedded_data)
        print("Embedding vector sent to server.")


        # tcp_client_socket.send(frame.tobytes())
        # # 接收服务器回传信息
        # recv_data = tcp_client_socket.recv(1024)
        # print(recv_data.decode('utf-8'))

finally:
    capture.release()
    cv2.destroyAllWindows()
    tcp_client_socket.close()