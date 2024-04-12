import socket
import cv2
import numpy as np
import pickle
import torch
import rgbd_multicam
import modules
from position_encode import PositionalEncodingPermute3D

p_enc_3d = PositionalEncodingPermute3D(11)
z = torch.zeros((1,11,5,6,4))
print(p_enc_3d(z).shape) # (1, 11, 5, 6, 4)
tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = "192.168.1.113"
server_port = 8088
tcp_client_socket.connect((server_ip, server_port))

# tobe changed
capture = cv2.VideoCapture(0)
cv2.namedWindow('camera', cv2.WINDOW_NORMAL)

try:
    while capture.isOpened():
        success, frame = capture.read()
        # 视频结束
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