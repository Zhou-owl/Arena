import socket
import cv2
import numpy as np
import threading
import pickle
import math
from collections import deque
from cluster import *

high_iou = 0.8
low_iou = 0.2
feature_thred = 0.8
class Server:
    def __init__(self, ip='', port=8088):
        self.server_ip = ip
        self.port = port
        self.client_data = {} # cam_addr:[raw]
        self.shape = [720, 1280, 3]
        self.tracker = {}
        self.global_map = {}
        self.pre_boxes_list = {}
        self.cur_boxes_list = {}
        self.overlap_iou = 0.5
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server_socket.bind((self.server_ip, self.port))
        self.tcp_server_socket.listen(128)
        print(f"Server started at {self.server_ip} on port {self.port}")

    def handle_client(self, client_socket, client_address):
        recv_data_whole = bytes()
        while True:
            recv_data = client_socket.recv(3000000)
            if len(recv_data) == 0:
                print('Client disconnected:', client_address)
                break
            recv_data_whole += recv_data
            try:
                new_frame_box = pickle.loads(recv_data_whole) # if all data are received
                # [timestamp, [cla, conf, xyzbounder]]...
                if client_address not in self.client_data:
                    self.client_data[client_address] = deque() 
                self.client_data[client_address].append(new_frame_box) 
                recv_data_whole = bytes()
                client_socket.send("Image has been received!".encode('utf-8'))
            except pickle.UnpicklingError:
                # Data not fully received, continue accumulating
                continue
            
    def start_server(self):
        while True:
            client_socket, client_address = self.tcp_server_socket.accept()
            print(f"Connection from {client_address} has been established!")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_handler.start()
    
    def select_curr_frame(self, thred = 0.1):
        # call this function to save cur frame data into cur_boxes_list
        min_timestamp = math.inf
        self.cur_boxes_list = {} #empty the cur list
        for queue in self.client_data.values():
            if queue and queue[0][0] < min_timestamp:  # Checking the first element's timestamp
                min_timestamp = queue[0][0]
        for client_address, queue in self.client_data.items():
            if queue and abs(queue[0][0] - min_timestamp) <= thred:
                frame_data = queue.popleft()
                timestamp, *boxes = frame_data
                for box in boxes:
                    cla, conf, xmax,xmin,ymax,ymin,zmax,zmin = box
                    if cla not in self.cur_boxes_list:
                        self.cur_boxes_list[cla] = []
                        self.cur_boxes_list[cla].append([client_address, conf, xmax,xmin,ymax,ymin,zmax,zmin])
        
    def valid_curr_boxes(self):
        # call this after update cur_boxes_list
        for cla,boxes in self.cur_boxes_list.items():
            n = len(boxes)
            boxes.sort(key=lambda x: x[1], reverse=True) # sort boxes by conf from high to low
            skip = set()
            merged = []

            for i in range(n):
                if i in skip:
                    continue
                current = boxes[i]
                for j in range(i+1,n):
                    if j in skip:
                        continue
                    iou = iou_3d(boxes[i], boxes[j])
                    #Different origin, same class, IOU > 0.2
                    if boxes[i][0] != boxes[j][0]:
                        if iou >= high_iou:
                            is_fake = True
                            # Create an average box
                            average_box = [
                                current[0],  # higher conf origin
                                cla,  # class
                                max(boxes[i][1], boxes[j][1]),  # max confidence
                                (boxes[i][2]*boxes[i][1] + boxes[j][2]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][3]*boxes[i][1] + boxes[j][3]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][4]*boxes[i][1] + boxes[j][4]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][5]*boxes[i][1] + boxes[j][5]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][6]*boxes[i][1] + boxes[j][6]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][7]*boxes[i][1] + boxes[j][7]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                            ]
                            current = average_box
                            skip.add(boxes[j])
                        elif iou>low_iou:
                            if feature_similarity(boxes[i][8],boxes[j][8]) > feature_thred:
                                average_box = [
                                current[0],  # higher conf origin
                                cla,  # class
                                max(boxes[i][1], boxes[j][1]),  # max confidence
                                (boxes[i][2]*boxes[i][1] + boxes[j][2]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][3]*boxes[i][1] + boxes[j][3]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][4]*boxes[i][1] + boxes[j][4]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][5]*boxes[i][1] + boxes[j][5]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][6]*boxes[i][1] + boxes[j][6]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                                (boxes[i][7]*boxes[i][1] + boxes[j][7]*boxes[j][1]) / (boxes[i][1]+boxes[j][1]), 
                            ]
                            current = average_box
                            skip.add(boxes[j])
                merged.append(current)
            self.cur_boxes_list[cla] = merged

    def integerate_pre_boxes(self):
        # call this after validate cur_boxes_list
        if not self.pre_boxes_list:
            self.pre_boxes_list = self.cur_boxes_list
            for cla,boxes in self.pre_boxes_list.items():
                self.pre_boxes_list[cla] = [[boxes,-1]]
        
        else:
            # process cur and pre



            # move cur to pre and delete earlist
            for cla, boxes in self.pre_boxes_list.values(): #boxes: [[boxes,-1],...,[boxes,-2],...]
                boxes = [[x[0],-2] if x[1]==-1 else x for x in boxes if x[1]!=-2]
                    
            for cla, boxes in self.cur_boxes_list.values():
                self.pre_boxes_list[cla].append([boxes,-1])
        
        
        
    
        
            
    
                

                        





    def overlap(self, box1, box2, threshold=0.5):
        # Example check for simple bounding box overlap in one dimension
        # This needs to be extended based on actual box dimensions and the correct geometry
        return abs(box1[0] - box2[0]) < threshold

    def similarity(self):
        # call this after update cur_boxes_list
        # collected_3ds: {cam: [[class conf x+ x- y+ y- z+ z-]]}
        # boxes_list:{class: [conf, [xyz],1][conf, [xyz],0])...
    
        latest_box_list = {} # {class: [conf, [xyz],2]}
        if 
        for key, boxes in collected_3ds.items():
            for box in boxes: #[class conf x+ x- y+ y- z+ z-]
                cur_class = box[0]
                pre_boxes = self.boxes_list[cur_class] #[conf, [xyz],2],...,[conf, [xyz],1],...,[conf, [xyz],0],...
                
                for pb in pre_boxes:#[conf, [xyz],1]

                    if overlap(box[2:],pb[1]):
                        # delete pb from boxes_list

    def update_global_map(self, update_data):
        # Update the global map with new data
        # Example to integrate or update information in the global map
        pass



    

# Usage
if __name__ == '__main__':
    server = Server()
    server.start_server()