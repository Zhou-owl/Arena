import socket
import cv2
import datetime
import numpy as np
import threading
import pickle
import math
import time
from collections import deque
from cluster import *


class Server:
    def __init__(self, ip="127.0.0.1", port=8088):
        self.server_ip = ip
        self.port = port
        self.cur_frame_thred = 0.2
        self.data_change = False
        # [[timestamp,[cla,conf,[6 bounder]],[cla,conf,[6 bounder]]]] -> deque
        self.client_data = {} # cam_addr:[raw]
        self.data_lock = threading.Lock()
        self.shape = [720, 1280, 3]
        self.tracker = {}
        self.global_map = {}
        self.pre_boxes_list = {}
        self.cur_boxes_list = {}
        self.overlap_iou = 0.6
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server_socket.bind((self.server_ip, self.port))
        self.tcp_server_socket.listen(128)
        print(f"Server started at {self.server_ip} on port {self.port}")

    def handle_client(self, client_socket, client_address):
        while True:
            recv_data_whole = b''
            length_data = client_socket.recv(4)
            if not length_data:
                break
            length = int.from_bytes(length_data, byteorder='big')
              # Initialize frame_data as empty bytes

            # Continue to receive data until the full frame is received
            while len(recv_data_whole) < length:
                packet = client_socket.recv(min(1024, length - len(recv_data_whole)))
                if not packet:
                    break
                recv_data_whole += packet
            if len(recv_data_whole) < length:
                print("Incomplete frame received")
            try:    
                new_frame_box = pickle.loads(recv_data_whole) # if all data are received
                # [timestamp, [cla, conf, [xyzbounder]]]...
                with self.data_lock:
                    if client_address not in self.client_data:
                        self.client_data[client_address] = deque() 
                    self.client_data[client_address].append(new_frame_box) 
                    recv_data_whole = bytes()
                    current_datetime = datetime.datetime.now()
                    current_timestamp = current_datetime.timestamp()
                    reply_string = "received "+format(new_frame_box[0], ".2f")+" from "+str(client_address)+" at "+format(current_timestamp, ".2f")
                    print(reply_string)
                    client_socket.send(reply_string.encode('utf-8'))
                    self.data_change = True
            except pickle.UnpicklingError:
                # Data not fully received, continue accumulating
                print("fail to deserialize data")
            
    def start_server(self):
        threading.Thread(target=self.process_data).start()  # Start the processing thread
        while True:
            client_socket, client_address = self.tcp_server_socket.accept()
            print(f"Connection from {client_address} has been established!")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_handler.start()

    def process_data(self):
        while True:
            if not self.data_change:
                continue
            with self.data_lock:
                self.select_curr_frame() # move valid frame to cur_boxes_list
                self.valid_curr_boxes() # filter valid boxes in cur_boxes_list


                self.data_change = False
                
                

    def select_curr_frame(self):
        # [[timestamp,[cla,conf,[6 bounder]],[cla,conf,[6 bounder]]]] -> deque

        # call this function to save cur frame data into cur_boxes_list
        min_timestamp = math.inf
        self.cur_boxes_list = {} #empty the cur list
        for queue in self.client_data.values(): # frame queue of each camera
            if queue and queue[0][0] < min_timestamp:  # Checking the first element's timestamp
                min_timestamp = queue[0][0]
        for client_address, queue in self.client_data.items():
            if queue and abs(queue[0][0] - min_timestamp) <= self.cur_frame_thred:
                frame_data = queue.popleft()
                #[timestamp,[cla,conf,[6 bounder]],[cla,conf,[6 bounder]],...] -> list
                boxes = frame_data[1:]
                for box in boxes:
                    cla, conf, bounders = box
                    if cla not in self.cur_boxes_list:
                        self.cur_boxes_list[cla] = []
                    self.cur_boxes_list[cla].append([client_address, conf, bounders])
        
    def valid_curr_boxes(self):
        # call this after update cur_boxes_list
        # cur_boxes_list -> dict -> cla:[[origin, conf, [6bounders]],[]]
        for cla,boxes in self.cur_boxes_list.items():
            # boxes detected within a timethred, from multipal origins.
            n = len(boxes)
            valid_boxes = []
            boxes.sort(key=lambda x: x[1], reverse=True) # sort boxes by conf from high to low
            skip = set() # skip after merge
            for i in range(n):
                if i in skip:
                    continue
                merged = [] # id of boxes merged by box i
                sum_conf = 1 # conf of all merged boxes, set the max conf to 1  -> Mark
                for j in range(i+1,n):
                    if j in skip:
                        continue
                    iou = iou_3d(boxes[i], boxes[j])
                    # Different origin, same class, IOU > overlap_iou
                    if boxes[i][0] != boxes[j][0]:
                        if iou >= self.overlap_iou:
                            merged.append(j)
                            skip.add(j)
                    # todo: set 2 iou threds and apply color similarity
                num = len(merged)
                new_bounders = boxes[i][2]
                for m in merged:
                    this_conf = boxes[j][1]
                    sum_conf += this_conf
                    this_bounders = boxes[m][2]
                    new_bounders = [x + this_conf * y for x, y in zip(new_bounders, this_bounders)]
                
                new_bounders = [i/sum_conf for i in new_bounders]
                new_conf = sum_conf / num
                valid_boxes.append([new_conf, new_bounders])


            self.cur_boxes_list[cla] = valid_boxes
            # cur_boxes_list -> cla: [[conf, [bounders]],...]



    def integerate_pre_boxes(self):
        # rewrite/apply Sort here
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



    

# Usage
if __name__ == '__main__':
    server = Server()
    server.start_server()