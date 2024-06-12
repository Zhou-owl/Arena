import socket
import cv2
import numpy as np
import threading
import pickle
import math
import time
from collections import deque
from cluster import *
from arena import *


data_lock = threading.Lock()
class Server:
    def __init__(self, ip="127.0.0.1", port=8188):
        self.server_ip = ip
        self.port = port
        self.cur_frame_thred = 0.2
        self.data_change = False
        # [[timestamp,[cla,conf,[6 bounder]],[cla,conf,[6 bounder]]]] -> deque
        self.client_data = {} # cam_addr:[raw]
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
                with data_lock:
                    if client_address not in self.client_data:
                        self.client_data[client_address] = deque() 
                    self.client_data[client_address].append(new_frame_box) 
                    recv_data_whole = bytes()
                    current_timestamp = time.time()
                    reply_string = "received frame from "+str(client_address)+" at "+format(current_timestamp, ".2f") + " within " + format(current_timestamp - new_frame_box[0], ".2f") + " second"
                    print(reply_string)
                    client_socket.send(reply_string.encode('utf-8'))
                    self.data_change = True
            except pickle.UnpicklingError:
                # Data not fully received, continue accumulating
                print("fail to deserialize data")
            
    def start_server(self):
        threading.Thread(target=self.recv_client).start() # Start client connection thread
        print("Client connection thread started")
        threading.Thread(target=self.process_data).start()  # Start processing thread
        print("processing thread started")
    def recv_client(self):
        while True:
            client_socket, client_address = self.tcp_server_socket.accept()
            print(f"Connection from {client_address} has been established!")
            # every client has independent thread
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_handler.start()
    
    def process_data(self):
        while True:
            if not self.data_change:
                continue
            with data_lock:
                self.select_curr_frame() # move valid frame to cur_boxes_list
                # self.valid_curr_boxes() # filter valid boxes in cur_boxes_list
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



server = Server()
server.start_server()


while not server.cur_boxes_list:
    time.sleep(0.1)
# connect to arena scene
scene = Scene(host="arenaxr.org", namespace = "yongqiz2", scene="yolo")
@scene.run_forever(interval_ms=200)
def periodic():
    # cur_boxes_list -> cla: [[origin, conf, [bounders]],...]
    with data_lock:
        cur_boxes = server.cur_boxes_list
    box_list = []
    text_list = []
    for cla, cla_boxes in cur_boxes.items():
        for cla_box in cla_boxes:
            bounder =  cla_box[2]
            x_length = bounder[1] - bounder[0]
            x_center = (bounder[1] + bounder[0]) / 2
            z_length = bounder[3] - bounder[2]
            z_center = (bounder[3] + bounder[2]) / 2
            y_length = bounder[5] - bounder[4] # height
            y_center = (bounder[5] + bounder[4]) / 2
            box = Box(
                object_id = str(cla)+"box_conf"+str(cla_box[1]),
                position = (-x_center, y_center, -z_center),
                scale = (abs(x_length), abs(y_length), abs(z_length)),
                color = Color(200,0,200),
                material= Material(opacity=0.2, transparent=True, visible=True),
                persist=True
            )
            box_list.append(box)
            scene.add_object(box)
            my_text = Text(
                object_id=str(cla)+"text_conf"+str(cla_box[1]),
                text="str(cla)",
                align="center",
                position=(-x_center, y_center+y_length, -z_center),
                scale=(0.6,0.6,0.6),
                color=(100,255,255),
                persist = True
            )
            text_list.append(my_text)
            scene.add_object(my_text)

    time.sleep(0.2)
    for b,t in zip(box_list,text_list):
        scene.delete_object(b)
        scene.delete_object(t)


scene.run_tasks()

