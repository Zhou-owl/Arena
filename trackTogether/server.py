import socket
import cv2
import numpy as np
import threading
import modules
embed_dim = 2764800
shape = [720, 1280, 3]
client_data = {}
global_map = None
server_ip = ""
tracker = {}
def handle_client(client_socket, client_address):
    global client_data
    recv_data_whole = bytes()
    while True:
        recv_data = client_socket.recv(3000000)  
        if len(recv_data) == 0:
            print('disconnect')
            break
        else:
            if client_address not in client_data:
                client_data[client_address] = bytes()
            client_data[client_address] += recv_data
            if len(client_data[client_address]) == embed_dim: 
#               embedding_vector = pickle.loads(recv_data)

                frame = np.frombuffer(client_data[client_address], dtype=np.uint8).reshape(shape)
                client_data[client_address] = bytes()
                # 在此处进行图像拼接或其他处理
                cv2.imshow(f'camera_received from {client_address}', frame)
                cv2.waitKey(1)
                client_socket.send("image has been received!".encode('utf-8'))

def overlap (box1, box2, thred=0.3):
    # box: [x+ x- y+ y- z+ z-]
    # return bool
def similarity(collected_3ds,self.boxes_list):
    # collected_3ds: {cam: [[class conf x+ x- y+ y- z+ z-]]}
    # boxes_list:{class: [conf, [xyz],1][conf, [xyz],0])...
    latest_box_list = {} # {class: [conf, [xyz],2]}
    for key, boxes in collected_3ds.items():
        for box in boxes: #[class conf x+ x- y+ y- z+ z-]
            cur_class = box[0]
            pre_boxes = boxes_list[cur_class] #[conf, [xyz],2],...,[conf, [xyz],1],...,[conf, [xyz],0],...
            for pb in pre_boxes:#[conf, [xyz],1]
                if overlap(box[2:],pb[1]):
                    # delete pb from boxes_list
                    # 


            



    
    # compute similarity and reduce dimention
    # output: [cla, embed]
    # https://github.com/MCG-NKU/SERE
def update_global_map(update_data):
    # output: [cla, 3dbboxes]
def start_server():
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = (server_ip, 8088)
    tcp_server_socket.bind(address)
    tcp_server_socket.listen(128)

    while True:
        client_socket, client_addr = tcp_server_socket.accept()
        print(f"Connection from {client_addr} has been established!")
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_addr))
        client_handler.start()


if __name__ == "__main__":
    init_server= threading.Thread(target=start_server)
    if client_data:
        global_map = update_global_map(similarity())
        for cla in global_map.keys():
            tracker[cla] =  modules.tracker()

        tracked_object = tracker[cla].update(global_map[cla])
    