from arena import *
import random
import time
import json
# setup library
scene = Scene(host="arenaxr.org", namespace = "yongqiz2", scene="yolo")

# make a box

scale_map = {
    "chair":(.45,.9,.45),
    "person":(.3,1.5,.3),
    "default":(.3,.3,.3)
}
color_map = {
    "chair":[200,0,200],
    "person":[250,0,0],
    "default":[0,200,200]
}



@scene.run_forever(interval_ms=200)
def periodic():
    box_list = []
    text_list = []
    delet_list = []
    obj_list = scene.all_objects

    '''
    for k in obj_list.keys():
        #print(type(obj_list[k]["object_id"]))
        if(len(obj_list[k]["object_id"]) < 4):
            delet_list.append(obj_list[k]["object_id"])
    for i in delet_list:
        obj = scene.get_persisted_obj(i)
        scene.delete_object(obj)
    '''
        
    with open("value.txt",'r') as file:
        box_string = file.readlines()
        for idx, i in enumerate(box_string):
            if '|' not in i:
                continue
            pos = i.split('|')[0]
            class_name = i.split('|')[1].split("\n")[0]
            try:
                values = pos.split(']')
                x = float(values[0].split('[')[-1])
                y = float(values[2].split('[')[-1])
                z = float(values[1].split('[')[-1])
            
                if type(x)==float and type(y)==float and type(z)==float:

                    if class_name in scale_map.keys():
                        box_size = scale_map[class_name]
                        box_color = color_map[class_name]
                    else:
                        box_size = scale_map["default"]
                        box_color = color_map["default"]

                    box = Box(
                        object_id=str(idx), 
                        position=(-x,y+box_size[1]*0.5,-z), 
                        scale=box_size,
                        color = Color(box_color[0],box_color[1],box_color[2]),
                        material = Material(opacity=0.2, transparent=True, visible=True),
                        persist=True
                    )
                    box_list.append(box)
                    scene.add_object(box)
                    my_text = Text(
                        object_id=str(-idx-1),
                        text=class_name,
                        align="center",
                        position=(-x,y+box_size[1]+0.15,-z),
                        scale=(0.6,0.6,0.6),
                        color=(100,255,255),
                        persist = True
                    )
                    text_list.append(my_text)
                    scene.add_object(my_text)
            except Exception as e:
                print(e)
    
    time.sleep(0.19)
    for b,t in zip(box_list,text_list):
        scene.delete_object(b)
        scene.delete_object(t)

 


# start tasks
scene.run_tasks()

