from arena import *
import random

cam_num = 1
# setup library
scene = Scene(host="arenaxr.org", namespace = "yongqiz2", scene="yolo")

# make a box
box = Box(
    object_id="myBox", 
    position=(0,0.45,-3), 
    scale=(.45,.9,.45),
    color = Color(200,0,200),
    material = Material(opacity=0.2, transparent=True, visible=True),
    persist=True
)
scene.add_object(box)

my_text = Text(
    object_id="my_text",
    text="chair",
    align="center",
    position=(0,1.0,0),
    
    scale=(1.0,1.0,1.0),
    color=(100,255,255),

    persist = True

)
scene.add_object(my_text)



@scene.run_forever(interval_ms=200)
def periodic():
    x_list = []
    y_list = []
    z_list = []
    for i in range(cam_num):
        with open('../value_rgbd.txt'.format(i), 'r') as file:
            try:
                values = file.read().split(']')
                temp_x = float(values[0].split('[')[-1])
                temp_y = float(values[2].split('[')[-1])
                temp_z = float(values[1].split('[')[-1])
                if type(temp_x)==float and type(temp_y)==float and type(temp_z)==float:
                    x_list.append(temp_x)
                    y_list.append(temp_y)
                    z_list.append(temp_z)
            except Exception as e:
                print(e)
    x = 0
    y = 0
    z = 0
    count = len(x_list)
    print("count:",count)
    for xi,yi,zi in zip(x_list,y_list,z_list):
        x += xi/count
        y += yi/count
        z += zi/count
        
    height = box.data.scale.y
    if type(x)==float and type(y)==float and type(z)==float:
        print(random.randint(0,10))
        box.update_attributes(position=Position(-x,y,-z))
        my_text.update_attributes(position=Position(-x,y + 0.5,-z))

        scene.update_object(box)
        scene.update_object(my_text)

# start tasks
scene.run_tasks()

