def iou_3d(box1, box2):
    # Unpack the coordinates
    _, _, _, x1_max, x1_min, y1_max, y1_min, z1_max, z1_min = box1
    _, _, _, x2_max, x2_min, y2_max, y2_min, z2_max, z2_min = box2

    # Calculate the coordinates of the intersection box
    x_max = min(x1_max, x2_max)
    x_min = max(x1_min, x2_min)
    y_max = min(y1_max, y2_max)
    y_min = max(y1_min, y2_min)
    z_max = min(z1_max, z2_max)
    z_min = max(z1_min, z2_min)

    # Check if there is an intersection
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return 0.0

    # Calculate volumes
    intersection_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    box1_volume = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    box2_volume = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)

    # Calculate union
    union_volume = box1_volume + box2_volume - intersection_volume

    # Calculate IOU
    return intersection_volume / union_volume

def process_boxes(boxes):
    n = len(boxes)
    valid_boxes = []

    # First, sort boxes by confidence for easier processing
    boxes.sort(key=lambda x: x[2], reverse=True)

    # Compare each box with every other box
    for i in range(n):
        is_fake = False
        for j in range(n):
            if i != j:
                iou = iou_3d(boxes[i], boxes[j])

                # Same origin, same class, IOU > 0.4
                if boxes[i][0] == boxes[j][0] and boxes[i][1] == boxes[j][1] and iou > 0.4:
                    is_fake = True
                    break

                # Different origin, same class, IOU > 0.2
                if boxes[i][0] != boxes[j][0] and boxes[i][1] == boxes[j][1] and iou > 0.2:
                    is_fake = True
                    # Create an average box
                    average_box = [
                        'average',  # new origin
                        boxes[i][1],  # class
                        max(boxes[i][2], boxes[j][2]),  # max confidence
                        (boxes[i][3] + boxes[j][3]) / 2,  # average xmax
                        (boxes[i][4] + boxes[j][4]) / 2,  # average xmin
                        (boxes[i][5] + boxes[j][5]) / 2,  # average ymax
                        (boxes[i][6] + boxes[j][6]) / 2,  # average ymin
                        (boxes[i][7] + boxes[j][7]) / 2,  # average zmax
                        (boxes[i][8] + boxes[j][8]) / 2   # average zmin
                    ]
                    valid_boxes.append(average_box)
                    break

        if not is_fake:
            valid_boxes.append(boxes[i])

    return valid_boxes