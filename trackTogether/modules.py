def dataconcat(bboxes, depth, color, track):
    """
    function:
        generate the input data of a clustering model
    input: 
    - bboxes: 1*(center, x, y)
    - depth: x*y*(x,y,z)
    - color: x*y*(r,g,b)
    (points(x*y), num_channels, 1, 1, 1)
    - track: 1* (center, x, y,tracked_id,cla,conf)

    output:
    - (points(x*y), num_channels, xyz, center, x, y, cla, tracked_id)
    """



