import redis
import cv2
import numpy as np
import wget
import shutil


def download_image(url):
    filename = wget.download(url)
    move_to = 'images/' + filename
    shutil.move(filename, move_to)
    return move_to


def detect(url):
    img_path = download_image(url)

    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = net.getUnconnectedOutLayersNames()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))  # size=(how many colors, for blue-green-red)

    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1.5, fy=1.5)
    height, width, channels = img.shape

    # Detect image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Show info on screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)    # Center

                # Rectangle coordinates
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)    # Rectangles

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)   # Get indices of meaningful boxes
    font = cv2.FONT_HERSHEY_PLAIN

    obj_in_image = []
    for i in range(len(boxes)):
        if i in indices:

            x, y, w, h = boxes[i]
            label = str(classes[int(class_ids[i])])
            obj_in_image.append(label)
            conf = round(confidences[i], 2)
            text = label + ' ' + str(conf)
            color = colors[i]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, text, (x, y + 15), font, 1, color, 2)
    
    
    client = redis.Redis()
    client.flushall()

    ### LIST ###
    for obj in obj_in_image:
        client.lpush('obj-list', str(obj))
    print(client.lrange('obj-list', 0, -1))

   ### HASH ### 
    for obj in obj_in_image:
        if client.hexists('obj-hash', obj): 
            client.hincrby('obj-hash', obj, 1)
        else:
            client.hset('obj-hash', obj, 1)
    print(client.hgetall('obj-hash'))


    #cv2.imshow("Image", img)     # Uncomment these to display.
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return obj_in_image


detect('http://images.clipartpanda.com/rock-clipart-alpine-landscape-rock-rubble-01b-al1.png')