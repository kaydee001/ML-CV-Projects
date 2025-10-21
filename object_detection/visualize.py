import cv2
import os

def read_yolo_annotation(label_path):
    boxes = []
    file = open(label_path, 'r')
    lines = file.readlines()
    for line in lines:
        numbers = line.split()
        box = [float(n) for n in numbers]    
        boxes.append(box)
    
    file.close()
    return boxes

def draw_boxes(image, boxes, img_width, img_height):
    colors = {0 : (0, 255, 0), 1 : (0, 0, 255)}
    for box in boxes:
        class_id = int(box[0])
        x_center = box[1]
        y_center = box[2]
        width = box[3]
        height = box[4]

        x_center_pixels = x_center * img_width
        y_center_pixels = y_center * img_height
        width_pixels = width * img_width
        height_pixels = height * img_height

        x_min = int(x_center_pixels - width_pixels/2)
        y_min = int(y_center_pixels - height_pixels/2)
        x_max = int(x_center_pixels + width_pixels/2)
        y_max = int(y_center_pixels + height_pixels/2)

        color = colors[class_id]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)

    return image

if __name__ == "__main__":
    img_path = "data/train/images/helmet_jacket_00039.jpg"
    label_path = "data/train/labels/helmet_jacket_00039.txt"

    image = cv2.imread(img_path)
    img_height, img_width, channels = image.shape

    boxes = read_yolo_annotation(label_path)

    fd_mig = draw_boxes(image, boxes, img_width, img_height)
    filename = 'savedimg.jpg'

    cv2.imwrite(filename, fd_mig)
    