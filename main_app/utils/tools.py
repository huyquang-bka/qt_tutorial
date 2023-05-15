import numpy as np
import cv2
import base64


def most_frequent(ls):
    return max(set(ls), key=ls.count)

def is_prime_number(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def recommend_row_col(num):
    if num == 1:
        return 1, 1
    if num == 2:
        return 1, 2
    recommend_dict = {}
    if is_prime_number(num):
        num += 1
    for i in range(1, num):
        if num % i == 0:
            recommend_dict[abs(i - num // i)] = [i, int(num // i)]
    recommend = min(recommend_dict.keys())
    return min(recommend_dict[recommend]), max(recommend_dict[recommend])

def concate_image(images):
    bbox_locate_list = []
    bbox_dict = {"x": "", "y": "", "width": "", "height": ""}
    max_width = sorted(images, key=lambda x: x.shape[1], reverse=True)[
        0].shape[1]
    max_height = sorted(images, key=lambda x: x.shape[0], reverse=True)[
        0].shape[0]
    row, col = recommend_row_col(len(images))
    black_image = np.zeros((max_height * row, max_width * col, 3), np.uint8)
    if row == 1 and col == 1:
        return images[0]
    count = 0
    for i in range(row):
        for j in range(col):
            if count < len(images):
                image = images[count]
                black_image[i * max_height:i * max_height + image.shape[0],
                            j * max_width:j * max_width + image.shape[1]] = image
                bbox_dict["x"] = j * max_width
                bbox_dict["y"] = i * max_height
                bbox_dict["width"] = image.shape[1]
                bbox_dict["height"] = image.shape[0]
                bbox_locate_list.append(bbox_dict.copy())
                count += 1
    return [black_image, bbox_locate_list]


def image_to_base64(image):
    img_to_byte = cv2.imencode('.jpg', image)[1].tobytes()
    byte_to_base64 = base64.b64encode(img_to_byte)
    return byte_to_base64.decode('ascii')  # chuyển về string


def base64_to_image(base64_string):
    base64_to_byte = base64.b64decode(base64_string)
    byte_to_image = np.frombuffer(base64_to_byte, np.uint8)
    image = cv2.imdecode(byte_to_image, cv2.IMREAD_COLOR)
    return image
