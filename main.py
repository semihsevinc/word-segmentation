from PIL import Image
import cv2
import numpy as np
import os
import page
import words
from utils import *


folder_path = r'segmented'
#using listdir() method to list the files of the folder
test = os.listdir(folder_path)
#taking a loop to remove all the images
#using ".png" extension to remove only png images
for images in test:
    if images.endswith(".png"):
        os.remove(os.path.join(folder_path, images))

    ##### Shadow removing from image #####

image = cv2.cvtColor(cv2.imread("image/my_htr.png"), cv2.COLOR_BGR2RGB)
rgb_planes = cv2.split(image)
result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 7)
    diff_img = 255 - (cv2.absdiff(plane, bg_img, cv2.CV_32S))
    norm_img = cv2.normalize(diff_img, diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC4)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
result = cv2.merge(result_planes)
shadows_out_image = cv2.merge(result_norm_planes)

    ##### shadow removed from image #####
thresh = 233
shadows_out_image = cv2.threshold(shadows_out_image, thresh, 255, cv2.THRESH_BINARY)[1]
# shadows out image saved JUST IN CASE #
cv2.imwrite('image/shadows_out_image.png', shadows_out_image)
cv2.imwrite('image/diff_img.png', result)
cv2.imwrite('image/image.png', image)


# Crop image and get limiting lines of boxes
crop = page.detection(shadows_out_image)
boxes = words.detection(crop)
lines = words.sort_words(boxes)

i = 0
for line in lines:
    text = crop.copy()
    for (x1, y1, x2, y2) in line:

        save = Image.fromarray(text[y1:y2, x1:x2])
        save.save("segmented/" + str(i + 100) + ".png")
        i += 1
