"""
Count based

How many resisters are on the left of the ammeter ?
How many gates are connected to the and gates directly ?

"""

import cv2 
import pandas as pd 
import os 
import ast 
import cv2
from utils.preprocessing import xml_processor,randomize_qs
import utils.templates as templates
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys 
import math
from shapely.geometry import Polygon, LineString


np.set_printoptions(sys.maxsize)



#IMAGE_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/d1_254_png.rf.0b2e9377b97914d428d85e23cbf099b1.jpg"
#XML_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/xml/d1_254_png.rf.0b2e9377b97914d428d85e23cbf099b1.xml"

# IMAGE_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/d1_autockt_-66_png.rf.a30811b86284a145cd9ed227d824e5c1.jpg"
# XML_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/xml/d1_autockt_-66_png.rf.a30811b86284a145cd9ed227d824e5c1.xml"

# GATES
IMAGE_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/d4_210_png.rf.98ef2db47026408a69c3716e4b7d83e0.jpg"
XML_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/xml/d4_210_png.rf.98ef2db47026408a69c3716e4b7d83e0.xml"

#IMAGE_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/d5_IMG_20220810_193854_jpg.rf.a2d16599be5d3788ddf63b7c9ceabb57.jpg"
#XML_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/xml/d5_IMG_20220810_193854_jpg.rf.a2d16599be5d3788ddf63b7c9ceabb57.xml"


DATA_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/"
OUTPUT_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/questions/position"


if __name__ == "__main__":

    # Add labels to gates 
    image = cv2.imread(IMAGE_PATH)
    symbol_df = xml_processor(XML_PATH)
    print(symbol_df)

    # Line detection 
    image = mpimg.imread(IMAGE_PATH)

    # Convert the image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, threshold1 =100, threshold2=200)#,apertureSize=3)

     ###Read gray image
    # result = image.copy()
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    

    # contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, 0, (0, 230, 255), 6)
    # cv2.imwrite('image1.jpg', image)
    


    # Zig zag lines
    # Load your image here
    #image = cv2.imread('your_image.jpg')
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect objects (contours) in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    # Step 2: Calculate distance between objects
    def calculate_distance(contour1, contour2):
        moments1 = cv2.moments(contour1)
        moments2 = cv2.moments(contour2)
        print(moments1)
        print(moments2)
        cx1 = int(moments1['m10'] / moments1['m00'])
        cy1 = int(moments1['m01'] / moments1['m00'])
        cx2 = int(moments2['m10'] / moments2['m00'])
        cy2 = int(moments2['m01'] / moments2['m00'])
        return np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

    # Step 3: Define threshold distance
    threshold_distance = 1  # You can adjust this value as per your requirement

    # Step 4: Check for connected zig-zag objects and draw lines
    def draw_zigzag_line(start_point, end_point):
        cv2.line(image, tuple(start_point), tuple(end_point), (0, 255, 0), 2)

    for i in range(len(contours)):
        for j in range(i+1, len(contours)):
            distance = calculate_distance(contours[i], contours[j])
            if distance < threshold_distance:
                # Draw a zig-zag line between the two connected objects
                draw_zigzag_line(contours[i][0][0], contours[j][0][0])
                draw_zigzag_line(contours[j][0][0], contours[i][0][0])

    # Display the result
    cv2.imwrite('image1.jpg', image)
