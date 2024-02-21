# """
# Get bounding boxes and the text value from the images
# """


# import cv2
# import pytesseract

# #pytesseract.pytesseract.tesseract_cmd=r"C:\ProgramFiles\Tesseract-OCR\tesseract.exe"
# # Reading image
# #img = cv2.imread("/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/3_d3.jpg")
# #img  = cv2.imread("/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/train/images/7_d3.jpg")
# img = cv2.imread("/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/others/cropped.jpg")


# # Convert to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(img, (3,3), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# invert = 255 - opening

# # Show the Output
# # cv2.imshow("Output", img)
# # cv2.waitKey(0)
# cv2.imshow('thresh', thresh)
# cv2.imshow('opening', opening)
# cv2.imshow('invert', invert)

# # Detect texts from image
# #texts = pytesseract.image_to_string(img)
# data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
# print(data)

# #print(texts)

# # Return each detected character and their bounding boxes. 
# boxes = pytesseract.image_to_boxes(img)
# print(boxes)
