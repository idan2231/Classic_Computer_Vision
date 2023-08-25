import numpy as np
import cv2

# Capturing video through webcam
cap = cv2.VideoCapture("dance1.mp4")




# Start a while loop
while True:
    # Reading the video from the
    # webcam in image frames
    _, imageFrame = cap.read()

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)


    # Set range for yellow and white color
    # define mask
    yellow_lower = np.array([14, 145, 22], np.uint8)
    yellow_upper = np.array([47, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    white_lower = np.array([77, 0, 100], np.uint8)
    white_upper = np.array([115, 120, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)



    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    #res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask= yellow_mask)
    # For white color
    white_mask = cv2.dilate(white_mask, kernal)
    #res_white = cv2.bitwise_and(imageFrame, imageFrame, mask= white_mask)



    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 500) :
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 255, 255), 2)
            cv2.putText(imageFrame, "Yellow", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 255))

    # Creating contour to track white color
    contours, hierarchy = cv2.findContours(white_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 800) :
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(255, 255, 255), 2)
            cv2.putText(imageFrame, "White", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255, 255, 255))


    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

#######################################################################
# def empty(a):
#     pass
#
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor = np.hstack(imgArray)
#         ver = hor
#     return ver
#
# cap = cv2.VideoCapture("dance1.mp4")
# cap.set(3, 640)
# cap.set(4, 480)
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 55, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 163, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 184, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
#
# while True:
#     _, img = cap.read()
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower = np.array( [h_min,s_min,v_min])
#     upper = np.array( [h_max,s_max,v_max])
#     mask = cv2.inRange(imgHSV, lower, upper)
#     imgResult = cv2.bitwise_and(img, img, mask=mask)
#
#     cv2.imshow("Original",img)
#     cv2.imshow("HSV",imgHSV)
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Result", imgResult)
#
#     imgStack = stackImages(0.6, ([img,imgHSV],[mask,imgResult]))
#     cv2.imshow("Stacked Images", imgStack)
#     cv2.waitKey(1)