# import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time
import sys
import cvzone
from ColorModuleExtended import ColorFinder
import math
from decimal import *
import requests
from configparser import ConfigParser
import ast
import os
import shutil
# For OCR Test
import pytesseract
# MachineLearning Cam
import gxipy as gx
from PIL import Image
# Add multiprocessing for frame aquire
from multiprocessing import Process, Queue, Pipe

# Global switches

debug = False
profiling = False
step = ""
stepId = 0
profiling_data = []
calcSpin = False

# Global Objects

# Configparser 

parser = ConfigParser()
CFG_FILE = 'config.ini'

parser.read(CFG_FILE)

startcoord=[]

coord=[]


if parser.has_option('putting', 'maincamtype'):
    maincamtype=int(parser.get('putting', 'maincamtype'))
else:
    maincamtype=0

vs = cv2.VideoCapture




# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#dilation
def dilate(image):
    kernel = np.ones((7,7),np.uint8)
    return cv2.dilate(image, kernel, iterations = 2)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


def decode(myframe):
    left = np.zeros((400,632,3), np.uint8)
    right = np.zeros((400,632,3), np.uint8)
    
    for i in range(400):
        left[i] = myframe[i, 32: 640 + 24] 
        right[i] = myframe[i, 640 + 24: 640 + 24 + 632] 
    
    return (left, right)

def setFPS(value):
    print(value)
    vs.set(cv2.CAP_PROP_FPS,value)
    pass 

def setXStart(value):
    global startcoord
    global coord
    global parser
    print(value)
    startcoord[0][0]=value
    startcoord[2][0]=value

    global sx1
    sx1=int(value)    
    parser.set('putting', 'startx1', str(sx1))
    parser.write(open(CFG_FILE, "w"))
    pass

def setXEnd(value):
    global startcoord
    global coord
    global parser
    print(value)
    startcoord[1][0]=value
    startcoord[3][0]=value 

    global x1
    global x2
    global sx2
     
    # Detection Gateway
    x1=int(value+10)
    x2=int(x1+10)

    #coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    coord[0][0]=x1
    coord[2][0]=x1
    coord[1][0]=x2
    coord[3][0]=x2

    sx2=int(value)    
    parser.set('putting', 'startx2', str(sx2))
    parser.write(open(CFG_FILE, "w"))
    pass  

def setYStart(value):
    global startcoord
    global coord
    global parser
    print(value)
    startcoord[0][1]=value
    startcoord[1][1]=value

    global y1

    #coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    coord[0][1]=value   
    coord[1][1]=value

    y1=int(value)    
    parser.set('putting', 'y1', str(y1))
    parser.write(open(CFG_FILE, "w"))     
    pass


def setYEnd(value):
    global startcoord
    global coord
    global parser
    print(value)
    startcoord[2][1]=value
    startcoord[3][1]=value 

    global y2

    #coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]
    coord[2][1]=value   
    coord[3][1]=value

    y2=int(value)    
    parser.set('putting', 'y2', str(y2))
    parser.write(open(CFG_FILE, "w"))     
    pass 

def setBallRadius(value):
    global startcoord
    global coord
    global parser
    print(value)    
    global ballradius
    ballradius = int(value)
    parser.set('putting', 'radius', str(ballradius))
    parser.write(open(CFG_FILE, "w"))
    pass

def setFlip(value):
    global startcoord
    global coord
    global parser
    print(value)    
    global flipImage
    flipImage = int(value)
    parser.set('putting', 'flip', str(flipImage))
    parser.write(open(CFG_FILE, "w"))
    pass

def setFlipView(value):
    global startcoord
    global coord
    global parser
    print(value)    
    global flipView
    flipView = int(value)
    parser.set('putting', 'flipView', str(flipView))
    parser.write(open(CFG_FILE, "w"))
    pass

def setMjpeg(value):
    global startcoord
    global coord
    global parser
    print(value)    
    global mjpegenabled
    global message
    if mjpegenabled != int(value):
        vs.release()
        message = "Video Codec changed - Please restart the putting app"
    mjpegenabled = int(value)
    parser.set('putting', 'mjpeg', str(mjpegenabled))
    parser.write(open(CFG_FILE, "w"))
    pass

def setOverwriteFPS(value):
    global startcoord
    global coord
    global parser
    print(value)    
    global overwriteFPS
    global message
    if overwriteFPS != int(value):
        vs.release()
        message = "Overwrite of FPS changed - Please restart the putting app"
    overwriteFPS = int(value)
    parser.set('putting', 'fps', str(overwriteFPS))
    parser.write(open(CFG_FILE, "w"))
    pass

def setDarkness(value):
    global startcoord
    global coord
    global parser
    print(value)    
    global darkness
    darkness = int(value)
    parser.set('putting', 'darkness', str(darkness))
    parser.write(open(CFG_FILE, "w"))
    pass

def GetAngle (p1, p2):
    global startcoord
    global coord
    global parser
    global flipImage
    global videofile
    x1, y1 = p1
    x2, y2 = p2
    dX = x2 - x1
    dY = y2 - y1
    rads = math.atan2 (-dY, dX)

    if flipImage == 1 and videofile == False:    	
        rads = rads*-1
    return math.degrees (rads)

def rgb2yuv(rgb):
    m = np.array([
        [0.29900, -0.147108,  0.614777],
        [0.58700, -0.288804, -0.514799],
        [0.11400,  0.435912, -0.099978]
    ])
    yuv = np.dot(rgb, m)
    yuv[:,:,1:] += 0.5
    return yuv

def yuv2rgb(yuv):
    m = np.array([
        [1.000,  1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])
    yuv[:, :, 1:] -= 0.5
    rgb = np.dot(yuv, m)
    return rgb

# combine lines which are close to each other from cv2 lines detection into one bigger line
# check that only lines are combined which are close to each other
def merge_and_group_lines(lines):
    # Initialize an empty array of lists to store merged lines
    merged_lines = []
    # Create an empty list to store temporary line groups
    temporary_groups = []

    # Iterate over the lines
    for line in lines:
        # Calculate the orientation of the line
    
        # print("line[0]: "+str(line[0][0]))
        # print("line[1]: "+str(line[0][1]))
        # print("line[2]: "+str(line[0][2]))
        # print("line[3]: "+str(line[0][3]))
        orientation = np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0])
        # round orientation to 1 decimal place
        orientation = round(orientation, 1)
        
        # Check if the line's orientation matches any existing group
        found_match = False
        for group in temporary_groups:
            # Check if the line's orientation falls within a certain tolerance of the group's mean orientation
            mean_orientation = np.mean([line[0][2] for line in group])
            tolerance = np.pi / 12  # A tolerance of 15 degrees

            if abs(orientation - mean_orientation) < tolerance:
                group.append(line)
                found_match = True
                break

        # If there's no match, create a new group
        if not found_match:
            temporary_groups.append([line])

    # Merge the temporary groups into the final list of merged lines, add an enumerator
    for i, group in enumerate(temporary_groups):
        merged_lines.append(np.vstack(group))

    return merged_lines

# crop detailframe to x,y and radius and show contours
def showCircleContours(x,y,radius,detailframe,frame):

    global debug

    scaledx = int(width/(frame.shape[1]/x))
    scaledy = int(height/(frame.shape[0]/y))
    scaledradius = int(width/(frame.shape[1]/radius))
    addRadius = (int(scaledradius/4)*-1)
    scalingfactor = (scaledradius+addRadius)/radius
    # Scalingfactor for line detection zoom
    scalingfactor2 = 10
    
    #cv2.imshow("Original Frame no rezise", detailframe)
    zoomframeorigin = detailframe[scaledy-scaledradius-addRadius:scaledy+scaledradius+addRadius, scaledx-scaledradius-addRadius:scaledx+scaledradius+addRadius]

    spin = True
    if spin == False:
        return zoomframeorigin, 0, (50/scalingfactor,50/scalingfactor,50/scalingfactor,50/scalingfactor)
    
    
    actualwidth = int(scaledradius+addRadius)*2
    actualheight = int(scaledradius+addRadius)*2
    actualx = int(actualwidth/2)
    actualy = int(actualheight/2)

    zoomframe = zoomframeorigin.copy()
    if debug == True:
        cv2.imshow("Detail Frame", imutils.resize(zoomframeorigin, width=640, height=360))

    # Convert to gray
    grayzoomframe = cv2.cvtColor(zoomframe, cv2.COLOR_BGR2GRAY)
    # remove glare
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayzoomframe = clahe.apply(grayzoomframe)

    grayzoomframe = imutils.resize(grayzoomframe, width=(actualwidth*scalingfactor2), height=(actualheight*scalingfactor2))


    # Gabor Filter Creation
     
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)

    for kern in filters:  # Loop through the kernels in our GaborFilter
        gabor_img = cv2.filter2D(zoomframe, -1, kern)

    #Show Gabor Image
    if debug == True:
        cv2.imshow("Detail Frame Gabor", imutils.resize(gabor_img, width=640, height=360))

    gabor_img = cv2.cvtColor(gabor_img, cv2.COLOR_BGR2GRAY)
    
    canny_img = canny(gabor_img)

    #Show canny Image
    if debug == True:
        cv2.imshow("Detail Frame canny", imutils.resize(canny_img, width=640, height=360))

    # manipulate image to enhance texture    
    dilated_img = dilate(grayzoomframe.copy())
    # Show dilated_img Image
    if debug == True:
        cv2.imshow("Detail Frame Dialated", imutils.resize(dilated_img, width=640, height=360))
    bg_img = cv2.medianBlur(dilated_img, 21)
    # Show bg_img Image
    if debug == True:
        cv2.imshow("Detail Blurred Image", imutils.resize(bg_img, width=640, height=360))
    diff_img = 255 - cv2.absdiff(grayzoomframe.copy(), bg_img)
    
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(thr_img, 230, 0, cv2.THRESH_TRUNC)    
    

    # # Remove the edge from the original ball
    # cv2.circle(thr_img, (int(actualx), int(actualy)), int(scaledradius), (255, 255, 255), 2)
    # cv2.circle(thr_img, (int(actualx), int(actualy)), int(scaledradius+2), (255, 255, 255), 2)
    # cv2.circle(thr_img, (int(actualx), int(actualy)), int(scaledradius+4), (255, 255, 255), 2)
    # cv2.circle(thr_img, (int(actualx), int(actualy)), int(scaledradius+6), (255, 255, 255), 2)
    
    # Show Threshold Image
    if debug == True:
        cv2.imshow("Detail Frame threshold", imutils.resize(thr_img, width=640, height=360))

    # convert zoomframe to show only black color
    invertedimage = cv2.bitwise_not(thr_img)
    # invertedimage = canny_img.copy()

    # show any light grey areas in mat invertedimage as white and every dark grey as black
    seperator = 30
    invertedimage[invertedimage <= seperator] = 0
    invertedimage[invertedimage > seperator] = 255
    if debug == True:
        cv2.imshow("Detail Frame inverted", imutils.resize(invertedimage, width=640, height=360))

    # get contours in zoomframe
    cnts = cv2.findContours(invertedimage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    # find lines in invertedimage
    myLineLength = int(scaledradius/3)  

    
    lines = cv2.HoughLinesP(invertedimage, 1, np.pi / 180, 100, minLineLength=myLineLength, maxLineGap=10)

    # Overwrites with line from canny edge detection
    # lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 100, minLineLength=1, maxLineGap=100)

    if debug == True:
        line_img = zoomframeorigin.copy()
        if lines is not None:
            for linegroup in lines:
                x1, y1, x2, y2 = linegroup[0]
                x1 = int(x1/scalingfactor2)
                y1 = int(y1/scalingfactor2)
                x2 = int(x2/scalingfactor2)
                y2 = int(y2/scalingfactor2)
                line_img = cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.imshow("Detail Frame lines", imutils.resize(line_img, width=640, height=360))
    # eliminate the line if it is on the radius
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 = int(x1/scalingfactor2)
            y1 = int(y1/scalingfactor2)
            x2 = int(x2/scalingfactor2)
            y2 = int(y2/scalingfactor2)
            #calculate the distance between the line represented by x1, y1, x2, y2 and the center of the circle represented by scaledx and scaledy

            distanceline = math.sqrt((y2-y1)** 2 + (x2-x1) ** 2)
            distancetox1y1 = math.sqrt((actualy-y1)** 2 + (actualx-x1) ** 2)
            distancetox2y2 = math.sqrt((actualy-y2)** 2 + (actualx-x2) ** 2)
            
            if distancetox1y1 < scaledradius + 10 and distancetox1y1 > scaledradius - 10:
                if distancetox2y2 < scaledradius + 10 and distancetox2y2 > scaledradius - 10:
                    #remove the line from the list of lines
                    lines = np.delete(lines, (np.where(lines == line)[0][0]), axis=0)
                    break

    
    

    # draw lines in zoomframe
    addedshapes = zoomframeorigin.copy()

    # combine lines which are next to each other
    if lines is not None:
        lines = merge_and_group_lines(lines)
 
        if lines is not None:
            for linegroup in lines:
                x1, y1, x2, y2 = linegroup[0]
                x1 = int(x1/scalingfactor2)
                y1 = int(y1/scalingfactor2)
                x2 = int(x2/scalingfactor2)
                y2 = int(y2/scalingfactor2)
                addedshapes = cv2.line(addedshapes, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cnts = imutils.grab_contours(cnts)
    # draw contours in zoomframe
    if cnts is not None:                
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            x = int(x/scalingfactor2)
            y = int(y/scalingfactor2)
            w = int(w/scalingfactor2)
            h = int(h/scalingfactor2)
            # Drawing a rectangle on copied image
            addedshapes = cv2.rectangle(addedshapes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # seperate the combinedlines into the cnts rectangles and draw them in different color per rectangle
    cntrectangles = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        x = int(x/scalingfactor2)
        y = int(y/scalingfactor2)
        w = int(w/scalingfactor2)
        h = int(h/scalingfactor2)
        cntrectangles.append((x, y, w, h))

    # TODO: Add weighting system to choose best found line shapes in combinedlines

    
    resultlines = []
    if lines is not None:
        for linegroup in lines:
            x1, y1, x2, y2 = linegroup[0]
            x1 = int(x1/scalingfactor2)
            y1 = int(y1/scalingfactor2)
            x2 = int(x2/scalingfactor2)
            y2 = int(y2/scalingfactor2)
            for cntrect in cntrectangles:
                x, y, w, h = cntrect
                
                if x1 >= x and x1 <= x+w and y1 >= y and y1 <= y+h and x2 >= x and x2 <= x+w and y2 >= y and y2 <= y+h and x2-x1 > (w/3):
                    addedshapes = cv2.line(addedshapes, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    resultlines.append((x1, y1, x2, y2))

    # get the mean values out of resultlines
    angle = 0
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    if len(resultlines) > 0:
        sumx1 = 0
        sumx2 = 0
        sumy1 = 0
        sumy2 = 0
        for i, resultline in enumerate(resultlines):
            x1, y1, x2, y2 = resultline
            sumx1 = sumx1 + x1
            sumx2 = sumx2 + x2
            sumy1 = sumy1 + y1
            sumy2 = sumy2 + y2
        x1 = int(sumx1/len(resultlines))
        x2 = int(sumx2/len(resultlines))
        y1 = int(sumy1/len(resultlines))
        y2 = int(sumy2/len(resultlines))
        angle = (GetAngle((x1,y1),(x2,y2))*-1)
        # print angle on mat
        drawtexty = y1
        cv2.putText(zoomframeorigin, str(angle), (x1, drawtexty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print("Detected Shape Angle: "+str(angle))
        cv2.line(zoomframeorigin, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.line(zoomframeorigin, (x1, y1), (x2, y1), (0, 255, 0), 1)
    if debug == True:
        cv2.imshow("Added Shapes", imutils.resize(addedshapes, width=640, height=360))

    
    # return image with the markings
    return zoomframeorigin, angle, (x1/scalingfactor,x2/scalingfactor,y1/scalingfactor,y2/scalingfactor)

def showOutput(child_output_conn,outputqueue):
    global a_key_pressed
    global d_key_pressed
    global vs
    global parser
    global maincamtype
    global args
    global debug
    global myColorFinder
    global mjpegenabled

    while True:        
        outputframe = child_output_conn.recv()
        cv2.imshow("Putting View: Press q to exit / a for adv. settings", outputframe)
        cv2.waitKey(1)



def getFrames(child_frame_conn, framequeue, maincamtype):
    
    if maincamtype == 0 or maincamtype == 1:
        # initialize webcam vs object
        print("Initializing Webcam")
        while True:
            ret, frame = vs.read()
            if frame is None:
                print("Webcam image failed.")
                continue
            else:
                #framequeue.put(frame)
                
                #frame = imutils.resize(frame, width=640, height=360)
                child_frame_conn.send(frame)
        
    if maincamtype == 2:
        webcamindex = 1 # Number of Webcamindex

        Width_set = 600 # Set the resolution width
        Height_set = 300 # Set the resolution height
        framerate_set = 300 # Set frame rate
        offset_x = 550
        offset_y = 500

        #Position offset



        gain = 24
        exposure = 5000

        counter = 0
        last_frame_id = 0
        # time.sleep(3)
        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            print("Device not found - Check Camera Connection")
            return

        # open the first device
        device = device_manager.open_device_by_index(webcamindex)

        #Set width and height
        device.Width.set(Width_set)
        device.Height.set(Height_set)

        #Set offset
        device.OffsetX.set(offset_x)
        device.OffsetY.set(offset_y)

        # Grayscale Mode ON/OFF
        device.SaturationMode.set(gx.GxSwitchEntry.OFF)
        device.Saturation.set(0)
        
        #Set up continuous collection
        device.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)

        #Set the frame rate
        device.AcquisitionFrameRate.set(framerate_set)

        # exit when the camera is a mono camera
        if device.PixelColorFilter.is_implemented() == False:
            print("This sample does not support mono camera.")
            device.close_device()
            return

        # set continuous acquisition
        device.TriggerMode.set(gx.GxSwitchEntry.OFF)
        
        # cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

        # set exposure
        device.ExposureTime.set(exposure)

        # set gain
        device.Gain.set(gain)

        # set white balance
        device.BalanceWhiteAuto.set(gx.GxAutoEntry.ONCE)

        device.LightSourcePreset.set(gx.GxLightSourcePresetEntry.DAYLIGHT_6500K)
        
        # start data acquisition
        device.stream_on()

        # acquisition image: num is the image number
        
        while True:      

            if isinstance(device, gx.U3VDevice):
                
                    
                    #time.sleep(0.1)
                    # send software trigger command
                    # device.TriggerSoftware.send_command()

                    # get raw image
                    raw_image = device.data_stream[0].get_image()
                    if raw_image is None:
                        print("Getting image failed.")
                        continue
                    else:
                        counter = counter + 1

                    # print height, width, and frame ID of the acquisition image
                    # print("Frame ID: %d   Height: %d   Width: %d    Current FPS: %d     Current time: %d"
                    #    % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width(), device.CurrentAcquisitionFrameRate.get(), time.time()))

                    # get RGB image from raw image
                    rgb_image = raw_image.convert("RGB")
                    if rgb_image is None:
                        print("RGB image failed.")
                        continue

                    # improve image quality
                    #rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

                    # create numpy array with data from raw image
                    numpy_image = rgb_image.get_numpy_array()
                    if numpy_image is None:
                        print("Numpy image failed.")
                        continue
                    else:
                        # framequeue.put(numpy_image)
                        
                        #numpy_image = imutils.resize(numpy_image, width=640, height=360)
                        child_frame_conn.send(numpy_image)
                        last_frame_id = raw_image.get_frame_id()

                    # if (child_frame_conn.recv() == 1):      
        # stop data acquisition
        device.stream_off()

        # close device
        device.close_device()

                    #     break

def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

# Multiprocessing

framequeue = Queue()
parent_frame_conn, child_frame_conn = Pipe()
frameprocess = Process(target=getFrames, args=(child_frame_conn,framequeue,maincamtype,))


outputqueue = Queue()
parent_output_conn, child_output_conn = Pipe()
outputprocess = Process(target=showOutput, args=(child_output_conn,outputqueue,))



def main():

    # Globals

    global debug
    global maincamtype

    # Profiling

    global profiling
    global step
    global stepId

    global width
    global height

    global startcoord
    global coord
    global parser
    global flipImage
    global videofile

    global vs
    global frameprocess
    global framequeue
    global outputprocess
    global outputqueue
    global parent_frame_conn
    global child_frame_conn

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-i", "--img",
                    help="path to the (optional) image file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size - default is 64")
    ap.add_argument("-w", "--camera", type=int, default=0,
                    help="webcam index number - default is 0")
    ap.add_argument("-c", "--ballcolor",
                    help="ball color - default is yellow")
    ap.add_argument("-d", "--debug",
                    help="debug - color finder and wait timer")
    ap.add_argument("-r", "--resize", type=int, default=640,
                    help="window resize in width pixel - default is 640px")
    args = vars(ap.parse_args())

    # if a webcam index is supplied, grab the reference
    if args.get("camera", False):
        webcamindex = args["camera"]
        print("Putting Cam activated at "+str(webcamindex))

    golfballradius = 21.33; # in mm

    actualFPS = 0

    videoStartTime = time.time()

    # initialize variables to store the start and end positions of the ball
    startCircle = (0, 0, 0)
    endCircle = (0, 0, 0)
    startPos = (0,0)
    endPos = (0,0)
    startTime = time.time()
    timeSinceEntered = 0
    replaytimeSinceEntered = 0
    pixelmmratio = 0

    # initialize variable to store start candidates of balls
    startCandidates = []
    startminimum = 30

    # Initialize Entered indicator
    entered = False
    started = False
    left = False

    lastShotStart = (0,0)
    lastShotEnd = (0,0)
    lastShotSpeed = 0
    lastShotHLA = 0 

    speed = 0

    tim1 = 0
    tim2 = 0
    replaytrigger = 0

    # calibration

    colorcount = 0
    calibrationtime = time.time()
    calObjectCount = 0
    calColorObjectCount = []
    calibrationTimeFrame = 30

    # Calibrate Recording Indicator

    record = True

    # Spin
    deltaangle = 0
    shapeangle1 = 0
    shapeangle2 = 0
    shapeangle3 = 0
    spin1 = False
    spin2 = False
    spin3 = False

    # Videofile Indicator

    videofile = False

    # remove duplicate advanced screens for multipla 'a' and 'd' key presses)
    a_key_pressed = False 
    d_key_pressed = False 

    frame = cv2.Mat

    # Startpoint Zone

    ballradius = 0
    darkness = 0
    flipImage = 0
    mjpegenabled = 0
    overwriteFPS = 0

    customhsv = {}

    replaycam=0
    replaycamindex=0
    timeSinceTriggered = 0
    replaycamtype = 0
    replay = False
    noOfStarts = 0
    replayavail = False
    frameskip = 0

    resetinseconds = 0.5

    if parser.has_option('putting', 'startx1'):
        sx1=int(parser.get('putting', 'startx1'))
    else:
        sx1=10
    if parser.has_option('putting', 'startx2'):
        sx2=int(parser.get('putting', 'startx2'))
    else:
        sx2=180
    if parser.has_option('putting', 'y1'):
        y1=int(parser.get('putting', 'y1'))
    else:
        y1=180
    if parser.has_option('putting', 'y2'):
        y2=int(parser.get('putting', 'y2'))
    else:
        y2=450
    if parser.has_option('putting', 'radius'):
        ballradius=int(parser.get('putting', 'radius'))
    else:
        ballradius=0
    if parser.has_option('putting', 'flip'):
        flipImage=int(parser.get('putting', 'flip'))
    else:
        flipImage=0
    if parser.has_option('putting', 'flipview'):
        flipView=int(parser.get('putting', 'flipview'))
    else:
        flipView=0
    if parser.has_option('putting', 'darkness'):
        darkness=int(parser.get('putting', 'darkness'))
    else:
        darkness=0
    if parser.has_option('putting', 'mjpeg'):
        mjpegenabled=int(parser.get('putting', 'mjpeg'))
    else:
        mjpegenabled=0
    if parser.has_option('putting', 'maincamtype'):
        maincamtype=int(parser.get('putting', 'maincamtype'))
    else:
        maincamtype=0
    if parser.has_option('putting', 'fps'):
        overwriteFPS=int(parser.get('putting', 'fps'))
    else:
        overwriteFPS=0
    if parser.has_option('putting', 'height'):
        height=int(parser.get('putting', 'height'))
    else:
        height=360
    if parser.has_option('putting', 'width'):
        width=int(parser.get('putting', 'width'))
    else:
        width=640
    if parser.has_option('putting', 'customhsv'):
        customhsv=ast.literal_eval(parser.get('putting', 'customhsv'))
        print(customhsv)
    else:
        customhsv={}
    if parser.has_option('putting', 'showreplay'):
        showreplay=int(parser.get('putting', 'showreplay'))
    else:
        showreplay=0
    if parser.has_option('putting', 'replaycam'):
        replaycam=int(parser.get('putting', 'replaycam'))
    else:
        replaycam=0
    if parser.has_option('putting', 'replaycamindex'):
        replaycamindex=int(parser.get('putting', 'replaycamindex'))
    else:
        replaycamindex=0
    if parser.has_option('putting', 'replaycamtype'):
        replaycamtype=int(parser.get('putting', 'replaycamtype'))
    else:
        replaycamtype=0

    # Detection Gateway
    x1=sx2+10
    x2=x1+10

    #coord of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    startcoord=[[sx1,y1],[sx2,y1],[sx1,y2],[sx2,y2]]

    #coord of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    coord=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]]

    # Create the color Finder object set to True if you need to Find the color
    debug = False

    pts = deque(maxlen=args["buffer"])
    tims = deque(maxlen=args["buffer"])
    fpsqueue = deque(maxlen=240)
    replay1queue = deque(maxlen=600)
    replay2queue = deque(maxlen=600)

    webcamindex = 0

    message = ""

    ## Check for folder replay1 and replay2 and empty if necessary

    if os.path.exists('replay1'):
        try:
            shutil.rmtree('replay1')
            time.sleep(1)
            os.mkdir('replay1')
        except os.error as e:  # This is the correct syntax
            print(e)
    else:
        os.mkdir('replay1')

    if os.path.exists('replay2'):
        try:
            shutil.rmtree('replay2')
            time.sleep(1)
            os.mkdir('replay2')
        except os.error as e:  # This is the correct syntax
            print(e)
    else:
        os.mkdir('replay2')






    

    # define the lower and upper boundaries of the different ball color options (-c)
    # ball in the HSV color space, then initialize the

    #red                   
    red = {'hmin': 1, 'smin': 208, 'vmin': 0, 'hmax': 50, 'smax': 255, 'vmax': 249} # light
    red2 = {'hmin': 1, 'smin': 240, 'vmin': 61, 'hmax': 50, 'smax': 255, 'vmax': 249} # dark

    #white
    white = {'hmin': 168, 'smin': 218, 'vmin': 118, 'hmax': 179, 'smax': 247, 'vmax': 216} # very light
    white2 = {'hmin': 159, 'smin': 217, 'vmin': 152, 'hmax': 179, 'smax': 255, 'vmax': 255} # light
    white3 = {'hmin': 0, 'smin': 181, 'vmin': 0, 'hmax': 42, 'smax': 255, 'vmax': 255}

    #yellow

    yellow = {'hmin': 0, 'smin': 210, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255} # light
    yellow2 = {'hmin': 0, 'smin': 150, 'vmin': 100, 'hmax': 46, 'smax': 255, 'vmax': 206} # dark

    #green
    green = {'hmin': 0, 'smin': 169, 'vmin': 161, 'hmax': 177, 'smax': 204, 'vmax': 255} # light
    green2 = {'hmin': 0, 'smin': 109, 'vmin': 74, 'hmax': 81, 'smax': 193, 'vmax': 117} # dark

    #orange
    orange = {'hmin': 0, 'smin': 219, 'vmin': 147, 'hmax': 19, 'smax': 255, 'vmax': 255}# light
    orange2 = {'hmin': 3, 'smin': 181, 'vmin': 134, 'hmax': 40, 'smax': 255, 'vmax': 255}# dark
    orange3 = {'hmin': 0, 'smin': 73, 'vmin': 150, 'hmax': 40, 'smax': 255, 'vmax': 255}# test
    orange4 = {'hmin': 3, 'smin': 181, 'vmin': 216, 'hmax': 40, 'smax': 255, 'vmax': 255}# ps3eye

    calibrate = {}

    # for Colorpicker
    # default yellow option
    hsvVals = yellow

    if customhsv == {}:

        if args.get("ballcolor", False):
            if args["ballcolor"] == "white":
                hsvVals = white
            elif args["ballcolor"] == "white2":
                hsvVals = white2
            elif args["ballcolor"] ==  "yellow":
                hsvVals = yellow 
            elif args["ballcolor"] ==  "yellow2":
                hsvVals = yellow2 
            elif args["ballcolor"] ==  "orange":
                hsvVals = orange
            elif args["ballcolor"] ==  "orange2":
                hsvVals = orange2
            elif args["ballcolor"] ==  "orange3":
                hsvVals = orange3
            elif args["ballcolor"] ==  "orange4":
                hsvVals = orange4
            elif args["ballcolor"] ==  "green":
                hsvVals = green 
            elif args["ballcolor"] ==  "green2":
                hsvVals = green2               
            elif args["ballcolor"] ==  "red":
                hsvVals = red             
            elif args["ballcolor"] ==  "red2":
                hsvVals = red2             
            else:
                hsvVals = yellow

            if args["ballcolor"] is not None:
                print("Ballcolor: "+str(args["ballcolor"]))
    else:
        hsvVals = customhsv
        print("Custom HSV Values set in config.ini")

    calibrationcolor = [("white",white),("white2",white2),("yellow",yellow),("yellow2",yellow2),("orange",orange),("orange2",orange2),("orange3",orange3),("orange4",orange4),("green",green),("green2",green2),("red",red),("red2",red2)]

    # Start output as process
    outputprocess.start() 

    # Start Splash Screen

    frame = cv2.imread("error.png")
    origframe2 = cv2.imread("error.png")
    cv2.putText(frame,"Starting Video: Try MJPEG option in advanced settings for faster startup",(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
    outputframe = resizeWithAspectRatio(frame, width=int(args["resize"]))

    parent_output_conn.send(outputframe)

    if args.get("debug", False):
        debug = True
        myColorFinder = ColorFinder(True)
        myColorFinder.setTrackbarValues(hsvVals)
        if int(args["debug"]) == 0:
            resetinseconds = 30
    else:
        myColorFinder = ColorFinder(False)

    

    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        # Check for the webcamtype
        if maincamtype == 0 or maincamtype == 1:
            if mjpegenabled == 0:
                vs = cv2.VideoCapture(webcamindex)
            else:
                vs = cv2.VideoCapture(webcamindex + cv2.CAP_DSHOW)
                # Check if FPS is overwritten in config
                if overwriteFPS != 0:
                    vs.set(cv2.CAP_PROP_FPS, overwriteFPS)
                    print("Overwrite FPS: "+str(vs.get(cv2.CAP_PROP_FPS)))
                if height != 0 and width != 0:
                    vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                mjpeg = cv2.VideoWriter_fourcc('M','J','P','G')
                vs.set(cv2.CAP_PROP_FOURCC, mjpeg)
                if maincamtype == 1:
                        vs.set(cv2.CAP_PROP_FPS, 120)
                        vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1724)
                        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 404)
                        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
                        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)
                if vs.get(cv2.CAP_PROP_BACKEND) == -1:
                    message = "No Camera could be opened at webcamera index "+str(webcamindex)+". If your webcam only supports compressed format MJPEG instead of YUY2 please set MJPEG option to 1"
                else:
                    print("Backend: "+str(vs.get(cv2.CAP_PROP_BACKEND)))
                    print("FourCC: "+str(vs.get(cv2.CAP_PROP_FOURCC)))
                    print("FPS: "+str(vs.get(cv2.CAP_PROP_FPS)))
        if maincamtype == 2:       
            # Start frame aquire as process
            frameprocess.start()


    else:
        vs = cv2.VideoCapture(args["video"])
        videofile = True



    # Get video metadata

    if maincamtype == 0 or maincamtype == 1:
        video_fps = vs.get(cv2.CAP_PROP_FPS)
        height = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vs.get(cv2.CAP_PROP_FRAME_WIDTH)

        if parser.has_option('putting', 'saturation'):
            saturation=float(parser.get('putting', 'saturation'))
        else:
            saturation = vs.get(cv2.CAP_PROP_SATURATION)
        if parser.has_option('putting', 'exposure'):
            exposure=float(parser.get('putting', 'exposure'))
        else:
            exposure = vs.get(cv2.CAP_PROP_EXPOSURE)
        if parser.has_option('putting', 'autowb'):
            autowb=float(parser.get('putting', 'autowb'))
        else:
            autowb = vs.get(cv2.CAP_PROP_AUTO_WB)
        if parser.has_option('putting', 'whiteBalanceBlue'):
            whiteBalanceBlue=float(parser.get('putting', 'whiteBalanceBlue'))
        else:
            whiteBalanceBlue = vs.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
        if parser.has_option('putting', 'whiteBalanceRed'):
            whiteBalanceRed=float(parser.get('putting', 'whiteBalanceRed'))
        else:
            whiteBalanceRed = vs.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
        if parser.has_option('putting', 'brightness'):
            brightness=float(parser.get('putting', 'brightness'))
        else:
            brightness = vs.get(cv2.CAP_PROP_BRIGHTNESS)
        if parser.has_option('putting', 'contrast'):
            contrast=float(parser.get('putting', 'contrast'))
        else:
            contrast = vs.get(cv2.CAP_PROP_CONTRAST)
        if parser.has_option('putting', 'hue'):
            hue=float(parser.get('putting', 'hue'))
        else:
            hue = vs.get(cv2.CAP_PROP_HUE)
        if parser.has_option('putting', 'gain'):
            gain=float(parser.get('putting', 'gain'))
        else:
            gain = vs.get(cv2.CAP_PROP_HUE)
        if parser.has_option('putting', 'monochrome'):
            monochrome=float(parser.get('putting', 'monochrome'))
        else:
            monochrome = vs.get(cv2.CAP_PROP_MONOCHROME)
        if parser.has_option('putting', 'sharpness'):
            sharpness=float(parser.get('putting', 'sharpness'))
        else:
            sharpness = vs.get(cv2.CAP_PROP_SHARPNESS)
        if parser.has_option('putting', 'autoexposure'):
            autoexposure=float(parser.get('putting', 'autoexposure'))
        else:
            autoexposure = vs.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        if parser.has_option('putting', 'gamma'):
            gamma=float(parser.get('putting', 'gamma'))
        else:
            gamma = vs.get(cv2.CAP_PROP_GAMMA)
        if parser.has_option('putting', 'zoom'):
            zoom=float(parser.get('putting', 'zoom'))
        else:
            zoom = vs.get(cv2.CAP_PROP_ZOOM)
            gamma = vs.get(cv2.CAP_PROP_GAMMA)
        if parser.has_option('putting', 'focus'):
            focus=float(parser.get('putting', 'focus'))
        else:
            focus = vs.get(cv2.CAP_PROP_FOCUS)
        if parser.has_option('putting', 'autofocus'):
            autofocus=float(parser.get('putting', 'autofocus'))
        else:
            autofocus = vs.get(cv2.CAP_PROP_AUTOFOCUS)


        vs.set(cv2.CAP_PROP_SATURATION,saturation)
        vs.set(cv2.CAP_PROP_EXPOSURE,exposure)
        vs.set(cv2.CAP_PROP_AUTO_WB,autowb)
        vs.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,whiteBalanceBlue)
        vs.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V,whiteBalanceRed)
        vs.set(cv2.CAP_PROP_BRIGHTNESS,brightness)
        vs.set(cv2.CAP_PROP_CONTRAST,contrast)
        vs.set(cv2.CAP_PROP_HUE,hue)
        vs.set(cv2.CAP_PROP_GAIN,gain)
        vs.set(cv2.CAP_PROP_MONOCHROME,monochrome)
        vs.set(cv2.CAP_PROP_SHARPNESS,sharpness)
        vs.set(cv2.CAP_PROP_AUTO_EXPOSURE,autoexposure)
        vs.set(cv2.CAP_PROP_GAMMA,gamma)
        vs.set(cv2.CAP_PROP_ZOOM,zoom)
        vs.set(cv2.CAP_PROP_FOCUS,focus)
        vs.set(cv2.CAP_PROP_AUTOFOCUS,autofocus)

    if maincamtype == 2:
        width = 600
        height = 300
        video_fps = 300


    print("video_fps: "+str(video_fps))
    print("height: "+str(height))
    print("width: "+str(width))
    if replaycam == 1:
        if replaycamindex == webcamindex:
            print("Replaycamindex must be different to webcam index")
            replaycam = 0
        else:

            print("Replay Cam activated at "+str(replaycamindex))


    # replay is enabled start a 2nd video capture
    if replaycam == 1:
        if mjpegenabled == 0:
            vs2 = cv2.VideoCapture(replaycamindex)
        else:
            vs2 = cv2.VideoCapture(replaycamindex + cv2.CAP_DSHOW)
            # Check if FPS is overwritten in config
            if overwriteFPS != 0:
                vs2.set(cv2.CAP_PROP_FPS, overwriteFPS)
                print("Overwrite FPS: "+str(vs.get(cv2.CAP_PROP_FPS)))
            if height != 0 and width != 0:
                vs2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                vs2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            mjpeg = cv2.VideoWriter_fourcc('M','J','P','G')
            vs2.set(cv2.CAP_PROP_FOURCC, mjpeg)
        if vs2.get(cv2.CAP_PROP_BACKEND) == -1:
            message = "No Camera could be opened at webcamera index "+str(replaycamindex)+". If your webcam only supports compressed format MJPEG instead of YUY2 please set MJPEG option to 1"
        else:
            if replaycamtype == 1:
                #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
                #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)
                vs2.set(cv2.CAP_PROP_FPS, 120)
                vs2.set(cv2.CAP_PROP_FRAME_WIDTH, 1724)
                vs2.set(cv2.CAP_PROP_FRAME_HEIGHT, 404)
            print("Backend: "+str(vs2.get(cv2.CAP_PROP_BACKEND)))
            print("FourCC: "+str(vs2.get(cv2.CAP_PROP_FOURCC)))
            print("FPS: "+str(vs2.get(cv2.CAP_PROP_FPS)))
        replaycamheight = vs2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        replaycamwidth = vs2.get(cv2.CAP_PROP_FRAME_WIDTH)
    else:
        print("Replay Cam not activated")



    if type(video_fps) == float:
        if video_fps == 0.0:
            e = vs.set(cv2.CAP_PROP_FPS, 60)
            new_fps = []
            new_fps.append(0)

        if video_fps > 0.0:
            new_fps = []
            new_fps.append(video_fps)
        video_fps = new_fps


    # we are using x264 codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out2 = cv2.VideoWriter('Calibration.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(width), int(height)))

    # allow the camera or video file to warm up
    time.sleep(0.5)

    previousFrame = cv2.Mat

    while True:
        # Get Frame from Queue
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Get Frame"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1
        # set the frameTime
        frameTime = time.time()
        fpsqueue.append(frameTime)
        
        actualFPS = actualFPS + 1
        videoTimeDiff = 0
        if len(fpsqueue) > 10:
            videoTimeDiff1 = fpsqueue[len(fpsqueue)-1] - fpsqueue[len(fpsqueue)-2]
            videoTimeDiff2 = fpsqueue[len(fpsqueue)-2] - fpsqueue[len(fpsqueue)-3]
            videoTimeDiff3 = fpsqueue[len(fpsqueue)-3] - fpsqueue[len(fpsqueue)-4]
            videoTimeDiff4 = fpsqueue[len(fpsqueue)-4] - fpsqueue[len(fpsqueue)-5]
            videoTimeDiff = (videoTimeDiff1 + videoTimeDiff2 + videoTimeDiff3 + videoTimeDiff4)/4
        if videoTimeDiff != 0:
            fps = 1 / videoTimeDiff
        else:
            fps = 0


        # Args Access
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Args Access"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1
        if args.get("img", False):
            frame = cv2.imread(args["img"])
        else:
            # get webcam frame either from device or from queue
            # while framequeue.empty() == True:
            #     i = 1
            #     #time.sleep(0.2) 
            #     #print("Framequeue empty")
            # Get actual Frame
            if profiling == True:
                step = "Step " + str(stepId) + ":" + "Get actual Frame"
                profiling_data.append((time.perf_counter(),step))
                stepId = stepId + 1
            #frame = framequeue.get()
            if videofile == True:
                ret, frame = vs.read()
            else:
                frame = parent_frame_conn.recv()
            if maincamtype == 1 and ret == True:
                leftframe, rightframe = decode(frame)
                frame = leftframe[0:400,20:632]
                width = 612
                height = 400
            # get replaycam frame
            if replaycam == 1:
                ret2, origframe2 = vs2.read()
                if replaycamtype == 1 and ret2 == True:
                    leftframe2, rightframe2 = decode(origframe2)
                    origframe2 = leftframe2[0:400,20:632]
                    replaycamwidth = 612
                    replaycamheight = 400
            # flip image on y-axis
            if flipImage == 1 and videofile == False:	
                frame = cv2.flip(frame, flipImage)
            
            if (args["ballcolor"] == "calibrate" and maincamtype != 2):
                if record == False:
                    if args.get("debug", False):
                        cv2.waitKey(int(args["debug"]))
                    if frame is None:
                        calColorObjectCount.append((calibrationcolor[colorcount][0],calObjectCount))
                        colorcount += 1
                        calObjectCount = 0
                        if colorcount == len(calibrationcolor):
                            vs.release()
                            vs = cv2.VideoCapture(webcamindex)
                            videofile = False
                            #vs.set(cv2.CAP_PROP_FPS, 60)
                            ret, frame = vs.read()
                            # flip image on y-axis
                            if flipImage == 1 and videofile == False:    	
                                frame = cv2.flip(frame, flipImage)
                            print("Calibration Finished:"+str(calColorObjectCount))
                            cv2.putText(frame,"Calibration Finished:",(150,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                            i = 20
                            texty = 100
                            for calObject in calColorObjectCount:
                                texty = texty+i
                                cv2.putText(frame,str(calObject),(150,texty),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                            texty = texty+i
                            cv2.putText(frame,"Hit any key and choose color with the highest count.",(150,texty),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                            cv2.imshow("Putting View: Press Q to exit / changing Ball Color", frame)
                            cv2.waitKey(0)
                            # Show Results back to Connect App and set directly highest count - maybe also check for false Exit lowest value if 2 colors have equal hits
                            break
                        else:
                            vs.release()                        
                            # grab the calibration video
                            vs = cv2.VideoCapture('Calibration.mp4')
                            videofile = True
                            # grab the current frame
                            ret, frame = vs.read()
                    else:
                        hsvVals = calibrationcolor[colorcount][1]
                        if args.get("debug", False):
                            myColorFinder.setTrackbarValues(hsvVals)
                        cv2.putText(frame,"Calibration Mode:"+str(calibrationcolor[colorcount][0]),(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                else:
                    if (frameTime - calibrationtime) > calibrationTimeFrame:
                        record =  False
                        out2.release()
                        vs.release()
                        # grab the calibration video
                        vs = cv2.VideoCapture('Calibration.mp4')
                        videofile = True
                        # grab the current frame
                        ret, frame = vs.read()
                    cv2.putText(frame,"Calibration Mode:",(200,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255)) 

            # handle the frame from VideoCapture or VideoStream
            # frame = frame[1] if args.get("video", False) else frame

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if frame is None:
                print("no frame")
                frame = cv2.imread("error.png")
                cv2.putText(frame,"Error: "+"No Frame",(20, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                cv2.putText(frame,"Message: "+message,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                cv2.imshow("Putting View: Press q to exit / a for adv. settings", frame)
                cv2.waitKey(0)
                break

        # Orig Frame Copy
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Orig Frame Copy"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1
        # Save original version of frame
        origframe = frame.copy()
        frame = imutils.resize(frame, width=640, height=360)

        # Initial modification of frame
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Initial Frame Mod 1"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        cv2.normalize(frame, frame, 0-darkness, 255-darkness, norm_type=cv2.NORM_MINMAX)
        
        # cropping needed for video files as they are too big
        if args.get("debug", False):   
            # wait for debugging
            cv2.waitKey(int(args["debug"]))
        
        # resize the frame, blur it, and convert it to the HSV
        # color space
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Initial modification of frame
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Initial Frame Mod 2"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        
        # Find the Color Ball
        
        img_color, mask, newHSV = myColorFinder.update(hsv, hsvVals)
        if hsvVals != newHSV:
            print(newHSV)
            parser.set('putting', 'customhsv', str(newHSV)) #['hmin']+newHSV['smin']+newHSV['vmin']+newHSV['hmax']+newHSV['smax']+newHSV['vmax']))
            parser.write(open(CFG_FILE, "w"))
            hsvVals = newHSV
            print("HSV values changed - Custom Color Set to config.ini")

        # Find contours in Mask
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Find Contours"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        mask = mask[y1:y2, sx1:640]

        # Mask now comes from ColorFinder
        #mask = cv2.erode(mask, None, iterations=1)
        #mask = cv2.dilate(mask, None, iterations=5)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # testing with cirlces
        # grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,10) 
        # # loop over the (x, y) coordinates and radius of the circles
        # if (circles and len(circles) >= 1):
        #     for (x, y, r) in circles:
        #         cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


        cnts = imutils.grab_contours(cnts)
        center = None
        
        # Startpoint Zone

        cv2.line(frame, (startcoord[0][0], startcoord[0][1]), (startcoord[1][0], startcoord[1][1]), (0, 210, 255), 2)  # First horizontal line
        cv2.line(frame, (startcoord[0][0], startcoord[0][1]), (startcoord[2][0], startcoord[2][1]), (0, 210, 255), 2)  # Vertical left line
        cv2.line(frame, (startcoord[2][0], startcoord[2][1]), (startcoord[3][0], startcoord[3][1]), (0, 210, 255), 2)  # Second horizontal line
        cv2.line(frame, (startcoord[1][0], startcoord[1][1]), (startcoord[3][0], startcoord[3][1]), (0, 210, 255), 2)  # Vertical right line

        # Detection Gateway

        cv2.line(frame, (coord[0][0], coord[0][1]), (coord[1][0], coord[1][1]), (0, 0, 255), 2)  # First horizontal line
        cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (0, 0, 255), 2)  # Vertical left line
        cv2.line(frame, (coord[2][0], coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)  # Second horizontal line
        cv2.line(frame, (coord[1][0], coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2)  # Vertical right line

        
        

        # Evaluate contours
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Evaluate Contours"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # only proceed if at least one contour was found


        if len(cnts) > 0:

            x = 0
            y = 0
            radius = 0
            center= (0,0)
            
            for index in range(len(cnts)):
                circle = (0,0,0)
                center= (0,0)
                radius = 0
                # Eliminate countours that are outside the y dimensions of the detection zone
                ((tempcenterx, tempcentery), tempradius) = cv2.minEnclosingCircle(cnts[index])
                tempcenterx = tempcenterx + sx1
                tempcentery = tempcentery + y1
                if (tempcentery >= y1 and tempcentery <= y2):
                    rangefactor = 50
                    cv2.drawContours(mask, cnts, index, (60, 255, 255), 1)
                    #cv2.putText(frame,"Radius:"+str(int(tempradius)),(int(tempcenterx)+3, int(tempcentery)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                    # Eliminate countours significantly different than startCircle by comparing radius in range
                    if (started == True and startCircle[2]+rangefactor > tempradius and startCircle[2]-rangefactor < tempradius):
                        x = int(tempcenterx)
                        y = int(tempcentery)
                        radius = int(tempradius)
                        center= (x,y)
                    else:
                        if not started:
                            x = int(tempcenterx)
                            y = int(tempcentery)
                            radius = int(tempradius)
                            center= (x,y)
                            #print("No Startpoint Set Yet: "+str(center)+" "+str(startCircle[2]+rangefactor)+" > "+str(radius)+" AND "+str(startCircle[2]-rangefactor)+" < "+str(radius))
                else:
                    break
                
                
                # Find Start
                if profiling == True:
                    step = "Step " + str(stepId) + ":" + "Find Start"
                    profiling_data.append((time.perf_counter(),step))
                    stepId = stepId + 1

                # only proceed if the radius meets a minimum size
                if radius >=5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points  
                    circle = (x,y,radius)
                    if circle:
                        # check if the circle is stable to detect if a new start is there
                        if not started or startPos[0]+10 <= center[0] or startPos[0]-10 >= center[0]:
                            if (center[0] >= sx1 and center[0] <= sx2):
                                startCandidates.append(center)
                                if len(startCandidates) > startminimum :
                                    startCandidates.pop(0)
                                    #filtered = startCandidates.filter(center.x == value.x and center.y == value.y)
                                    arr = np.array(startCandidates)
                                    # Create an empty list
                                    filter_arr = []
                                    # go through each element in arr
                                    for element in arr:
                                    # if the element is completely divisble by 2, set the value to True, otherwise False
                                        if (element[0] == center[0] and center[1] == element[1]):
                                            filter_arr.append(True)
                                        else:
                                            filter_arr.append(False)

                                    filtered = arr[filter_arr]

                                    #print(filtered)
                                    if len(filtered) >= (startminimum/2):
                                        print("New Start Found")
                                        replayavail = False
                                        noOfStarts = noOfStarts + 1
                                        lastShotSpeed = 0
                                        pts.clear()
                                        tims.clear()
                                        filteredcircles = []
                                        filteredcircles.append(circle)
                                        startCircle = circle
                                        startPos = center
                                        startTime = frameTime
                                        #print("Start Position: "+ str(startPos[0]) +":" + str(startPos[1]))
                                        # Calculate the pixel per mm ratio according to z value of circle and standard radius of 2133 mm
                                        if ballradius == 0:
                                            pixelmmratio = circle[2] / golfballradius
                                        else:
                                            pixelmmratio = ballradius / golfballradius
                                        #print("Pixel ratio to mm: " +str(pixelmmratio))    
                                        started = True
                                        replay = True
                                        replaytrigger = 0          
                                        entered = False
                                        left = False
                                        # update the points and tims queues - add frame to the queue for async spin detection
                                        pts.appendleft(center)
                                        tims.appendleft(frameTime) 
                                        global zoomframe1
                                        deltaangle = 0
                                        if calcSpin == True:
                                            zoomframe1, zoomangle, (shapex1,shapex2,shapey1,shapey2) = showCircleContours(startCircle[0],startCircle[1], startCircle[2],origframe,frame)
                                            if shapex1 != 0 and shapex2 != 0 and shapey1 != 0 and shapey2 != 0 and zoomangle != 0:
                                                spin1 = True
                                                shapeangle1 = zoomangle
                                        global replay1
                                        global replay2

                                        replay1 = cv2.VideoWriter('replay1/Replay1_'+ str(noOfStarts) +'.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(width), int(height)))
                                        if replaycam == 1:
                                            replay2 = cv2.VideoWriter('replay2/Replay2_'+ str(noOfStarts) +'.mp4', apiPreference=0, fourcc=fourcc,fps=120, frameSize=(int(replaycamwidth), int(replaycamheight)))

                            else:
                                
                                
                                if (x >= coord[0][0] and entered == False and started == True):
                                    
                
                                    # Find Entered
                                    if profiling == True:
                                        step = "Step " + str(stepId) + ":" + "Find Entered"
                                        profiling_data.append((time.perf_counter(),step))
                                        stepId = stepId + 1
                                    cv2.line(frame, (coord[0][0], coord[0][1]), (coord[2][0], coord[2][1]), (0, 255, 0),2)  # Changes line color to green
                                    tim1 = frameTime
                                    print("Ball Entered. Position: "+str(center))
                                    startPos = center
                                    entered = True
                                    # update the points and tims queues
                                    pts.appendleft(center)
                                    tims.appendleft(frameTime)
                                    global zoomframe2
                                    
                                    if calcSpin == True:
                                        zoomframe2, zoomangle, (shapex1,shapex2,shapey1,shapey2) = showCircleContours(startPos[0],startPos[1], startCircle[2],origframe,frame)
                                        if shapex1 != 0 and shapex2 != 0 and shapey1 != 0 and shapey2 != 0 and zoomangle != 0:
                                            spin2 = True
                                            shapeangle2 = zoomangle
                                            if spin1 == True:
                                                deltaangle = shapeangle2 - shapeangle1
                                        
                                    break
                                else:
                                    if ( x > coord[1][0] and entered == True and started == True):
                                        # Find Left
                                        if profiling == True:
                                            step = "Step " + str(stepId) + ":" + "Find Left"
                                            profiling_data.append((time.perf_counter(),step))
                                            stepId = stepId + 1
                                        #calculate hla for circle and pts[0]
                                        previousHLA = (GetAngle((startCircle[0],startCircle[1]),pts[0])*-1)
                                        #calculate hla for circle and now
                                        currentHLA = (GetAngle((startCircle[0],startCircle[1]),center)*-1)
                                        #check if HLA is inverted
                                        similarHLA = False
                                        if left == True:
                                            if ((previousHLA <= 0 and currentHLA <=2) or (previousHLA >= 0 and currentHLA >=-2)):
                                                hldDiff = (pow(currentHLA, 2) - pow(previousHLA, 2))
                                                if  hldDiff < 30:
                                                    similarHLA = True
                                            else:
                                                similarHLA = False
                                        else:
                                            similarHLA = True
                                        if ( x > (pts[0][0]+50)and similarHLA == True): # and (pow((y - (pts[0][1])), 2)) <= pow((y - (pts[1][1])), 2) 
                                            cv2.line(frame, (coord[1][0], coord[1][1]), (coord[3][0], coord[3][1]), (0, 255, 0),2)  # Changes line color to green
                                            tim2 = frameTime # Final time
                                            print("Ball Left. Position: "+str(center))
                                            left = True
                                            endPos = center
                                            global zoomframe3
                                            
                                            if calcSpin == True:
                                                zoomframe3, zoomangle, (shapex1,shapex2,shapey1,shapey2) = showCircleContours(endPos[0],endPos[1], startCircle[2],origframe,frame)
                                                if shapex1 != 0 and shapex2 != 0 and shapey1 != 0 and shapey2 != 0 and zoomangle != 0:
                                                    spin3 = True
                                                    shapeangle3 = zoomangle
                                                    if spin1 == True and deltaangle == 0:
                                                        # TODO: Check the timings between the frames for shapeangle - Delta should only be between consecutive frames --> Deltaangle / number of frames
                                                        deltaangle = shapeangle3 - shapeangle1
                                            # calculate the distance traveled by the ball in pixel
                                            a = endPos[0] - startPos[0]
                                            b = endPos[1] - startPos[1]
                                            distanceTraveled = math.sqrt( a*a + b*b )
                                            if not pixelmmratio is None:
                                                # convert the distance traveled to mm using the pixel ratio
                                                distanceTraveledMM = distanceTraveled / pixelmmratio
                                                # take the time diff from ball entered to this frame
                                                timeElapsedSeconds = (tim2 - tim1)
                                                # calculate the speed in MPH
                                                if not timeElapsedSeconds  == 0:
                                                    speed = ((distanceTraveledMM / 1000 / 1000) / (timeElapsedSeconds)) * 60 * 60 * 0.621371
                                                # debug out
                                                print("Time Elapsed in Sec: "+str(timeElapsedSeconds))
                                                print("Distance travelled in MM: "+str(distanceTraveledMM))
                                                print("Speed: "+str(speed)+" MPH")
                                                # update the points and tims queues
                                                pts.appendleft(center)
                                                tims.appendleft(frameTime)
                                                break
                                        else:
                                            print("False Exit after the Ball")

                                            # flip image on y-axis for view only
        # loop over the set of tracked points
        if len(pts) != 0 and entered == True:
            for i in range(1, len(pts)):
                
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines 
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 150), 1)
                


            timeSinceEntered = (frameTime - tim1)
            replaytrigger = tim1

        if left == True:

            # Send Shot Data
            if (tim2 and timeSinceEntered > resetinseconds and distanceTraveledMM and timeElapsedSeconds and speed >= 0.5 and speed <= 200):
                # Send Shot
                if profiling == True:
                    step = "Step " + str(stepId) + ":" + "Send Shot"
                    profiling_data.append((time.perf_counter(),step))
                    stepId = stepId + 1
                print("----- Shot Complete --------")
                print("Time Elapsed in Sec: "+str(timeElapsedSeconds))
                print("Distance travelled in MM: "+str(distanceTraveledMM))
                print("Speed: "+str(speed)+" MPH")

                #     ballSpeed: ballData.BallSpeed,
                #     totalSpin: ballData.TotalSpin,
                totalSpin = 0
                #     hla: ballData.LaunchDirection,
                launchDirection = (GetAngle((startCircle[0],startCircle[1]),endPos)*-1)
                print("HLA: Line"+str((startCircle[0],startCircle[1]))+" Angle "+str(launchDirection))
                #Decimal(launchDirection);
                if (launchDirection > -40 and launchDirection < 40):

                    lastShotStart = (startCircle[0],startCircle[1])
                    lastShotEnd = endPos
                    lastShotSpeed = speed
                    lastShotHLA = launchDirection
                        
                    # Data that we will send in post request.
                    data = {"ballData":{"BallSpeed":"%.2f" % speed,"TotalSpin":totalSpin,"LaunchDirection":"%.2f" % launchDirection}}

                    # The POST request to our node server
                    if args["ballcolor"] == "calibrate":
                        print("calibration mode - shot data not send")
                    else:
                        try:
                            res = requests.post('http://127.0.0.1:8888/putting', json=data)
                            res.raise_for_status()
                            # Convert response data to json
                            returned_data = res.json()

                            print(returned_data)
                            result = returned_data['result']
                            print("Response from Node.js:", result)

                        except requests.exceptions.HTTPError as e:  # This is the correct syntax
                            print(e)
                        except requests.exceptions.RequestException as e:  # This is the correct syntax
                            print(e)
                else:
                    print("Misread on HLA - Shot not send!!!")    
                if len(pts) > calObjectCount:
                    calObjectCount = len(pts)
                print("----- Data reset --------")
                started = False
                entered = False
                left = False
                speed = 0
                timeSinceEntered = 0
                tim1 = 0
                tim2 = 0
                distanceTraveledMM = 0
                timeElapsedSeconds = 0
                startCircle = (0, 0, 0)
                endCircle = (0, 0, 0)
                startPos = (0,0)
                endPos = (0,0)
                startTime = time.time()
                pixelmmratio = 0
                pts.clear()
                tims.clear()

                # Further clearing - startPos, endPos
        else:
            # Send Shot Data
            if (tim1 and timeSinceEntered > resetinseconds):
                print("----- Data reset --------")
                started = False
                entered = False
                left = False
                replay = False
                speed = 0
                timeSinceEntered = 0
                tim1 = 0
                tim2 = 0
                distanceTraveledMM = 0
                timeElapsedSeconds = 0
                startCircle = (0, 0, 0)
                endCircle = (0, 0, 0)
                startPos = (0,0)
                endPos = (0,0)
                startTime = time.time()
                pixelmmratio = 0
                pts.clear()
                tims.clear()
                
        #cv2.putText(frame,"entered:"+str(entered),(20,180),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        #cv2.putText(frame,"FPS:"+str(fps),(20,200),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))


        # Draw Output
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Draw Output"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        if not lastShotSpeed == 0:
            cv2.line(frame,(lastShotStart),(lastShotEnd),(0, 255, 255),4,cv2.LINE_AA)      
        
        if started:
            cv2.line(frame,(sx2,startCircle[1]),(sx2+400,startCircle[1]),(255, 255, 255),4,cv2.LINE_AA)
        else:
            cv2.line(frame,(sx2,int(y1+((y2-y1)/2))),(sx2+400,int(y1+((y2-y1)/2))),(255, 255, 255),4,cv2.LINE_AA) 

            # Mark Start Circle
        if started:
            cv2.circle(frame, (startCircle[0],startCircle[1]), startCircle[2],(0, 0, 255), 2)
            cv2.circle(frame, (startCircle[0],startCircle[1]), 5, (0, 0, 255), -1)
            if spin1 == True:
                linepos1 = (startCircle[0],startCircle[1])
                linepos2 = (int(startCircle[0]+(startCircle[2]*math.cos(math.radians(shapeangle1)))),int(startCircle[1]+(startCircle[2]*math.sin(math.radians(shapeangle1)))))
                cv2.line(frame, (linepos1), (linepos2),(0, 0, 255), 2)
            #if debug == True:
                cv2.imshow("zoom1",zoomframe1)

        # Mark Entered Circle
        if entered:
            cv2.circle(frame, (startPos), startCircle[2],(0, 0, 255), 2) 
            if spin2 == True:
                linepos1 = startPos
                linepos2 = (int(startPos[0]+(startCircle[2]*math.cos(math.radians(shapeangle2)))),int(startPos[1]+(startCircle[2]*math.sin(math.radians(shapeangle2)))))
                cv2.line(frame, (linepos1), (linepos2),(0, 0, 255), 2)
            #if debug == True:
                cv2.imshow("zoom2",zoomframe2)

        # Mark Exit Circle
        if left:
            cv2.circle(frame, (endPos), startCircle[2],(0, 0, 255), 2)
            # draw a line from endPos with the angle shapeangle3 and  length as startCircle[2]
            if spin3 == True:
                linepos1 = endPos
                linepos2 = (int(endPos[0]+(startCircle[2]*math.cos(math.radians(shapeangle3)))),int(endPos[1]+(startCircle[2]*math.sin(math.radians(shapeangle3)))))
                cv2.line(frame, (linepos1), (linepos2),(0, 0, 255), 2)
            #if debug == True:
                cv2.imshow("zoom3",zoomframe3)

        if flipView:	
            frame = cv2.flip(frame, -1)
                                        
        cv2.putText(frame,"Start Ball",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        cv2.putText(frame,"x:"+str(startCircle[0]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        cv2.putText(frame,"y:"+str(startCircle[1]),(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))

        if not lastShotSpeed == 0:
            cv2.putText(frame,"Last Shot",(400,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
            cv2.putText(frame,"Ball Speed: %.2f" % lastShotSpeed+" MPH",(400,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
            cv2.putText(frame,"HLA:  %.2f" % lastShotHLA+" Degrees",(400,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)

        if not deltaangle == 0:
            cv2.putText(frame,"Est. Spin Axis:  %.2f" % deltaangle+" Degrees",(400,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
            print("Est. Spin Axis:", deltaangle)

        
        if ballradius == 0:
            cv2.putText(frame,"radius:"+str(startCircle[2]),(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        else:
            cv2.putText(frame,"radius:"+str(startCircle[2])+" fixed at "+str(ballradius),(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))    

        cv2.putText(frame,"Actual FPS: %.2f" % fps,(200,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        if overwriteFPS != 0:
            cv2.putText(frame,"Fixed FPS: %.2f" % overwriteFPS,(400,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        else:
            cv2.putText(frame,"Detected FPS: %.2f" % video_fps[0],(400,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
        
        #if args.get("video", False):
        #    out1.write(frame)
        if (args["ballcolor"] == "calibrate"):
            if out2:
                try:
                    out2.write(origframe)
                except Exception as e:
                    print(e)

        # Record Replay1 Video

        if replay == True:
            # Write Replay
            if profiling == True:
                step = "Step " + str(stepId) + ":" + "Write Replay"
                profiling_data.append((time.perf_counter(),step))
                stepId = stepId + 1
            if replaytrigger != 0:
                timeSinceTriggered = frameTime - replaytrigger
            if timeSinceTriggered < 3:
                replay1queue.appendleft(origframe)
                if replaycam == 1:
                    replay2queue.appendleft(origframe2)
            else:
                print("Replay recording stopped")

        try:
            if len(replay1queue) > 0 and replaytrigger != 0:
                replay1frame = replay1queue.pop()
                replay1.write(replay1frame)
                if replaycam == 1:
                    replay2frame = replay2queue.pop()
                    replay2.write(replay2frame)
        except Exception as e:
            print(e)

        try:
            if replaytrigger != 0 and timeSinceTriggered > 3 :
                while len(replay1queue) > 0:
                    replay1frame = replay1queue.pop()
                    replay1.write(replay1frame)                
                replay1.release()
                print("Replay 1 released")
                # grab the replay video
                global vs_replay1
                vs_replay1 = cv2.VideoCapture('replay1/Replay1_'+ str(noOfStarts) +'.mp4')
                replayavail = True
                frameskip = 0
                replay1queue.clear()
                if replaycam == 1:
                    while len(replay2queue) > 0:
                        replay2frame = replay2queue.pop()
                        replay2.write(replay2frame)             
                    replay2.release()
                    print("Replay 2 released")
                    global vs_replay2
                    vs_replay2 = cv2.VideoCapture('replay2/Replay2_'+ str(noOfStarts) +'.mp4')
                    replay2queue.clear()
                replaytrigger = 0
                timeSinceTriggered = 0
                replay = False
                print("Replay reset")
        except Exception as e:
            print(e)

        if showreplay == 1 and replayavail == True:
            frameskip = frameskip + 1
            if frameskip%2 == 0:
                # grab the current frame from Replay1
                _, frame_vs_replay1 = vs_replay1.read()
                if frame_vs_replay1 is not None:
                    cv2.imshow("Replay1", frame_vs_replay1)
                else:
                    print("Reset Replay Video")
                    vs_replay1 = cv2.VideoCapture('replay1/Replay1_'+ str(noOfStarts) +'.mp4')
                if replaycam == 1:
                    _, frame_vs_replay2 = vs_replay2.read()
                    if frame_vs_replay2 is not None:
                        cv2.imshow("Replay2", frame_vs_replay2)
                                    
                    else:
                        print("Reset Replay Video")
                        vs_replay2 = cv2.VideoCapture('replay2/Replay2_'+ str(noOfStarts) +'.mp4')    
    
        # show main putting window
                        
        # Show Output      
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Show Output"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        outputframe = resizeWithAspectRatio(frame, width=int(args["resize"]))
        parent_output_conn.send(outputframe)

        key = cv2.waitKey(1) & 0xFF
        # key = ord("s")
        # if the 'q' key is pressed, stop the loop
        if (key == ord("q") or key == ord("Q")):
            break
        
        if (key == ord("a") or key == ord("A")):

            if not a_key_pressed:
                cv2.namedWindow("Advanced Settings")
                if (mjpegenabled != 0 and maincamtype != 2):
                    vs.set(cv2.CAP_PROP_SETTINGS, 37)  
                cv2.resizeWindow("Advanced Settings", 1000, 440)
                cv2.createTrackbar("X Start", "Advanced Settings", int(sx1), 640, setXStart)
                cv2.createTrackbar("X End", "Advanced Settings", int(sx2), 640, setXEnd)
                cv2.createTrackbar("Y Start", "Advanced Settings", int(y1), 460, setYStart)
                cv2.createTrackbar("Y End", "Advanced Settings", int(y2), 460, setYEnd)
                cv2.createTrackbar("Radius", "Advanced Settings", int(ballradius), 50, setBallRadius)
                cv2.createTrackbar("Flip Image", "Advanced Settings", int(flipImage), 1, setFlip)
                cv2.createTrackbar("Flip View", "Advanced Settings", int(flipView), 1, setFlipView)
                cv2.createTrackbar("MJPEG", "Advanced Settings", int(mjpegenabled), 1, setMjpeg)
                cv2.createTrackbar("FPS", "Advanced Settings", int(overwriteFPS), 240, setOverwriteFPS)
                cv2.createTrackbar("Darkness", "Advanced Settings", int(darkness), 255, setDarkness)
                # cv2.createTrackbar("Saturation", "Advanced Settings", int(saturation), 255, setSaturation)
                # cv2.createTrackbar("Exposure", "Advanced Settings", int(exposure), 255, setExposure)
                a_key_pressed = True
            else:
                cv2.destroyWindow("Advanced Settings")
                
                if maincamtype != 2:
                    exposure = vs.get(cv2.CAP_PROP_EXPOSURE)
                    saturation = vs.get(cv2.CAP_PROP_SATURATION)
                    autowb = vs.get(cv2.CAP_PROP_AUTO_WB)
                    whiteBalanceBlue = vs.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
                    whiteBalanceRed = vs.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
                    brightness = vs.get(cv2.CAP_PROP_BRIGHTNESS)
                    contrast = vs.get(cv2.CAP_PROP_CONTRAST)
                    hue = vs.get(cv2.CAP_PROP_HUE)
                    gain = vs.get(cv2.CAP_PROP_GAIN)
                    monochrome = vs.get(cv2.CAP_PROP_MONOCHROME)
                    sharpness = vs.get(cv2.CAP_PROP_SHARPNESS)
                    autoexposure = vs.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    gamma = vs.get(cv2.CAP_PROP_GAMMA)
                    zoom = vs.get(cv2.CAP_PROP_ZOOM)
                    focus = vs.get(cv2.CAP_PROP_FOCUS)
                    autofocus = vs.get(cv2.CAP_PROP_AUTOFOCUS)


                    print("Saving Camera Settings to config.ini for restart")

                    parser.set('putting', 'exposure', str(exposure))
                    parser.set('putting', 'saturation', str(saturation))
                    parser.set('putting', 'autowb', str(autowb))
                    parser.set('putting', 'whiteBalanceBlue', str(whiteBalanceBlue))
                    parser.set('putting', 'whiteBalanceRed', str(whiteBalanceRed))
                    parser.set('putting', 'brightness', str(brightness))
                    parser.set('putting', 'contrast', str(contrast))
                    parser.set('putting', 'hue', str(hue))
                    parser.set('putting', 'gain', str(gain))
                    parser.set('putting', 'monochrome', str(monochrome))
                    parser.set('putting', 'sharpness', str(sharpness))
                    parser.set('putting', 'autoexposure', str(autoexposure))
                    parser.set('putting', 'gamma', str(gamma))
                    parser.set('putting', 'zoom', str(zoom))
                    parser.set('putting', 'focus', str(focus))
                    parser.set('putting', 'autofocus', str(autofocus))

                    parser.write(open(CFG_FILE, "w"))

                a_key_pressed = False

        if (key == ord("d") or key == ord("D")):
            if not d_key_pressed:
                args["debug"] = 1
                myColorFinder = ColorFinder(True)
                myColorFinder.setTrackbarValues(hsvVals)
                d_key_pressed = True
                debug = True
            else:
                args["debug"] = 0            
                myColorFinder = ColorFinder(False)
                cv2.destroyWindow("Original")
                cv2.destroyWindow("MaskFrame")
                cv2.destroyWindow("TrackBars")
                d_key_pressed = False
                debug = False
            
        

        
        
                
        if args.get("debug", False):    
            # flip image on y-axis for view only
            if flipView:	
                mask = cv2.flip(mask, flipView)	
                origframe = cv2.flip(origframe, flipView)
            cv2.imshow("MaskFrame", mask)
            cv2.imshow("Original", origframe)

        
        if replaycam == 1:
            cv2.imshow("Replay Camera", origframe2)

        
                            
        # FPS handling   
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "FPS Handling"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1

        # if actualFPS > 1:
        #     grayPreviousFrame = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2GRAY)
        #     grayOrigframe = cv2.cvtColor(origframe, cv2.COLOR_BGR2GRAY)
        #     changedFrame = cv2.compare(grayPreviousFrame, grayOrigframe,cv2.CMP_NE)
        #     nz = cv2.countNonZero(changedFrame)
        #     #print(nz)
        #     if nz == 0:
        #         actualFPS = actualFPS - 1
        #         fpsqueue.pop()
        # previousFrame = origframe.copy()

        
        #--------------- No logic after this point -----------------
        # Show Profiling Data   
        if profiling == True:
            step = "Step " + str(stepId) + ":" + "Finished"
            profiling_data.append((time.perf_counter(),step))
            stepId = stepId + 1
            steps = len(profiling_data)
            for i in range(steps):
                # print profiling data
                (stepstarttime, stepname) = profiling_data[i]
                if i != (steps - 1):
                    (stependtime, nextstepname) = profiling_data[i+1]
                else:
                    stependtime = time.perf_counter()
                    nextstepname = "End"
                print("Step ID: %d   Start: %f   End: %f   Step Time: %f   Step Name: %s"
                    % (i, stepstarttime, stependtime, stependtime-stepstarttime, stepname))
        # if outputqueue.empty() == False:
        #     if outputqueue.get() == "QUIT":
        #         break

        
        

    if maincamtype == 0 or maincamtype == 1:
        # close all windows
        vs.release()

    if maincamtype == 2:  
        parent_frame_conn.send(1) 

    if replaycam == 1:
        vs2.release()

    cv2.destroyAllWindows()
    frameprocess.terminate()
    time.sleep(1)
    frameprocess.close()


if __name__ == "__main__":
    main()