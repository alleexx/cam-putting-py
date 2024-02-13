import time
import numpy as np
import cv2
import gxipy as gx
from PIL import Image
# Add multiprocessing for frame aquire
from multiprocessing import Process, Queue, Pipe

# print the demo information
print("")
print("-------------------------------------------------------------")
print("Sample to show how to acquire color image continuously and show acquired image in OpenCV.")
print("-------------------------------------------------------------")
print("")
print("Initializing......")
print("")



def getFrames(child_frame_conn, framequeue):
    webcamindex = 1 # Number of Webcamindex

    Width_set = 1440 # Set the resolution width
    Height_set = 1080 # Set high resolution
    framerate_set = 192 # Set frame rate

    gain = 6
    exposure = 5000

    counter = 0
    last_frame_id = 0
    # time.sleep(3)
    # create a device manager
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("Number of enumerated devices is 0")
        return

    # open the first device
    cam = device_manager.open_device_by_index(webcamindex)

    #Set width and height
    cam.Width.set(Width_set)
    cam.Height.set(Height_set)
    
    #Set up continuous collection
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)

    #Set the frame rate
    cam.AcquisitionFrameRate.set(framerate_set)

    # exit when the camera is a mono camera
    if cam.PixelColorFilter.is_implemented() == False:
        print("This sample does not support mono camera.")
        cam.close_device()
        return

    # set continuous acquisition
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    
    # cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

    # set exposure
    cam.ExposureTime.set(exposure)

    # set gain
    cam.Gain.set(gain)

    # set white balance
    cam.BalanceWhiteAuto.set(gx.GxAutoEntry.ONCE)

    cam.LightSourcePreset.set(gx.GxLightSourcePresetEntry.DAYLIGHT_6500K)
      
    # start data acquisition
    cam.stream_on()

    # acquisition image: num is the image number
    device = cam
    num = 1000
    for i in range(num):

        if isinstance(device, cv2.VideoCapture):
            ret, frame = device.read()
            framequeue.put(frame)        

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
                #   % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width(), device.CurrentAcquisitionFrameRate.get(), time.time()))

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
                    framequeue.put(numpy_image)
                    last_frame_id = raw_image.get_frame_id()
    
    
    
    lost = last_frame_id - counter
    if lost == 0:
        print("No Lost Frames")
    else:
        print("Lost Frames: ", (lost/last_frame_id))
    
    
    # stop data acquisition
    cam.stream_off()

    # close device
    cam.close_device()

    print("All frames read from device")


def getNextFrame(device):

    if isinstance(device, gx.U3VDevice):
        # acquisition image: num is the image number
        global counter
        global last_frame_id
        num = 1
        for i in range(num):
            
            #time.sleep(0.1)
            # send software trigger command
            # device.TriggerSoftware.send_command()
            counter = counter + 1
            # get raw image
            raw_image = device.data_stream[0].get_image()
            if raw_image is None:
                print("Getting image failed.")
                continue

            # print height, width, and frame ID of the acquisition image
            print("NumID: %d    Frame ID: %d   Height: %d   Width: %d    Current FPS: %d     Current time: %d"
               % (i,raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width(), device.CurrentAcquisitionFrameRate.get(), time.time()))

            # get RGB image from raw image
            rgb_image = raw_image.convert("RGB")
            if rgb_image is None:
                continue

            # improve image quality
            #rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

            # create numpy array with data from raw image
            numpy_image = rgb_image.get_numpy_array()
            if numpy_image is None:
                return 1, numpy_image

            last_frame_id = raw_image.get_frame_id()
            return 0, numpy_image

              


def main():
    
    targetframes = 1000

    # Start frame aquire as process
    framequeue = Queue()
    parent_frame_conn, child_frame_conn = Pipe()
    frameprocess = Process(target=getFrames, args=(child_frame_conn,framequeue,))
    frameprocess.start()

    while(True):
        targetframes = targetframes -1
        
        # get webcam frame either from device or from queue
        # ret, frame = getNextFrame(vs)
        while framequeue.empty() == True:
            i = 1
            #time.sleep(0.2) 
            #print("Framequeue empty")
        frame = framequeue.get()

        # show imagecam
        
        cv2.imshow('frame',frame)
        #print("Frame displayed")

        if cv2.waitKey(1) & 0xFF == ord('q') or targetframes == 0:
            print("All frames read from queue")
            break
                    

    # release resource
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()