import time
import numpy as np
import cv2
import gxipy as gx
from PIL import Image

# print the demo information
print("")
print("-------------------------------------------------------------")
print("Sample to show how to acquire color image continuously and show acquired image in OpenCV.")
print("-------------------------------------------------------------")
print("")
print("Initializing......")
print("")

webcamindex = 1 # Number of Webcamindex

Width_set = 1280 # Set the resolution width
Height_set = 720 # Set high resolution
framerate_set = 145 # Set frame rate

gain = 6
exposure = 5000

counter = 0
last_frame_id = 0


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
    
     # create a device manager
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num is 0:
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
    if cam.PixelColorFilter.is_implemented() is False:
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

    targetframes = 1000

    while(True):
        targetframes = targetframes -1
        ret, frame = getNextFrame(cam)
        if ret == 1:
            continue

        # show imagecam

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or targetframes == 0:
            
            # stop data acquisition
            cam.stream_off()

            # close device
            cam.close_device()
            break
                    

    # release resource
    cv2.destroyAllWindows()
    print(last_frame_id)
    print(counter)
    lost = last_frame_id - counter
    print(lost/last_frame_id)
if __name__ == "__main__":
    main()