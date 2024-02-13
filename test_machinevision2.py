import gxipy as gx
from PIL import Image
import datetime

"""
Author:NoamaNelson
Date:2019-11-21
Discription:Secondary development of pythonsdk of Daheng camera.
"""

def main():
    
    Width_set = 600 # Set the resolution width
    Height_set = 360 # Set high resolution
    framerate_set = 750 # Set frame rate
    num = 10000 # Acquisition frame rate times (for debugging purposes, you can set the subsequent image acquisition to a while loop for unlimited loop acquisition)

    
    counter = 0
    last_frame_id = 0
    
    #print
    print("")
    print("###############################################################")
    print("Continuously acquire color images and display the acquired images.")
    print("###############################################################")
    print("")
    print("Camera initialization...")
    print("")
 
    #Create device
    device_manager = gx.DeviceManager() # Create device object
    dev_num, dev_info_list = device_manager.update_device_list() #Enumerate devices, that is, enumerate all available devices
    if dev_num is 0:
        print("Number of enumerated devices is 0")
        return
    else:
        print("")
        print("**********************************************************")
        print("The device was created successfully, the device number is: %d" % dev_num)

    #Open a device by device serial number
    cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))

    #If it is a black and white camera
    if cam.PixelColorFilter.is_implemented() is False: # is_implemented Determine whether the enumerated attribute parameter has been implemented
        print("This example does not support black and white cameras.")
        cam.close_device()
        return
    else:
        print("")
        print("**********************************************************")
        print("Open the color camera successfully, the SN number is: %s" % dev_info_list[0].get("sn"))


    #Set width and height
    cam.Width.set(Width_set)
    cam.Height.set(Height_set)
    
    #Set up continuous collection
    #cam.TriggerMode.set(gx.GxSwitchEntry.OFF) # Set trigger mode
    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    #Set the frame rate
    cam.AcquisitionFrameRate.set(framerate_set)

    # set continuous acquisition
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    
    # cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

    # set exposure
    cam.ExposureTime.set(500.0)

    # set gain
    cam.Gain.set(24.0)

    # set white balance
    cam.BalanceWhiteAuto.set(gx.GxAutoEntry.ONCE)

    cam.LightSourcePreset.set(gx.GxLightSourcePresetEntry.DAYLIGHT_6500K)


    print("")
    print("**********************************************************")
    print("The frame rate set by the user: %d fps"%framerate_set)
    framerate_get = cam.CurrentAcquisitionFrameRate.get() #Get the frame rate of the current acquisition
    print("The frame rate of the current acquisition: %d fps"%framerate_get)


     #Start data collection
    print("")
    print("**********************************************************")
    print("Start data collection...")
    print("")
    cam.stream_on()

    #Capture image
    for i in range(num):
        
        counter = counter + 1
        raw_image = cam.data_stream[0].get_image() # Open the 0th channel data stream
        if raw_image is None:
            print("Failed to obtain color original image.")
            continue

        # rgb_image = raw_image.convert("RGB") # Get RGB image from color original image
        # if rgb_image is None:
        #     continue

        # #rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut) # realize image enhancement

        # numpy_image = rgb_image.get_numpy_array() # Create numpy array from RGB image data
        # if numpy_image is None:
        #     continue

        # img = Image.fromarray(numpy_image, 'RGB') # Show the acquired image
        # #img.show()
        # mtime = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    
        # #img.save(r"D:\image\\" + str(i) + str("-") + mtime + ".jpg") # Save the picture locally

        print("Frame ID: %d   Height: %d   Width: %d   framerate_set:%dfps   framerate_get:%dfps"
              % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width(), framerate_set, framerate_get)) # Print the height, width, frame ID of the captured image, the frame rate set by the user, and the currently captured frame rate
        last_frame_id = raw_image.get_frame_id()
    
    print("**********************************************************")
    print("Lost Frames")
    print(last_frame_id)
    print(counter)
    lost = last_frame_id - counter
    print(lost/last_frame_id)


    #Stop collecting
    print("")
    print("**********************************************************")
    print("Camera has stopped capturing")
    cam.stream_off()

    #Close device
    print("")
    print("**********************************************************")
    print("The system prompts you: The device has been turned off!")
    cam.close_device()

if __name__ == "__main__":
    main()

