# Webcam based putting simulation for GSPRO utilizing the R10 and MLM2Pro connector:

Calculation includes BallSpeed in MPH and HLA of the putt. Initial insperation on the solution comes from natter where I forked the initial OpenCV code.

These connectors integrate the putting app to use in GSPRO

- extension of original R10 connector of Travis Lang https://github.com/alleexx/gspro-garmin-connect-v2
- matth version of R10 adapter with direct bluetooth support https://github.com/mholow/gsp-r10-adapter
- MLM2Pro connector from etienne https://github.com/springbok/MLM2PRO-GSPro-Connector

## Setup:

Here is a setup video from Roxor Golf / Bruce with the R10 Bluetooth connector https://www.youtube.com/watch?v=Ym1ozT9Ovew

You can find a general setup video with the my original R10 connector using the e6 method here: https://youtu.be/ShtEW0fziwM

A detailed video for tuning in the color options with the d - debug option from dialup56k with some of the theory behind here: https://www.youtube.com/watch?v=64npQCFaDAQ&t=583s

Here is also a good video on setup to reach 60 fps from jason https://www.youtube.com/watch?v=wy62vY95G_E&ab_channel=Jason

You can download the release or just install the packages and run the app via python.

Unpack the release zip and run ball_tracking.exe [-c (ballcolor OR calibrate) -w <webcamera number 0-5>]

- Position Webcam on the side and above the putting start area across from you - see video for example (flip image in "a" Advanced Settings to allow for left handed setups)
- Position the Ball (choose the right color or use calibrate as described below.) on a dark green surface - In my test the area visible to the webcam was about 1 meter high and across from me
- Adjust your webcam to the white line to reflect a straight putt and place your ball into the yellow rectangle.
- Use "a" to go to Advanced Settings - here you can adjust following settings
  - adjust camera settings by using the DirectShow camera settings window (only supported if MJPEG is enabled = 1 - dafault)
  - the start position rectangle
  - set a fixed radius if the detected radius is not consistent for you
  - flip webcam image if i.e. your a lefty
  - enable MJPEG option for changing the webcam codec to compressed MJPEG format - some webcams only support higher like 60 FPS settings on compressed video formats - auto detection of FPS settings will not work for this codec but acutal FPS should be accurate. In case the desired FPS is only available in certain resolution you can adjust/overwrite the resolution directly in the config.ini file
  - overwrite the detected FPS with a target FPS setting if not detected correctly - not all cameras support setting FPS through OpenCV
  - darken the images in case your webcam settings do not allow for this
  - beta option of ps4 enabling is done in config.ini directly
- Once identified the ball should get a red circle fully around. If it is smaller or bigger than the ball it will not reflect the right putting speed. It must match the ball dimensions as best as it can.
- If the ball is not detected try adjusting the light situation or your webcam brightness settings or try a different ball color option (hit q to exit the putting simulator and start again with another Ball Color)
- The putt needs to cross the red rectangle and needs to leave on the other side
- If a shot is detected it is send to http://localhost:8888/ where my extension of the garmin connect app (https://github.com/alleexx/gspro-garmin-connect-v2) is receiving the shot and passing it to GSPRO

Here are some collected FAQs you should also read if anything is in question:

FAQ https://github.com/alleexx/gspro-garmin-connect-v2/blob/main/FAQ.md

## WebCams

I have listed the webcam which were reported to me to work - I can not guarantee these are working in your setup though 

https://github.com/alleexx/cam-putting-py/wiki/Webcam-Compatability
  
## Some Troubleshooting steps if you can not get it to work

I guess you saw my setup video so here are some more details.

- Make sure to have a solid background. Green putting or hitting mat should be fine - The darker the better.
- Don't cast a direct shadow over the ball if possible have some light on the ball side
- Use a colored ball - orange works best for most people with color option orange2
- Use the advanced settings (hit a) to limit the detection area a bit around the ball placed
- Usage of MJPEG/DirectShow gives also now access to camera properties like saturation/exposure like you have it in kinovea - the window will open alongside if you hit "a" for advanced settings. This should help to find a good camera setting for ball detection. Adjusting the saturation, exposure and white level is helping to get a good reading. As all webcams are different this can not be done automatically and you have to do so manually to get a good result. Try to get a good red circle around the ball and try to eliminate the false detections/radius reading in other parts of the view. 
- Use the advanced settings darkness setting to limit the light reflections in the frame. Hopefully the ball will be detected in this way.
- If this does not help you can hit "d" and see the advanced debug options. You will see the mask frame which shows white for any detected object having the selected color scheme like "orange2". You can adjust the HSV values of the color scheme with the sliders and the result will be stored in the config.ini file and used to overwrite the color scheme going forward. If you want to reset to the default colors you need to delete the customhsv entry from config.ini file. This is an advanced option - you can read up on HSV and color detection using OpenCV if you are interested. 

Software is open source. Feel Free to buy me a coffee or a sleeve of golf balls if you like it.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/U6U2I70HX)

<img src="Camera-Putting-Alignment.png">

Here is a short video demonstration of the prototype

https://youtu.be/ZgcH25WkCWQ

## Available Color Options and HSV values used

#red

red = {'hmin': 1, 'smin': 208, 'vmin': 0, 'hmax': 50, 'smax': 255, 'vmax': 249} # bright environment 

red2 = {'hmin': 1, 'smin': 240, 'vmin': 61, 'hmax': 50, 'smax': 255, 'vmax': 249} # dark environment

#white 

white = {'hmin': 168, 'smin': 218, 'vmin': 118, 'hmax': 179, 'smax': 247, 'vmax': 216} # very bright environment

white2 = {'hmin': 159, 'smin': 217, 'vmin': 152, 'hmax': 179, 'smax': 255, 'vmax': 255} # bright environment 

white3 = {'hmin': 0, 'smin': 181, 'vmin': 0, 'hmax': 42, 'smax': 255, 'vmax': 255}# test environment

#yellow 

yellow = {'hmin': 0, 'smin': 210, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255} # bright environment 

yellow2 = {'hmin': 0, 'smin': 150, 'vmin': 100, 'hmax': 46, 'smax': 255, 'vmax': 206} # dark environment

#green 

green = {'hmin': 0, 'smin': 169, 'vmin': 161, 'hmax': 177, 'smax': 204, 'vmax': 255} # bright environment 

green2 = {'hmin': 0, 'smin': 109, 'vmin': 74, 'hmax': 81, 'smax': 193, 'vmax': 117} # dark environment

#orange 

orange = {'hmin': 0, 'smin': 219, 'vmin': 147, 'hmax': 19, 'smax': 255, 'vmax': 255}# bright environment 

orange2 = {'hmin': 3, 'smin': 181, 'vmin': 134, 'hmax': 40, 'smax': 255, 'vmax': 255}# dark environment

orange3 = {'hmin': 0, 'smin': 73, 'vmin': 150, 'hmax': 40, 'smax': 255, 'vmax': 255}# test environment

orange4 = {'hmin': 3, 'smin': 181, 'vmin': 216, 'hmax': 40, 'smax': 255, 'vmax': 255}# custom ps3eye environment
  
## Details on the python app

You can install the necessary packages by running pip install with the requirements file - some mentioned packages might not be necessary to run it as I did not clean it up yet

Install the app:

"pip install -r requirements.txt"

Run the app:

"python ball_tracking.py"

Default color to be found is yellow - You can adapt this by running -c as color options yellow, green, orange, red or white. White is a very general option if no color ball is availabe but might give you more false reads on other white objects. Best options for me are orange, yellow and red.

"python ball_tracking.py -c orange"

In the app you can hit "a" for advanced settings to define the ball detection area, flip and darken image or hit "d" to get a debug view of the video feed to see the color detection mask.

There are also other options to for start to pass a video file or image file

usage: ball_tracking.py [-h] [-v VIDEO] [-i IMG] [-b BUFFER] [-w CAMERA] [-c BALLCOLOR] [-d DEBUG]

options:

  -h, --help                  show this help message and exit

  -v VIDEO, --video VIDEO     path to the (optional) video file

  -i IMG, --img IMG           path to the (optional) image file

  -b BUFFER, --buffer BUFFER  max buffer size - default is 64

  -w CAMERA, --camera CAMERA  webcam index number - default is 0

  -c BALLCOLOR, --ballcolor   ball color - default is white

  -d DEBUG, --debug DEBUG     debug - color finder and wait timer




