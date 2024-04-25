#! /usr/bin/python3

from evdev import InputDevice, categorize, ecodes
from flask import Flask, Response
import cv2
import numpy as np
import threading
import RPi.GPIO as GPIO
import sys,time,random,math

GPIO.setmode(GPIO.BOARD)
GPIO.setup((11, 13, 15, 19), GPIO.OUT, initial=GPIO.LOW)

motor_pwm_period = 0.05
max_speed = 0.75
actual_speed1 = 0
actual_speed2 = 0
actual_speed3 = 0
actual_speed4 = 0

#top, center, bottom = [], [], []

x1, x2, x3 = 0, 0, 0
y1, y2, y3 = 0, 0, 0
ang = 0

app = Flask(__name__)

def __gstreamer_pipeline(
        camera_id,
        capture_width=480,
        capture_height=320,
        display_width=480,
        display_height=320,
        framerate=30,
        flip_method=0,
    ):
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
            % (
                    camera_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
            )
    )
def generate_frames():
    global top, center, bottom
    camera = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=2), cv2.CAP_GSTREAMER)
    while True:
        success, frame = camera.read()
        if not success:
           break
        else:
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width, channel = hls.shape
            top = hls[:height//100*50]
            center = hls[height//100*50:height//100*65]
            bottom = hls[height//100*65:]
            follow_line(top, center, bottom)
            frame = cv2.inRange(top,np.array([0,190,0]),np.array([150,255,200]))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def follow_line(top, center, bottom):
    global x1, x2, x3, y1, y2, y3, ang
    
    x1, y1 = white_point(top, x1, y1)
    x2, y2 = white_point(center, x2, y2)
    x3, y3 = white_point(bottom, x3, y3)
    ang = angle(x1, x2, x3, y1, y2, y3)

def white_point(frame, x, y):
    lower_white = np.array([0, 190, 0])
    upper_white = np.array([150, 255, 200])
    mask = cv2.inRange(frame, lower_white, upper_white)
    
    moments = cv2.moments(mask, 1)
    dm01 = moments['m01']
    dm10 = moments['m10']
    darea = moments['m00']
    if darea > 150:
        x = int(dm01/darea)
        y = int(dm10/darea)
    return x, y

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def angle(x1, x2, x3, y1, y2, y3):
    if x1 == x2 or x2 == x3:
        deg = 0
        return deg

    k1 = (y2 - y1)/(x2 - x1)
    k2 = (y3 - y2)/(x3 - x2)
    deg = np.degrees(np.arctan((k2 - k1) / (1 + k1*k2)))
    print(x1, x2, x3, y1, y2, y3, deg)
    return deg

def control_pwm1():
    global actual_speed1
    while control_pwm_enabled:
        if actual_speed1 > 0:
            GPIO.output(15, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed1,0) )
            
            GPIO.output(15, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed1),0) )
        else:
            time.sleep(motor_pwm_period)

def control_pwm2():
    global actual_speed2
    while control_pwm_enabled:
        if actual_speed2 > 0:
            GPIO.output(19, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed2,0) )
        
            GPIO.output(19, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed2),0) )
        else:
            time.sleep(motor_pwm_period)

def control_pwm3():
    global actual_speed3
    while control_pwm_enabled:
        if actual_speed3 > 0:
            GPIO.output(11, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed3,0) )
        
            GPIO.output(11, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed3),0) )
        else:
            time.sleep(motor_pwm_period)

def control_pwm4():
    global actual_speed4
    while control_pwm_enabled:
        if actual_speed4 > 0:
            GPIO.output(13, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed4,0) )
        
            GPIO.output(13, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed4),0) )
        else:
            time.sleep(motor_pwm_period)

def camera_control():
    app.run(host='0.0.0.0', port=4444)
 

control_pwm_thread1 = threading.Thread(target=control_pwm1, args=())
control_pwm_thread2 = threading.Thread(target=control_pwm2, args=())
control_pwm_thread3 = threading.Thread(target=control_pwm3, args=())
control_pwm_thread4 = threading.Thread(target=control_pwm4, args=())
control_camera = threading.Thread(target=camera_control, args=())
#control_line = threading.Thread(target=follow_line, args=())

control_pwm_enabled = True
#control_pwm_enabled = False
if control_pwm_enabled:
    control_pwm_thread1.start()
    control_pwm_thread2.start()
    control_pwm_thread3.start()
    control_pwm_thread4.start()
    control_camera.start()
    #control_line.start()
    
gamepad = InputDevice('/dev/input/event2')

#evdev takes care of polling the controller in a loop
for event in gamepad.read_loop():
    #print(categorize(event))
    if event.type == ecodes.EV_KEY:
        if event.code == ecodes.BTN_SELECT:
            if event.value == 0:
                break
    if ang > 0:
        normalized = (ang - 180) * 180
        print(ang, normalized)
        actual_speed3 = max(0, normalized/32768)
        actual_speed4 = max(0, -normalized/32768)
    elif ang < 0:
        normalized = (ang - 180) * 180
        actual_speed1 = max(0, -normalized/32768)
        actual_speed2 = max(0, normalized/32768)
    elif ang == 0:
        actual_speed1 = 0
        actual_speed2 = 0
        actual_speed3 = 0
        actual_speed4 = 0
    if event.type == ecodes.EV_ABS:
        if event.code == ecodes.ABS_RZ:
            normalized = (event.value - 128) * 256
            #print(event.value, normalized)
            actual_speed3 = max(0,  normalized/32768)
            actual_speed4 = max(0, -normalized/32768) 
        if event.code == ecodes.ABS_Y:
            normalized = (event.value - 128) * 256
            #print(event.value, normalized)
            actual_speed1 = max(0, -normalized/32768)
            actual_speed2 = max(0,  normalized/32768) 


if control_pwm_enabled:
    control_pwm_enabled = False
    control_pwm_thread1.join()
    control_pwm_thread2.join()
    control_pwm_thread3.join()
    control_pwm_thread4.join()
    control_camera.join()
    #control_line.join()
GPIO.cleanup()
