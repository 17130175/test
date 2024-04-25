#! /usr/bin/python3

from evdev import InputDevice, categorize, ecodes
import threading
import RPi.GPIO as GPIO
import sys,time,random,math
from flask import Flask, Response
import numpy as np
import cv2

TRIG = 16
ECHO = 18
w=0
start_time = 0

GPIO.setmode(GPIO.BOARD)
GPIO.setup((11, 13, 15, 19), GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

motor_pwm_period = 0.05
actual_speed1 = 0
actual_speed2 = 0
actual_speed3 = 0
actual_speed4 = 0

GPIO.output(TRIG, False)
print("Calibrating.....")
app = Flask(__name__)
time.sleep(2)
print("Start")

def __gstreamer_pipeline(
        camera_id,
        capture_width=320,
        capture_height=240,
        display_width=320,
        display_height=240,
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
    global actual_speed1, actual_speed4
    camera = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=2), cv2.CAP_GSTREAMER)
    x,y = 0, 0
    blue = np.uint8([[[255, 0, 0]]])
    hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    L_limit = np.array([100, 100, 100]) #hsvBlue[0][0][0] - 10, 100, 100
    U_limit = np.array([140, 255, 255]) #hsvBlue[0][0][0] + 10, 255, 255
    while True:
        success, frame = camera.read()
        height, width = frame.shape[:2]
        edge = 40
        dst = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(dst, (5, 5), 0)
        b_mask=cv2.inRange(blur, L_limit, U_limit)
        moments = cv2.moments(b_mask, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']
        if dArea > 150:
            x = int(dM10/dArea)
            y = int(dM01/dArea)
            cv2.circle(frame, (x,y), 10, (0,0,255), -1)
            cv2.circle(frame, (x,y//2), 10, (0,0,255), -1)
        if (x>(width/2+edge)) and x!=0:
            cv2.rectangle(frame, (0,0), (x-50,height), (0,255,0), -1)
            actual_speed1 = 1
            actual_speed3 = 1
            actual_speed2 = 0
            actual_speed4 = 0
        elif (x<(width/2-edge)) and x!=0:
            cv2.rectangle(frame, (x+50,0), (width,height), (0,255,0), -1)
            actual_speed4 = 1
            actual_speed2 = 1
            actual_speed1 = 0
            actual_speed3 = 0
        else:

            if w == 0:
                actual_speed1 = 1
                actual_speed4 = 1
            else:
                actual_speed1 = 0
                actual_speed4 = 0
            actual_speed2 = 0
            actual_speed3 = 0
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def video_enable():
    app.run(host='0.0.0.0', port=5000)

def control_pwm1():
    global actual_speed1, w
    while control_pwm_enabled:
        if w == 1:
            actual_speed1 = 0
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
    global actual_speed4, w
    while control_pwm_enabled:
        if w == 1:
            actual_speed4 = 0
        if actual_speed4 > 0:
            GPIO.output(13, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed4,0) )
        
            GPIO.output(13, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed4),0) )
        else:
            time.sleep(motor_pwm_period)

def hc_sensor():
    global w, start_time
    while True:
        GPIO.output(TRIG, True)
        time.sleep(0.0000001)
        GPIO.output(TRIG, False)

        while GPIO.input(ECHO)==0:
            pulse_start = time.time()

        while GPIO.input(ECHO)==1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = round(pulse_duration * 17150 + 1.15, 2)
        now_time = time.time()

        if distance<=20 and distance>=5:
            w=1

        if distance>20 and w==1 or now_time-start_time > 5:
            w=0

        if w==1:
            start_time = time.time()

control_video = threading.Thread(target=video_enable, args=())
control_sensor = threading.Thread(target=hc_sensor, args=())
control_pwm_thread1 = threading.Thread(target=control_pwm1, args=())
control_pwm_thread2 = threading.Thread(target=control_pwm2, args=())
control_pwm_thread3 = threading.Thread(target=control_pwm3, args=())
control_pwm_thread4 = threading.Thread(target=control_pwm4, args=())

control_pwm_enabled = True

if control_pwm_enabled:
    control_pwm_thread1.start()
    control_pwm_thread2.start()
    control_pwm_thread3.start()
    control_pwm_thread4.start()
    control_sensor.start()
    control_video.start()

gamepad = InputDevice('/dev/input/event2')

for event in gamepad.read_loop():
    if event.type == ecodes.EV_KEY:
        if event.code == ecodes.BTN_START:
            if event.value == 0:
                break

    if event.type == ecodes.EV_ABS:
        if event.code == ecodes.ABS_Y:
            normalized = (event.value - 128) * 256
            actual_speed1 = max(0, -normalized/32768)
            actual_speed2 = max(0, normalized/32768)
            actual_speed3 = max(0, normalized/32768)
            actual_speed4 = max(0, -normalized/32768)
        if event.code == ecodes.ABS_Z:
            normalized = (event.value - 128) * 256
            if normalized < 0:
                if w == 1:
                    actual_speed2 = max(0, -normalized/32768)
                else:
                    actual_speed2 = 0
                actual_speed4 = max(0, -normalized/32768)
                actual_speed1 = 0
                actual_speed3 = 0
            else:
                actual_speed1 = max(0, normalized/32768)
                if w == 1:
                    actual_speed3 = max(0, normalized/32768)
                else:
                    actual_speed3 = 0
                actual_speed2 = 0
                actual_speed4 = 0

if control_pwm_enabled:
    control_pwm_enabled = False
    control_pwm_thread1.join()
    control_pwm_thread2.join()
    control_pwm_thread3.join()
    control_pwm_thread4.join()
    control_sensor.join()
    control_video.join()

GPIO.cleanup()
