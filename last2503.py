#! /usr/bin/python3

from evdev import InputDevice, categorize, ecodes
import threading
import RPi.GPIO as GPIO
import sys,time,random,math

TRIG = 16
ECHO = 18
i=0

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
time.sleep(2)
print("Start")

def control_pwm1():
    global actual_speed1, i
    while control_pwm_enabled:
        if i == 1:
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
    global actual_speed4, i
    while control_pwm_enabled:
        if i == 1:
            actual_speed4 = 0
        if actual_speed4 > 0:
            GPIO.output(13, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed4,0) )
        
            GPIO.output(13, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed4),0) )
        else:
            time.sleep(motor_pwm_period)

def hc_sensor():
    global i
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

        if distance<=20 and distance>=5:
            i=1

        if distance>20 and i==1:
            i=0

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

gamepad = InputDevice('/dev/input/event2')

for event in gamepad.read_loop():
    if event.type == ecodes.EV_KEY:
        if event.code == ecodes.BTN_START:
            if event.value == 0:
                break

    if event.type == ecodes.EV_ABS:
        print(categorize(event))
        if event.code == ecodes.ABS_Y:
            normalized = (event.value - 128) * 256
            actual_speed1 = max(0, -normalized/32768)
            actual_speed2 = max(0, normalized/32768)
            actual_speed3 = max(0, normalized/32768)
            actual_speed4 = max(0, -normalized/32768)
        if event.code == ecodes.ABS_Z:
            normalized = (event.value - 128) * 256
            if normalized < 0:
                if i == 1:
                    actual_speed2 = max(0, -normalized/32768)
                else:
                    actual_speed2 = 0
                actual_speed4 = max(0, -normalized/32768)
                actual_speed1 = 0
                actual_speed3 = 0
            else:
                actual_speed1 = max(0, normalized/32768)
                if i == 1:
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

GPIO.cleanup()
