#! /usr/bin/python3

from evdev import InputDevice, categorize, ecodes

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

def control_pwm1():
    global actual_speed1
    while control_pwm_enabled:
        if actual_speed1 > 0:
            GPIO.output(11, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed1,0) )
            
            GPIO.output(11, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed1),0) )
        else:
            time.sleep(motor_pwm_period)

def control_pwm2():
    global actual_speed2
    while control_pwm_enabled:
        if actual_speed2 > 0:
            GPIO.output(13, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed2,0) )
        
            GPIO.output(13, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed2),0) )
        else:
            time.sleep(motor_pwm_period)

def control_pwm3():
    global actual_speed3
    while control_pwm_enabled:
        if actual_speed3 > 0:
            GPIO.output(15, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed3,0) )
        
            GPIO.output(15, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed3),0) )
        else:
            time.sleep(motor_pwm_period)

def control_pwm4():
    global actual_speed4
    while control_pwm_enabled:
        if actual_speed4 > 0:
            GPIO.output(19, GPIO.HIGH)
            time.sleep(max(motor_pwm_period * actual_speed4,0) )
        
            GPIO.output(19, GPIO.LOW)
            time.sleep(max(motor_pwm_period * (1 - actual_speed4),0) )
        else:
            time.sleep(motor_pwm_period)


control_pwm_thread1 = threading.Thread(target=control_pwm1, args=())
control_pwm_thread2 = threading.Thread(target=control_pwm2, args=())
control_pwm_thread3 = threading.Thread(target=control_pwm3, args=())
control_pwm_thread4 = threading.Thread(target=control_pwm4, args=())


control_pwm_enabled = True
#control_pwm_enabled = False
if control_pwm_enabled:
    control_pwm_thread1.start()
    control_pwm_thread2.start()
    control_pwm_thread3.start()
    control_pwm_thread4.start()
    
    
gamepad = InputDevice('/dev/input/event2')

#evdev takes care of polling the controller in a loop
for event in gamepad.read_loop():
    #print(categorize(event))
    if event.type == ecodes.EV_KEY:
        if event.code == ecodes.BTN_MODE:
            if event.value == 0:
                break
    if event.type == ecodes.EV_ABS:
        if event.code == ecodes.ABS_Y:
            print(event.value)
            if event.value == -1:
                actual_speed3 = 0
                actual_speed4 = 0
            else:
                actual_speed3 = max(0, -event.value/32768)
                actual_speed4 = max(0,  event.value/32768) 
        if event.code == ecodes.ABS_RY:
            print(event.value)
            if event.value == -1:
                actual_speed1 = 0
                actual_speed2 = 0
            else:
                actual_speed1 = max(0,  event.value/32768)
                actual_speed2 = max(0, -event.value/32768) 


if control_pwm_enabled:
    control_pwm_enabled = False
    control_pwm_thread1.join()
    control_pwm_thread2.join()
    control_pwm_thread3.join()
    control_pwm_thread4.join()

GPIO.cleanup()
