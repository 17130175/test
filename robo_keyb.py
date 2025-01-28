#! /usr/bin/python3

import curses, threading

import RPi.GPIO as GPIO
import sys,time,random,math
from pi74HC595 import pi74HC595

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
        GPIO.output(11, GPIO.HIGH)
        time.sleep(max(motor_pwm_period * actual_speed1,0) )
        
        GPIO.output(11, GPIO.LOW)
        time.sleep(max(motor_pwm_period * (1 - actual_speed1),0) )

def control_pwm2():
    global actual_speed2
    while control_pwm_enabled:
        GPIO.output(13, GPIO.HIGH)
        time.sleep(max(motor_pwm_period * actual_speed2,0) )
        
        GPIO.output(13, GPIO.LOW)
        time.sleep(max(motor_pwm_period * (1 - actual_speed2),0) )

def control_pwm3():
    global actual_speed3
    while control_pwm_enabled:
        GPIO.output(15, GPIO.HIGH)
        time.sleep(max(motor_pwm_period * actual_speed3,0) )
        
        GPIO.output(15, GPIO.LOW)
        time.sleep(max(motor_pwm_period * (1 - actual_speed3),0) )

def control_pwm4():
    global actual_speed4
    while control_pwm_enabled:
        GPIO.output(19, GPIO.HIGH)
        time.sleep(max(motor_pwm_period * actual_speed4,0) )
        
        GPIO.output(19, GPIO.LOW)
        time.sleep(max(motor_pwm_period * (1 - actual_speed4),0) )


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

def main(stdscr):
	global actual_speed1
	global actual_speed2
	global actual_speed3
	global actual_speed4
	global max_speed
	while 1:
		stdscr.keypad(True)
		Key = stdscr.getch()
		if Key == curses.KEY_UP:
			actual_speed1 = 0
			actual_speed2 = max_speed 
			actual_speed3 = max_speed 
			actual_speed4 = 0
		elif Key == curses.KEY_DOWN:
			actual_speed1 = max_speed
			actual_speed2 = 0
			actual_speed3 = 0
			actual_speed4 = max_speed 
		elif Key == curses.KEY_LEFT:
			actual_speed1 = max_speed
			actual_speed2 = 0
			actual_speed3 = max_speed
			actual_speed4 = 0
		elif Key == curses.KEY_RIGHT:
			actual_speed1 = 0
			actual_speed2 = max_speed
			actual_speed3 = 0
			actual_speed4 = max_speed
		elif Key == ord('1'):
			max_speed = 0.25
		elif Key == ord('2'):
			max_speed = 0.5
		elif Key == ord('3'):
			max_speed = 0.75
		elif Key == ord('4'):
			max_speed = 1
		elif Key == ord(' '):
			actual_speed1 = 0
			actual_speed2 = 0
			actual_speed3 = 0
			actual_speed4 = 0
		elif Key == ord('q'):
			break

curses.wrapper(main)

if control_pwm_enabled:
    control_pwm_enabled = False
    control_pwm_thread1.join()
    control_pwm_thread2.join()
    control_pwm_thread3.join()
    control_pwm_thread4.join()

GPIO.cleanup()
