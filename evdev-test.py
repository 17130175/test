#! /usr/bin/python3

#import evdev
from evdev import InputDevice, categorize, ecodes

#creates object 'gamepad' to store the data
#you can call it whatever you like
gamepad = InputDevice('/dev/input/event2')

#prints out device info at start
print(gamepad)

#evdev takes care of polling the controller in a loop
for event in gamepad.read_loop():
    print(categorize(event))
    if event.type == ecodes.EV_ABS:
        if event.code == ecodes.ABS_Y:
            print(event.value)
        if event.code == ecodes.ABS_RY:
            print(event.value)
