#! /usr/bin/python3

# Системные библиотеки
import sys,time,random,math,threading

# Библиотеки для связи с драйвером моторов
import RPi.GPIO as GPIO

# Настраиваем драйвер моторов
GPIO.setmode(GPIO.BOARD)

GPIO.setup((11, 13, 15, 19), GPIO.OUT, initial=GPIO.LOW)


# Библиотека для распознавания изображения
import cv2
import numpy as np

MODE_LINE = 0
MODE_ARUCO = 1
MODE_GAUGE = 2
MODE_BACK = 3
MODE_STRIFE_LEFT = 4
MODE_STRIFE_RIGHT = 5
MODE_RUSH1 = 6
MODE_RUSH2 = 7

#CAMERA_TEST = True
CAMERA_TEST = False

mode = MODE_LINE
return_mode = MODE_LINE

if len(sys.argv) > 1:
    if sys.argv[1] == '1':
    	CAMERA_TEST = True
    elif sys.argv[1] == '2':
        CAMERA_TEST = True
        mode = MODE_ARUCO

CAMERA_TEST2 = True
#CAMERA_TEST2 = False

mtx = np.array([[ 517.0985854  ,   0,      415.73676016],
 [   0,          494.13090972 , 334.26796893],
 [   0,            0,            1        ]])
dist = np.array([[-0.63071109,  0.53544269, -0.0045538,  -0.00638318, -0.22761004]])


# спец.камера
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3264,height=2464,framerate=21/1,format=NV12' ! nvvidconv ! videoconvert ! jpegenc ! filesink location=capture1.jpeg
#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3264,height=2464,framerate=21/1,format=NV12' ! nvvidconv ! video/x-raw, width=1280, height=960 ! videoconvert ! x264enc ! filesink location=capture1.mp4

#cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! appsink")
#cap = cv2.VideoCapture("nvarguscamerasrc ! nvvidconv ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, width=1280, height=960, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, width=320, height=240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

#cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=1848, format=(string)NV12, framerate=(fraction)28/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)


'''
if CAMERA_TEST:
    ret, frame = cap.read()
    while True:
        if ret:
            break
        ret, frame = cap.read()

    cv2.imwrite('CAMERAMAN.jpg', frame)
    exit()
'''

# обычная вебка
#cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
#cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080,format=UYVY,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink", cv2.CAP_GSTREAMER)

#ARUCO_ENABLE = False
ARUCO_ENABLE = True
if ARUCO_ENABLE:
    import cv2.aruco as aruco
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_parameters = aruco.DetectorParameters_create()

waitingTime = time.time()
strafe_start_time = time.time()
straef_delay = 0.3
strafe_speed = 0.55
random_counter = 0
preferred_direction = 'right'
#preferred_direction = 'back'

corner1 = 0
corner2 = 100
corner1size = 0
corner2size = 0

motor_pwm_period = 0.05
line_speed = 0.5
max_speed = line_speed 
actual_speed1 = 0
actual_speed2 = 0
actual_speed3 = 0
actual_speed4 = 0
smooth_factor = 1
direction_smooth = 0
direction_smooth_factor = 0.3

def actual_speed_set_smooth(a1, a2, a3, a4):
    global actual_speed1
    global actual_speed2
    global actual_speed3
    global actual_speed4
    global max_speed
    global smooth_factor 
    
    actual_speed1 += smooth_factor * (a1 * max_speed - actual_speed1)
    actual_speed2 += smooth_factor * (a2 * max_speed - actual_speed2)
    actual_speed3 += smooth_factor * (a3 * max_speed - actual_speed3)
    actual_speed4 += smooth_factor * (a4 * max_speed - actual_speed4)

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



def getDirection(gray, axis_y, reverseCycle=False):
    sum1 = 0
    sum2 = 0
    counter = 0
    
    iterator = range(gray.shape[1])
    if reverseCycle:
        iterator = reversed(iterator)
    
    for j in iterator:
        a = int(gray[axis_y,j] == 0)
        
        counter = counter * (1+a) + 1
        
        sum1 +=     a * counter
        sum2 += j * a * counter
        
        counter *= a

    direction = -2
    on_screen = 0
    if sum1 > 0 and sum2 > 0:
        on_screen = sum2 / sum1
        direction = on_screen / gray.shape[1] * 2 - 1
        
    return direction, on_screen
    
def getDirection2(gray, axis_y):
    direction1, on_screen1 = getDirection(gray, axis_y, False)
    direction2, on_screen2 = getDirection(gray, axis_y, True)
    return [(direction1+direction2)/2, (on_screen1+on_screen2)//2]
    #return [direction1, on_screen1, direction2, on_screen2]
    
def getDirection3(gray, axis_y, iterations, step):
    direction = 0
    on_screen = 0
    for i in range(iterations):
        direction1, on_screen1 = getDirection2(gray, axis_y+i*step)
        if direction1 == -2:
            return -2, 0
        direction += direction1
        on_screen += on_screen1
        
    return direction / iterations, on_screen // iterations


# intel gauge detector
def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calibrate_gauge(img):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # gray = cv2.medianBlur(gray, 5)

    # for testing, output gray image
    #cv2.imwrite('gauge-%s-bw.%s' %(gauge_number, file_type),gray)

    # detect circles
    # restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    # these are pixel values which correspond to the possible radii search range.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.08), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    if circles is None:
        return None, None, None, None, None, None, None, None
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)

    #draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    #for testing, output circles on image
    #cv2.imwrite('gauge-%s-circles.%s' % (gauge_number, file_type), img)


    #for calibration, plot lines from center going out at every 10 degrees and add marker
    #for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

    #cv2.imwrite('gauge-%s-calibration.%s' % (gauge_number, file_type), img)
    if CAMERA_TEST2:
        cv2.imwrite("CAMERAMAN3_CALIBRATION.png", img)

    #get user input on min, max, values, and units
    #print('gauge number: %s' %gauge_number)
    #min_angle = raw_input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    #max_angle = raw_input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    #min_value = raw_input('Min value: ') #usually zero
    #max_value = raw_input('Max value: ') #maximum reading of the gauge
    #units = raw_input('Enter units: ')

    #for testing purposes: hardcode and comment out raw_inputs above
    min_angle = 45
    max_angle = 320
    min_value = 0
    max_value = 200
    units = "PSI"

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r):

    #for testing purposes
    #img = cv2.imread('gauge-%s.%s' % (gauge_number, file_type))

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 175
    maxValue = 255

    # for testing purposes, found cv2.THRESH_BINARY_INV to perform the best
    # th, dst1 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY);
    # th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);
    # th, dst3 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TRUNC);
    # th, dst4 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TOZERO);
    # th, dst5 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_TOZERO_INV);
    # cv2.imwrite('gauge-%s-dst1.%s' % (gauge_number, file_type), dst1)
    # cv2.imwrite('gauge-%s-dst2.%s' % (gauge_number, file_type), dst2)
    # cv2.imwrite('gauge-%s-dst3.%s' % (gauge_number, file_type), dst3)
    # cv2.imwrite('gauge-%s-dst4.%s' % (gauge_number, file_type), dst4)
    # cv2.imwrite('gauge-%s-dst5.%s' % (gauge_number, file_type), dst5)

    # apply thresholding which helps for finding lines
    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);

    # found Hough Lines generally performs better without Canny / blurring, though there were a couple exceptions where it would only work with Canny / blurring
    #dst2 = cv2.medianBlur(dst2, 5)
    #dst2 = cv2.Canny(dst2, 50, 150)
    #dst2 = cv2.GaussianBlur(dst2, (5, 5), 0)

    # for testing, show image after thresholding
    #cv2.imwrite('gauge-%s-tempdst2.%s' % (gauge_number, file_type), dst2)
    if CAMERA_TEST2:
        cv2.imwrite("CAMERAMAN3_TMPDST2.png", dst2)

    # find lines
    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

    #for testing purposes, show all found lines
    # for i in range(0, len(lines)):
    #   for x1, y1, x2, y2 in lines[i]:
    #      cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #      cv2.imwrite('gauge-%s-lines-test.%s' %(gauge_number, file_type), img)

    # remove all lines outside a given radius
    final_line_list = []
    #print "radius: %s" %r

    diff1LowerBound = 0.15 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.25
    diff2LowerBound = 0.5 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1.0
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp
            # check if line is within an acceptable range
            if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                line_length = dist_2_pts(x1, y1, x2, y2)
                # add to final list
                final_line_list.append([x1, y1, x2, y2])

    #testing only, show all lines after filtering
    # for i in range(0,len(final_line_list)):
    #     x1 = final_line_list[i][0]
    #     y1 = final_line_list[i][1]
    #     x2 = final_line_list[i][2]
    #     y2 = final_line_list[i][3]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # assumes the first line is the best one
    if final_line_list is None:
        return -1
    if len(final_line_list) == 0:
        return -1
    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #for testing purposes, show the line overlayed on the original image
    #cv2.imwrite('gauge-1-test.jpg', img)
    #cv2.imwrite('gauge-%s-lines-2.%s' % (gauge_number, file_type), img)
    if CAMERA_TEST2:
        cv2.imwrite("CAMERAMAN3_LINES2.png", img)

    #find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    #np.rad2deg(res) #coverts to degrees

    # print x_angle
    # print y_angle
    # print res
    # print np.rad2deg(res)

    #these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  #in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  #in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  #in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  #in quadrant IV
        final_angle = 270 - res

    #print final_angle

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle


    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

counter = 0

# main cycle
print("Press CTRL+C to skip motor loop")
while True:
    try:
        ret, frame_src = cap.read()
        counter += 1

        #if counter % 3 != 0:
        #    continue

        frame = None
        frame2 = None
        if ARUCO_ENABLE:
            frame2 = cv2.resize(frame_src, (816, 616), interpolation=cv2.INTER_AREA)
            #frame2[:,:,0] = cv2.multiply(frame2[:,:,0], 1.5)
            #frame2[:,:,1] = cv2.multiply(frame2[:,:,1], 1.5)
            #frame2[:,:,2] = cv2.multiply(frame2[:,:,2], 1.5)
            frame2[:,:,0] = cv2.subtract(frame2[:,:,0], 24)
            frame2[:,:,1] = cv2.subtract(frame2[:,:,1], 24)
            frame2[:,:,2] = cv2.subtract(frame2[:,:,2], 24 )
            #frame2[:,:,0] *= 2
            #frame2[:,:,1] *= 2
            #frame2[:,:,2] *= 2
            frame = cv2.resize(frame2, (204, 154), interpolation=cv2.INTER_AREA)
            
            h, w, c = frame2.shape	
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
            frame2 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
            #x, y, w, h = roi
            #frame = frame[y:y+h, x:x+w]
        else:
            frame = cv2.resize(frame_src, (204, 154), interpolation=cv2.INTER_AREA)


        #frame = cv2.resize(frame, (320, 240))
        #frame = (frame/255.)*(mask/255)*255
        #frame = frame.astype(np.uint8)
        
        if CAMERA_TEST:
            cv2.imwrite('CAMERAMAN.jpg', frame)
    
        if mode == MODE_LINE:
            if ARUCO_ENABLE:
                corners, ids, rejected = aruco.detectMarkers(frame2, aruco_dict, parameters=aruco_parameters)

                #if ids is not None:
                #    if 4 in ids:
                #        mode = MODE_GAUGE
    
                if ids is not None:
                    for i in range(len(ids)):
                        print(ids[i])
                        if ids[i] == 7 or ids[i] == 8:
                            mode = MODE_ARUCO
                            break
                        if ids[i] == 4:
                            if (corners[i][0][2][0] - corners[i][0][0][0]) > frame.shape[1]*0.05:
                                mode = MODE_GAUGE
                                break
                        if ids[i] == 9 or ids[i] == 6:
                            mode = MODE_RUSH1
                            break

            # Обработка
            frame[:,:,0] //= 2
            frame[:,:,2] //= 2
            frame[:,:,0] += frame[:,:,2]
            
            cv2.multiply(frame[:,:,0], 3)
            
            frame = cv2.subtract(frame[:,:,1], frame[:,:,0])
            th, mask = cv2.threshold(frame, 64, 255, cv2.THRESH_BINARY)
            
            # Направление
            #y_line  = frame.shape[0] * 15//20
            y_line  = frame.shape[0] * 17//20
            y_line2 = frame.shape[0] * 19//20
            #direction, on_screen = getDirection2(frame, y_line)
            direction, on_screen = getDirection3(frame, y_line, 3, 1)

            '''
            direction2, on_screen2 = getDirection3(frame, y_line2, 4, 6)

            if abs(direction2) >   abs(direction) and direction2 != -2:
                direction += direction2
                direction /= 2

            if abs(direction2) > 2*abs(direction) and direction2 != -2:
                direction = direction2
            '''

            '''
            direction_smooth += direction_smooth_factor * (direction - direction_smooth)
            direction = direction_smooth 
            '''

            '''
            direction_smooth += 0.02 * (direction - direction_smooth)
            max_speed = (
                0.6 
                + 0.2 * int(abs(direction - direction_smooth) < 0.05)
                + 0.2 * int(abs(direction - direction_smooth) < 0.01)
            )
            '''
            

            if CAMERA_TEST:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.circle(frame, (int(on_screen), y_line), radius=10, color=(0, 0, 255), thickness=-1)
                #frame = cv2.circle(frame, (int(on_screen2), y_line2), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.imwrite('CAMERAMAN1.jpg', frame)
                break
        

            # Вперёд
            if abs(direction) < 0.2: # and abs(direction2) < 0.2:
                actual_speed_set_smooth(0, 1, 1, 0)
            # Назад
            elif direction == -2:
                # назад
                if preferred_direction == 'back':
                    actual_speed_set_smooth(1, 0, 0, 1)
                # влево
                if preferred_direction == 'left':
                    actual_speed_set_smooth(0, 1, 0, 1)
                # вправо
                if preferred_direction == 'right':
                    actual_speed_set_smooth(1, 0, 1, 0)
            # Влево
            elif direction < 0:
                if abs(direction) > 0.5:
                    actual_speed_set_smooth(0, 1, 0, 1)
                else:
                    actual_speed_set_smooth(0, 1, 0, 0)
            # Вправо
            elif direction > 0:
                if abs(direction) > 0.5	:
                    actual_speed_set_smooth(1, 0, 1, 0)
                else:
                    actual_speed_set_smooth(0, 0, 1, 0)
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
        
        if mode == MODE_ARUCO:
            corners, ids, rejected = aruco.detectMarkers(frame2, aruco_dict, parameters=aruco_parameters)

            found = 0
            foundLeft = False
            foundRight = False
            #corner1 = 0
            #corner2 = frame2.shape[1]
            
            if ids is not None:
                for i in range(len(ids)):
                    corner = corners[i][0]
                    average = (corner[0][0] + corner[1][0]) / 2
                    size = abs(corner[0][0] - corner[1][0])

                    #print(ids[i], average, size, corners[i])

                    if ids[i] == 7:
                        corner1 = average
                        corner1size = size
                        found += 1
                        foundLeft = True
                    elif ids[i] == 8:
                        corner2 = average
                        corner2size = size
                        found += 1
                        foundRight = True

            if CAMERA_TEST:
                image = frame2.copy()
                aruco.drawDetectedMarkers(image, corners, ids)
                cv2.imwrite('CAMERAMAN2.png', image)
                break

            direction = 0
            if found > 0:
                target_distance = 0.4

                distance = abs(corner1 - corner2) / frame2.shape[1]
		
                
                if found > 1:
                    direction = (corner1 + corner2) / frame2.shape[1] - 1
                    if abs(direction) < 0.1:
                        if corner1size > 1.15 * corner2size:
                            max_speed = strafe_speed
                            return_mode = mode
                            mode = MODE_STRIFE_RIGHT
                            strafe_start_time = time.time()
                        elif corner2size > 1.15 * corner1size:
                            max_speed = strafe_speed
                            return_mode = mode
                            mode = MODE_STRIFE_LEFT 
                            strafe_start_time = time.time()
                        else:
                            if distance > 1.1 * target_distance:
                                max_speed = 0.4
                                direction = -2
                            elif target_distance > 1.15 * distance:
                                max_speed = 0.4
                                direction = (corner1 + corner2) / frame2.shape[1] - 1
                            else:
                                direction = -3
                    print(direction, distance, corner1size, corner2size)
                elif found == 1:
                    max_speed = 0.5
                    if foundLeft:
                        print("found left")
                        max_speed = 0.35
                        direction = 1
                    elif foundRight:
                        print("found right")
                        max_speed = 0.35
                        direction = -1
                    else:
                        direction = -3
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
                continue
        
            # Вперёд
            if abs(direction) < 0.1:
                actual_speed_set_smooth(0, 1, 1, 0)
            # Назад
            elif direction == -2:
                actual_speed_set_smooth(1, 0, 0, 1)
            # Стоп
            elif direction == -3:
                actual_speed_set_smooth(0, 0, 0, 0)
            # Влево
            elif direction < 0:
                actual_speed_set_smooth(0, 1, 0, 1)
            # Вправо
            elif direction > 0:
                actual_speed_set_smooth(1, 0, 1, 0)
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
        
        if mode == MODE_GAUGE:
            corners, ids, rejected = aruco.detectMarkers(frame_src, aruco_dict, parameters=aruco_parameters)

            found = False
            corner = None
            
            if ids is not None:
                for i in range(len(ids)):
                    if ids[i] == 4:
                        corner = corners[i][0]
                        found = True
                        break

            actual_speed_set_smooth(0, 0, 0, 0)

            if found:
                direction1 = corner[1] - corner[0]
                direction2 = corner[3] - corner[0]
                newcorner1 = corner[0] - 2.5*direction1 - direction2
                newcorner2 = corner[2] - direction1 + 0.7*direction2
                newcorner1[0] = max(newcorner1[0], 0)
                newcorner1[1] = max(newcorner1[1], 0)
                newcorner2[0] = min(newcorner2[0], frame_src.shape[1])
                newcorner2[1] = min(newcorner2[1], frame_src.shape[0])
                cropped = frame_src[int(newcorner1[1]):int(newcorner2[1]), int(newcorner1[0]):int(newcorner2[0])]
                cropped_contrast = cropped.copy()
                cropped = cropped.clip(0, 127)
                cropped *= 2
                #cropped = cropped.clip(0, 85)
                #cropped *= 3
                if CAMERA_TEST:
                    cv2.imwrite('CAMERAMAN3.png', cropped)

                min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(cropped_contrast)
                if min_angle is None:
                    min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(cropped)
                else:
                    cropped = cropped_contrast

                if min_angle is None:
                    print("Not found")
                
                    random_counter = (random_counter + 1) % 4
                    if random_counter == 0:
                        actual_speed_set_smooth(0, 1, 1, 0) # вперёд
                    if random_counter == 1:
                        actual_speed_set_smooth(1, 0, 0, 1) # назад
                    if random_counter == 2:
                        actual_speed_set_smooth(0, 1, 0, 1) # влево
                    if random_counter == 3:
                        actual_speed_set_smooth(1, 0, 1, 0) # вправо
                else:
                    val = get_current_value(cropped, min_angle, max_angle, min_value, max_value, x, y, r)
                    print("Current reading: %s %s" %(val, units))

                    mode = MODE_BACK
                    waitingTime = time.time() + 3

                    if preferred_direction == 'left':
                        preferred_direction = 'right'
                    if preferred_direction == 'right':
                        preferred_direction = 'left'

            if CAMERA_TEST:
                image = frame_src.copy()
                aruco.drawDetectedMarkers(image, corners, ids)
                cv2.imwrite('CAMERAMAN2.png', image)
                break
        
        if mode == MODE_BACK:
            if time.time() < waitingTime:
                actual_speed_set_smooth(0, 1, 0, 1)
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
                mode = MODE_LINE
        
        if mode == MODE_STRIFE_LEFT:
            if time.time() < straef_delay * 1 + strafe_start_time:
                actual_speed_set_smooth(0, 1, 0, 1)
            elif time.time() < straef_delay * 2 + strafe_start_time: 
                actual_speed_set_smooth(0, 1, 1, 0)
            elif time.time() < straef_delay * 3 + strafe_start_time: 
                actual_speed_set_smooth(1, 0, 1, 0)
            elif time.time() < straef_delay * 4 + strafe_start_time: 
                actual_speed_set_smooth(1, 0, 0, 1)
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
                mode = return_mode
        
        if mode == MODE_STRIFE_RIGHT:
            if time.time() < straef_delay * 1 + strafe_start_time:
                actual_speed_set_smooth(1, 0, 1, 0)
            elif time.time() < straef_delay * 2 + strafe_start_time: 
                actual_speed_set_smooth(0, 1, 1, 0)
            elif time.time() < straef_delay * 3 + strafe_start_time: 
                actual_speed_set_smooth(0, 1, 0, 1)
            elif time.time() < straef_delay * 4 + strafe_start_time: 
                actual_speed_set_smooth(1, 0, 0, 1)
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
                mode = return_mode
        
        if mode == MODE_RUSH1:
            corners, ids, rejected = aruco.detectMarkers(frame2, aruco_dict, parameters=aruco_parameters)

            found = 0
            foundLeft = False
            foundRight = False
            #corner1 = 0
            #corner2 = frame2.shape[1]
            
            if ids is not None:
                for i in range(len(ids)):
                    corner = corners[i][0]
                    average = (corner[0][0] + corner[1][0]) / 2
                    size = abs(corner[0][0] - corner[1][0])

                    #print(ids[i], average, size, corners[i])

                    if ids[i] == 9:
                        corner1 = average
                        corner1size = size
                        found += 1
                        foundLeft = True
                    elif ids[i] == 6:
                        corner2 = average
                        corner2size = size
                        found += 1
                        foundRight = True

            direction = 0
            if found > 0:
                target_distance = 0.25 	

                distance = abs(corner1 - corner2) / frame2.shape[1]
		
                
                if found > 1:
                    direction = (corner1 + corner2) / frame2.shape[1] - 1
                    if abs(direction) < 0.1:
                        if corner1size > 1.15 * corner2size:
                            max_speed = strafe_speed
                            return_mode = mode
                            mode = MODE_STRIFE_RIGHT
                            strafe_start_time = time.time()
                        elif corner2size > 1.15 * corner1size:
                            max_speed = strafe_speed
                            return_mode = mode
                            mode = MODE_STRIFE_LEFT 
                            strafe_start_time = time.time()
                        else:
                            if distance > 1.1 * target_distance:
                                max_speed = 0.4
                                direction = -2
                            elif target_distance > 1.15 * distance:
                                max_speed = 0.4
                                direction = (corner1 + corner2) / frame2.shape[1] - 1
                            else:
                                waitingTime = time.time() + 4
                                mode = MODE_RUSH2
                    print(direction, distance, corner1size, corner2size)
                elif found == 1:
                    max_speed = 0.5
                    if foundLeft:
                        print("found left")
                        max_speed = 0.35
                        actual_speed_set_smooth(0, 1, 0, 0)
                        continue
                        direction = 1
                    elif foundRight:
                        print("found right")
                        max_speed = 0.35
                        actual_speed_set_smooth(0, 0, 1, 0)
                        continue
                        direction = -1
                    else:
                        direction = -3
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
                continue
        
            # Вперёд
            if abs(direction) < 0.1:
                actual_speed_set_smooth(0, 1, 1, 0)
            # Назад
            elif direction == -2:
                actual_speed_set_smooth(1, 0, 0, 1)
            # Стоп
            elif direction == -3:
                actual_speed_set_smooth(0, 0, 0, 0)
            # Влево
            elif direction < 0:
                actual_speed_set_smooth(0, 1, 0, 1)
            # Вправо
            elif direction > 0:
                actual_speed_set_smooth(1, 0, 1, 0)
            else:
                actual_speed_set_smooth(0, 0, 0, 0)
        
        if mode == MODE_RUSH2:
            if time.time() < waitingTime:
                max_speed = 0.8
                actual_speed_set_smooth(0, 1, 1, 0)
            else:
                max_speed = line_speed
                actual_speed_set_smooth(0, 0, 0, 0)
                mode = MODE_LINE
        
            
    except KeyboardInterrupt:
        print("Interrupted")
        break
    
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
        break
    
# Выключаем драйвер моторов
if control_pwm_enabled:
    control_pwm_enabled = False
    control_pwm_thread1.join()
    control_pwm_thread2.join()
    control_pwm_thread3.join()
    control_pwm_thread4.join()
GPIO.cleanup()
