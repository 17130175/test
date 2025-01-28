from pi74HC595 import pi74HC595
import RPi.GPIO as gpio
import time
import cv2
import numpy as np
import cv2.aruco as aruco
import math

import smtplib                                              # Импортируем библиотеку по работе с SMTP
import os                                                   # Функции для работы с операционной системой, не зависящие от используемой операционной системы

# Добавляем необходимые подклассы - MIME-типы
import mimetypes                                            # Импорт класса для обработки неизвестных MIME-типов, базирующихся на расширении файла
from email import encoders                                  # Импортируем энкодер
from email.mime.base import MIMEBase                        # Общий тип
from email.mime.text import MIMEText                        # Текст/HTML
from email.mime.image import MIMEImage                      # Изображения
from email.mime.audio import MIMEAudio                      # Аудио
from email.mime.multipart import MIMEMultipart              # Многокомпонентный объект

def isRotationMatrix(R):
    Rt = np.transpose(R)
    iden = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - iden)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])

def servo_drive(angle, prev):
    if angle == prev:
        return prev
    pwm = gpio.PWM(servo, 50)
    pwm.start(8)
    dutyCycle = angle / 18 + 3
    pwm.ChangeDutyCycle(dutyCycle)
    time.sleep(0.3)
    pwm.stop()
    prev = angle
    return prev

def detect_green(image, find_cop):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    low = np.array([0, 0, 100])
    up = np.array([130, 90, 160])
    mask = cv2.inRange(hsv_image, low, up)
        
    #res = cv2.bitwise_and(image, image, mask=mask)
    area = np.sum(mask == 255)
    total = mask.shape[0] * mask.shape[1]
    green_perc = (area / total) * 100
    find_cop = False
    if green_perc >= 80:
        find_cop = True
        
    return mask, find_cop

def move(x0,y0,x1,y1, shift):
    s = math.sqrt((x1-x0)**2+(y1-y0)**2)
    v = 750 #mm
    t = s/v
    dx = x1 - x0
    dy = y1 - y0
    angle = math.atan2(dx, dy)
    x0 = x0+t/10*v*math.cos(angle)
    y0 = y0+t/10*v*math.sin(angle)
    shift.set_by_list([0,0,1,0,0,1,1,1])
    time.sleep(t/10)
    shift.set_by_list([0,0,0,0,0,0,0,0])
    time.sleep(1)
    return x0,y0

def set_angle(x0,y0,x1,y1, r_angle):
    dx = x1 - x0
    dy = y1 - y0
    angle = math.degrees(math.atan2(dx, dy)) + 180
    if angle >= 180:
        angle = angle - 180
        dr = 0
    else:
        angle = 180 - angle
        dr = 1
    dr_t = 1.35 * angle / 360
    if angle == r_angle:
        dr_t = 0
        dr = 2
    return dr_t, dr, angle

def send_email(addr_to, msg_subj, msg_text, files):
    addr_from = "fanur-bayazitov@mail.ru"                         # Отправитель
    password  = "J5fBJrSacwqTiQMxN93b"                                  # Пароль

    msg = MIMEMultipart()                                   # Создаем сообщение
    msg['From']    = 'fanur-bayazitov@mail.ru'                              # Адресат
    msg['To']      = 'fanur-bayazitov@mail.ru'                                # Получатель
    msg['Subject'] = 'Фото'                               # Тема сообщения

    body = 'Проверка'                                        # Текст сообщения
    msg.attach(MIMEText(body, 'plain'))                     # Добавляем в сообщение текст

    process_attachement(msg, files)

    #======== Этот блок настраивается для каждого почтового провайдера отдельно ===============================================
    server = smtplib.SMTP_SSL('smtp.mail.ru', 465)    # Создаем объект SMTP
    #server.starttls()                                  # Начинаем шифрованный обмен по TLS
    server.login(addr_from, password)                   # Получаем доступ
    server.send_message(msg)                            # Отправляем сообщение
    server.quit()                                       # Выходим
    #==========================================================================================================================

def process_attachement(msg, files):                        # Функция по обработке списка, добавляемых к сообщению файлов
    for f in files:
        if os.path.isfile(f):                               # Если файл существует
            attach_file(msg,f)                              # Добавляем файл к сообщению
        elif os.path.exists(f):                             # Если путь не файл и существует, значит - папка
            dir = os.listdir(f)                             # Получаем список файлов в папке
            for file in dir:                                # Перебираем все файлы и...
                attach_file(msg,f+"/"+file)                 # ...добавляем каждый файл к сообщению

def attach_file(msg, filepath):                             # Функция по добавлению конкретного файла к сообщению
    filename = os.path.basename(filepath)                   # Получаем только имя файла
    ctype, encoding = mimetypes.guess_type(filepath)        # Определяем тип файла на основе его расширения
    if ctype is None or encoding is not None:               # Если тип файла не определяется
        ctype = 'application/octet-stream'                  # Будем использовать общий тип
    maintype, subtype = ctype.split('/', 1)                                     # После использования файл обязательно нужно закрыть
    if maintype == 'image':                               # Если изображение
        with open(filepath, 'rb') as fp:
            file = MIMEImage(fp.read(), _subtype=subtype)
            fp.close()                  
    file.add_header('Content-Disposition', 'attachment', filename=filename) # Добавляем заголовки
    msg.attach(file)                                        # Присоединяем файл к сообщению



# Использование функции send_email()
addr_to   = "fanur-bayazitov@mail.ru"            

m_size = 60
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

with open('camera_cal.npy', 'rb') as f:
    mtx = np.load(f)
    dist = np.load(f)

cap = cv2.VideoCapture(0)

width = 640
height = 480
fps = 20

cap.set(2, width)
cap.set(4, height)
cap.set(5, fps)

gpio.setmode(gpio.BOARD)
gpio.setwarnings(False)

latch, data, clock, enable = 40, 36, 16, 32
motor = [38, 12, 22, 18]
servo = 7

gpio.setup(servo, gpio.OUT, initial=gpio.HIGH)
gpio.setup(enable, gpio.OUT, initial=gpio.LOW)
gpio.setup(motor, gpio.OUT, initial=gpio.HIGH)
shift = pi74HC595(data, latch, clock)

task = 0
prev_angle = 361
mark = [(0,0),(750,0),(1500,0),(0,750),(750,750),(1500,750),(0,1500),(750,1500),(1500,1500)]
pos_x, pos_y = 0, 0
prev_time = time.time()
find_cop = False
k1,k2 = 1,0
k = 0
r_angle = 0
while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if task == 0:
        print('выберите действие')
        task = int(input())
        shift.set_by_list([0,0,0,0,0,0,0,0])
    if task == 1:
        angle = 10
        prev_angle = servo_drive(angle, prev_angle)
        if k == 0:
            print(pos_x, pos_y)
            print('введите координаты')
            target_x = int(input())
            target_y = int(input())
            k = 1   
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, reject = aruco.detectMarkers(gray, aruco_dict, mtx, dist)
        
        if ids is not None:
            if ids[0] > 8 or ids[0] < 0:
                continue
            aruco.drawDetectedMarkers(frame, corners)
            rvec_all, tvec_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, m_size, mtx, dist)
            rvec = rvec_all[0][0]
            tvec = tvec_all[0][0]
            aruco.drawAxis(frame, mtx, dist, rvec, tvec, 100)
            rvec_flip = rvec * -1
            tvec_flip = tvec * -1
            rt_mtx, jac = cv2.Rodrigues(rvec_flip)
            rlw_tvec = np.dot(rt_mtx, tvec_flip)
            pitch, roll, yaw = rotationMatrixToEulerAngles(rt_mtx)
            tvec_str = "x=%4.0f y=%4.0f dir=%4.0f"%(mark[ids[0][0]][0]+rlw_tvec[0], mark[ids[0][0]][1]+rlw_tvec[1], math.degrees(yaw))
            cv2.putText(frame, tvec_str, (20,460), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_AA)
            pos_x, pos_y = mark[ids[0][0]][0]+rlw_tvec[0], mark[ids[0][0]][1]+rlw_tvec[1]
    
        cv2.imshow('frame', frame)
        if abs(pos_x - target_x) <= target_x*0.1 and abs(pos_y - target_y) <= target_y*0.1:
            task = 0
            k = 0
        dr_t, dr, r_angle = set_angle(pos_x, pos_y, target_x, target_y, r_angle)    
        if dr_t == 0:
            shift.set_by_list([0,0,0,0,0,0,0,0])
            print(2)
        elif dr == 1:
            print(1)
            shift.set_by_list([0,1,1,0,1,0,1,0])
            time.sleep(dr_t)
        elif dr == 0:
            print(0)
            shift.set_by_list([1,0,0,1,0,1,0,1])
            time.sleep(dr_t)
       
        pos_x, pos_y = move(pos_x, pos_y, target_x, target_y, shift)
    if task == 2:
        angle = 30
        prev_angle = servo_drive(angle, prev_angle) # Получатель
        cv2.imwrite('cam.png', frame)  
        files = ['drive.py']  # Если нужно отправить все файлы из заданной папки, нужно указать её
        send_email(addr_to, "Тема сообщения", "Текст сообщения", files)
        task = 0
    if task == 3:
        angle = 90
        shift.set_by_list([0,0,0,0,0,0,0,0])
        time.sleep(1)
        prev_angle = servo_drive(angle, prev_angle)
        #frame = frame[340:440, 320:480]
        green_obj, find_cop = detect_green(frame, find_cop)
        if find_cop:
            task = 4
            print('yes')
        else:
            shift.set_by_list([0,0,1,0,0,0,1,0])
            time.sleep(k1)
            if k2 >= 0.005:
                shift.set_by_list([0,0,1,0,0,1,1,1])
                time.sleep(k2)
            if k1 != 0.95:
                k1 -= 0.005
                k2 += 0.005
        cv2.imshow('green', green_obj)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
gpio.cleanup()
