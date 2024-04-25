from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

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
    camera = cv2.VideoCapture(__gstreamer_pipeline(camera_id=0, flip_method=2), cv2.CAP_GSTREAMER)
    x,y = 0, 0
    blue = np.uint8([[[255, 0, 0]]])
    hsvBlue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    L_limit = np.array([110, 100, 100]) #hsvBlue[0][0][0] - 10, 100, 100
    U_limit = np.array([130, 255, 255]) #hsvBlue[0][0][0] + 10, 255, 255
    print(L_limit, U_limit)
    while True:
        success, frame = camera.read()
        height, width = frame.shape[:2]
        edge = 10
        dst = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b_mask=cv2.inRange(dst, L_limit, U_limit)
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
        if (x<(width/2-edge)) and x!=0:
            cv2.rectangle(frame, (x+50,0), (width,height), (0,255,0), -1)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
