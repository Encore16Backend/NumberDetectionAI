import sqlite3
from flask import Flask, render_template, Response
from flask import request
import cv2
import HandTrackingModule as htm
import numpy as np
import os
from keras.models import load_model

app = Flask(__name__)

model = load_model('.\\model\\mnist_model.h5')

db_path = os.path.dirname(__file__) + '\database'
db = os.path.join(db_path, 'dashboard.sqlite')

mousePaint = True
handPaint = False

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
imgText = np.zeros((720, 1280, 3), np.uint8)

POINT = 0
SUGGESTION = str(np.random.randint(low=10 ** POINT, high=10 ** (POINT + 1)))
predict = False
PredictText = ""

point_text = "Point: " + str(POINT)
suggestion_text = "SUGGESTION: " + str(SUGGESTION)
cv2.putText(imgText, suggestion_text, (10, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(imgText, point_text, (580, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
# cv2.imshow('test1', imgCanvas)
# cv2.imshow('test2', imgText)
# cv2.waitKey()
# cv2.destroyAllWindows()

# global img

def get_frames():
    global imgCanvas, POINT, SUGGESTION, predict, PredictText, imgText
    brushThickness = 15
    eraserThickness = 50

    folderPath = 'image'
    brushList = os.listdir(folderPath)
    overlayList = []
    for imPath in brushList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

    header = overlayList[0]
    drawColor = (255, 0, 255)


    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = htm.handDetector(detectionCon=0.85)
    xp, yp = 0, 0

    while True:
        # import image
        success, img = cap.read()

        # find hand landmarks
        img = cv2.flip(img, 1)

        #########################################################
        ##################### HAND PAINTING #####################
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # print(lmList)

            # tip of index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # check which fingures are up
            fingers = detector.fingersUp()
            # print(fingers)

            # if selection mode - two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # print("Selection Mode")
                if y1 < 125:
                    # checking for click
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 100)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # if drawing mode - index finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                # print("Drawing Mode")

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    # cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    # cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1
                if not predict:
                    predict = True
            elif sum(fingers) == 0 and predict:
                # print("predict")
                predict_num()
                predict = False
                xp, yp = 0, 0
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        img[imgText[:, :, :] == 255] = 0
        # img = cv2.bitwise_or(img, ~imgText)

        # setting the header image
        img[0:125, 0:1280] = header
        # cv2.imshow("Canvas", imgCanvas)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def predict_num():
    global imgCanvas, SUGGESTION, POINT, PredictText, imgText
    imgCanvas_gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(imgCanvas_gray, 15)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    predict_list = []
    if len(contours) >= 1:
        contour = sorted(cv2.boundingRect(c) for c in contours)
        # for cnt in contours:
        for x, y, w, h in contour:
            # x, y, w, h = cv2.boundingRect(cnt)
            if h < 120:
                continue
            digit = thresh[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            # padding digit img with 5 pixels of black color(zeros)
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            # cv2.imshow("digit" + str(count), padded_digit)
            # count += 1
            digit = np.array(padded_digit)
            digit = digit.flatten()
            digit = digit.reshape(digit.shape[0], 1)
            digit = digit.reshape(1, 28, 28, 1)
            # digit /= 255.0
            pred = model.predict(digit)
            predict_list.append(pred.argmax())
    if predict_list:
        # print("예측 숫자:", *predict_list)
        PredictText = ''.join(map(str, predict_list))
        if str(SUGGESTION) == PredictText:
            POINT += 1
            SUGGESTION = str(np.random.randint(low=10 ** POINT, high=10 ** (POINT + 1)))

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    imgText = np.zeros((720, 1280, 3), np.uint8)

    point_text = "Point: " + str(POINT)
    suggestion_text = "SUGGESTION: " + str(SUGGESTION)

    cv2.putText(imgText, str(PredictText), (10, 500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(imgText, suggestion_text, (10, 160), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.putText(imgText, point_text, (580, 160), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

def select_dashboard():
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("SELECT * FROM dashboard_db")
    dashboard = c.fetchall()
    conn.commit()
    conn.close()
    return dashboard

def insert_dashboard(nickname, score):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("INSERT INTO dashboard_db (nickname, score) VALUES (?, ?)", (nickname, score))
    conn.commit()
    conn.close()

@app.route('/')
def hello_world():  # put application's code here
    dashboard = select_dashboard()
    return render_template('index.html', dashboard=dashboard)

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mouseEvent', methods=['POST'])
def mouse_event():
    global mousePaint, handPaint
    if request.method == 'POST':
        if not mousePaint:
            mousePaint = True
            handPaint = False
        else:
            handPaint = True
            mousePaint = False
        return 'SUCCESS'

@app.route('/predictEvent', methods=['POST'])
def predict_event():
    if request.method == 'POST':
        predict_num()
        return 'SUCCESS'


if __name__ == '__main__':
    app.run(debug=True)