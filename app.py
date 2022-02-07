import sqlite3
from flask import Flask, render_template, Response
from flask import request
import cv2
import HandTrackingModule as htm
import numpy as np
import os
from keras.models import load_model
from pynput.mouse import Controller

app = Flask(__name__)

# start 변수들
name =''
start = True
end= False
imageIndex = 0

dir_path = os.path.dirname(__file__)

model_path = dir_path + '\model\mnist_model.h5'
model = load_model(model_path)

db_path = dir_path + '\database'
db = os.path.join(db_path, 'dashboard.sqlite')

img = np.zeros((720, 1280, 3), np.uint8)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
imgText = np.zeros((720, 1280, 3), np.uint8)
# lifeCanvas = np.zeros((720, 1280, 3), np.uint8)

LIFE_COUNT = 3
lifeList = cv2.imread('image/life'+str(LIFE_COUNT)+'.png')
lifeList = cv2.resize(lifeList, (240, 118))

POINT = 0
SUGGESTION = str(np.random.randint(low=10 ** POINT, high=10 ** (POINT + 1)))
predict = False
PredictText = ""

point_text = "Point: " + str(POINT)
suggestion_text = "SUGGESTION: " + str(SUGGESTION)
cv2.putText(imgText, suggestion_text, (10, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
cv2.putText(imgText, point_text, (580, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

detector = htm.handDetector(detectionCon=0.85)
mouse = Controller()
xp, yp = 0, 0

# 상단 브러쉬, 지우개
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

mode = True # Hand True, Mouse False
mouse = Controller()
width , height = 0,0
mouseState = False

def handPainting():
    global xp, yp, drawColor, header, img, imgCanvas
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
                if y1 > 125:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
            # if not predict:
            #     predict = True
    return

def mousePainting():
    global xp, yp, drawColor, header, width, height, mouseState, img, imgCanvas
    mx, my = mouse.position
    mx = int((mx-int(screenX))*1280/int(width))
    my = int((my-int(screenY)-110)*720/int(height))
    cv2.circle(img, (mx, my), 15, drawColor, cv2.FILLED)

    if my < 125:
        # checking for click
        if 250 < mx < 450:
            header = overlayList[0]
            drawColor = (255, 0, 255)
        elif 550 < mx < 750:
            header = overlayList[1]
            drawColor = (255, 0, 100)
        elif 800 < mx < 950:
            header = overlayList[2]
            drawColor = (0, 255, 0)
        elif 1050 < mx < 1200:
            header = overlayList[3]
            drawColor = (0, 0, 0)

    if mouseState: # 그리기 상태
        if xp == 0 and yp == 0:
            xp, yp = mx, my
        if drawColor == (0, 0, 0):
            # cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (mx, my), drawColor, eraserThickness)
        else:
            # cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            if my > 125:
                cv2.line(imgCanvas, (xp, yp), (mx, my), drawColor, brushThickness)
        xp, yp = mx, my
    return

def get_frames():
    global imgCanvas, start, img, header, imageIndex, img, lifeList, PredictText, point_text, suggestion_text

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        # import image
        if start:
            img = cv2.imread('./image/start' + str(imageIndex) + '.png')
            img = cv2.resize(img, (1280, 720))
            img = cv2.flip(img, 1)

            cv2.putText(img, "Your Nick Name", (360, 330), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 4, (255, 255, 255), 2)
            imageIndex = (imageIndex + 1) % 4
        elif end:
            if imageIndex > 1: imageIndex = 0

            img = cv2.imread('./image/end' + str(imageIndex) + '.png')
            img = cv2.resize(img, (1280, 720))
            img = cv2.flip(img, 1)
            cv2.putText(img, "Nick Name : " + name, (190, 350), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 4,
                        (255, 255, 255), 2)
            cv2.putText(img, "Score : {} POINT".format(POINT), (350, 450), cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 4,
                        (255, 255, 255), 2)
            cv2.putText(img, "Press 'Q' START", (270, 600), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)

            imageIndex = imageIndex + 1
        else:
            success, img = cap.read()
            img = cv2.resize(img,(1280,720))
            # find hand landmarks
            img = cv2.flip(img, 1)

            ##### HAND PAINTING #####
            handPainting() if mode else mousePainting()

            cv2.putText(imgText, str(PredictText), (10, 500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(imgText, point_text, (580, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv2.putText(imgText, suggestion_text, (10, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            ###### Display Img ######
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)
            img[imgText[:, :, :] == 255] = 0

            # setting the header image
            img[0:125, 0:1280] = header
            if LIFE_COUNT > 0:
                img[130:130 + 118, -240:] = cv2.bitwise_and(img[130:130 + 118, -240:], lifeList)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


def predict_num():
    global imgCanvas, SUGGESTION, POINT, PredictText, imgText, xp, yp, LIFE_COUNT, lifeList, end, img, point_text, suggestion_text

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
                PredictText = 'none'
            # 숫자 높이 120 이상일때 예측 시작
            else:
                digit = thresh[y:y + h, x:x + w]
                resized_digit = cv2.resize(digit, (18, 18))

                # padding digit img with 5 pixels of black color(zeros)
                padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

                digit = np.array(padded_digit)
                digit = digit.flatten()
                digit = digit.reshape(digit.shape[0], 1)
                digit = digit.reshape(1, 28, 28, 1)

                pred = model.predict(digit)
                predict_list.append(pred.argmax())

        if predict_list:
            # print("예측 숫자:", *predict_list)
            PredictText = ''.join(map(str, predict_list))
            if str(SUGGESTION) == PredictText:
                POINT += 1
                SUGGESTION = str(np.random.randint(low=10 ** POINT, high=10 ** (POINT + 1)))
            else:
                LIFE_COUNT -= 1
                if LIFE_COUNT == 0:
                    end = True
                else:
                    SUGGESTION = str(np.random.randint(low=10 ** POINT, high=10 ** (POINT + 1)))
                    lifeList = cv2.imread('image/life' + str(LIFE_COUNT) + '.png')
                    lifeList = cv2.resize(lifeList, (240, 118))
                # img[130:130 + 118, -240:] = cv2.bitwise_and(img[130:130 + 118, -240:], lifeList)
    # 아무것도 안쓰고 확인하면 예측 숫자 자리에 none
    if not predict_list:
        PredictText = 'none'
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    imgText = np.zeros((720, 1280, 3), np.uint8)

    point_text = "Point: " + str(POINT)
    suggestion_text = "SUGGESTION: " + str(SUGGESTION)


def select_dashboard():
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("SELECT nickname, score FROM dashboard_db ORDER BY score DESC LIMIT 10")
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
def main():  # put application's code here
    dashboard = select_dashboard()
    return render_template('index.html', dashboard=dashboard)

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/modeChange', methods=['POST'])
def mode_event():
    global mode, scaleX, scaleY, screenX, screenY, width, height, mouseState, xp, yp
    if request.method == 'POST':
        if mode: # mouse
            mode = False
            width, height = request.form.get('width'), request.form.get('height')
            screenX, screenY = request.form.get('screenX'), request.form.get('screenY')
        else:
            mode = True
        return 'HAND' if mode else 'MOUSE'

@app.route('/mouseState', methods=['POST'])
def mouse_state():
    global mouseState, xp, yp
    if request.method == 'POST':
        if not mouseState:
            xp, yp = 0, 0
            mouseState = True
        else:
            mouseState = False
        return 'SUCCESS'


@app.route('/predictEvent', methods=['POST'])
def predict_event():
    if request.method == 'POST':
        predict_num()
        return 'SUCCESS'

@app.route('/startEvent', methods=['POST'])
def start_event():
    global start,name, end
    # dashboard = select_dashboard()
    if request.method == 'POST':
        name = request.form.get('nickname')
        start = False
    return "SUCCES"

@app.route('/endEvent', methods=['POST'])
def end_Event():
    global start, end, POINT, name, LIFE_COUNT, lifeList, imgCanvas, imgText, point_text, suggestion_text, SUGGESTION, predict, PredictText
    if request.method == 'POST':
        start = True
        end = False
        insert_dashboard(name, POINT)

        imgText = np.zeros((720, 1280, 3), np.uint8)
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        POINT = 0
        SUGGESTION = str(np.random.randint(low=10 ** POINT, high=10 ** (POINT + 1)))
        predict = False
        PredictText = ""

        point_text = "Point: " + str(POINT)
        suggestion_text = "SUGGESTION: " + str(SUGGESTION)

        LIFE_COUNT = 3
        lifeList = cv2.imread('image/life' + str(LIFE_COUNT) + '.png')
        lifeList = cv2.resize(lifeList, (240, 118))

        return "SUCCES"

if __name__ == '__main__':
    app.run()#debug=True)