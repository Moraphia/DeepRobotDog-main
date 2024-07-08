import time
import struct
import cv2 as cv
import numpy as np
import threading

from utils.DashboardRecognition import DashboardRecognition
from utils.Controller import Controller

Develop_Mode = True  # True means use computer camera. False means use dog camera
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

global frame
global number
global is_update_number

if __name__ == '__main__':
    dashboard_detector = DashboardRecognition()
    controller = Controller(server_address)

    if Develop_Mode:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        stop_heartbeat = False


        # start to exchange heartbeat pack
        def heart_exchange(con):
            pack = struct.pack('<3i', 0x21040001, 0, 0)
            while True:
                if stop_heartbeat:
                    return
                con.send(pack)
                time.sleep(0.25)  # 4Hz


        heart_exchange_thread = threading.Thread(target=heart_exchange, args=(controller,))
        heart_exchange_thread.start()

        # stand up
        print("Wait 10 seconds and stand up......")
        pack = struct.pack('<3i', 0x21010202, 0, 0)
        controller.send(pack)
        time.sleep(5)
        controller.send(pack)
        time.sleep(5)
        controller.send(pack)
        print("Dog should stand up, otherwise press 'ctrl + c' and re-run the demo")

    # try to use CUDA
    if cv.cuda.getCudaEnabledDeviceCount() != 0:
        backend = cv.dnn.DNN_BACKEND_CUDA
        target = cv.dnn.DNN_TARGET_CUDA
    else:
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        print('CUDA is not set, will fall back to CPU.')

    status_buffer = [None] * 3
    template = cv.imread(r'doc_static/dashboard.png')
    print(template)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        dashboard_detector.detect(frame)
        corrected_image = None
        if dashboard_detector.bbox is not None:

            frame2 = frame[dashboard_detector.bbox[0][1]:dashboard_detector.bbox[1][1],
                     dashboard_detector.bbox[0][0]:dashboard_detector.bbox[1][0], :]

            cv.imshow('Image', frame2)
            keypoints1, keypoints2, matches, matched_image = dashboard_detector.detect_and_match_features(frame2,
                                                                                                          template)

            if matches:
                # 计算旋转角度

                try:
                    angle = -1 * dashboard_detector.calculate_rotation_angle(keypoints1, keypoints2, matches)
                except Exception as e:
                    print(e)
                    continue

                # 打印角度用于调试
                # print("Calculated rotation angle: ", angle)

                # 获取图像中心
                center = (frame2.shape[1] // 2, frame2.shape[0] // 2)

                # 校正图像
                corrected_image = dashboard_detector.rotate_image(frame2, angle, center)

                # 显示校正后的图像
                cv.imshow('Corrected Image', corrected_image)

                # 显示匹配结果
                cv.imshow('Matched Image', matched_image)

        bbox = dashboard_detector.detect(frame)
        frame = dashboard_detector.visualize(frame)
        status = dashboard_detector.get_status(corrected_image)

        status_buffer.insert(0, status)
        status_buffer.pop()
        if status is not None and all(s == status_buffer[0] for s in status_buffer):
            print("当前仪表盘压力值为 {}".format(status))

        cv.imshow("Danger Sign Recognition", frame)
        k = cv.waitKey(1)
        if k == 113 or k == 81:  # q or Q to quit
            print("Demo is quiting......")
            if not Develop_Mode:
                controller.drive_dog("squat")
            cap.release()
            cv.destroyWindow("Danger Sign Recognition")
            stop_heartbeat = True
            is_update_number = False
            break
