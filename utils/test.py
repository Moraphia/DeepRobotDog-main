import cv2
import numpy as np
from utils.DashboardRecognition import DashboardRecognition

def detect_and_match_features(image, template):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_template, None)

    # FLANN匹配器参数
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行特征点匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    matched_image = cv2.drawMatches(image, keypoints1, template, keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return keypoints1, keypoints2, good_matches, matched_image

def calculate_rotation_angle(keypoints1, keypoints2, matches):
    # 提取匹配点的坐标
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算变换矩阵
    M, mask = cv2.estimateAffinePartial2D(points1, points2)

    # 打印变换矩阵用于调试
    print("Transform matrix: ", M)

    # 计算旋转角度
    if isinstance(M, type(None)):
        return
    angle = -np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi

    # 确保角度在 -180 到 180 度之间
    if angle < -180:
        angle += 360
    elif angle > 180:
        angle -= 360

    return angle

def rotate_image(image, angle, center):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated

# 读取模板图像
template = cv2.imread(r'dashboard.png')
print(template)
# 打开摄像头
cap = cv2.VideoCapture(0)
dash = DashboardRecognition()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测仪表盘并获取边界框
    dash.detect(frame)
    if dash.bbox is not None:
        frame = frame[dash.bbox[0][1]:dash.bbox[1][1], dash.bbox[0][0]:dash.bbox[1][0], :]

        cv2.imshow('Image', frame)
        keypoints1, keypoints2, matches, matched_image = detect_and_match_features(frame, template)

        if matches:
            try:
                angle = -1 * calculate_rotation_angle(keypoints1, keypoints2, matches)
            except Exception as e:
                print(e)
                continue

            print("Calculated rotation angle: ", angle)

            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            corrected_image = rotate_image(frame, angle, center)

            cv2.imshow('Corrected Image', corrected_image)
            cv2.imshow('Matched Image', matched_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
