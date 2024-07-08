import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class DashboardRecognition:
    def __init__(self, min_area=50 * 50):
        self.min_area = min_area
        self.image_width = 0
        self.image_height = 0
        self.bbox = None
        self.dashboard_status = None

    def detect(self, image, bbox=None):
        if bbox is None:
            bias = np.array([0, 0]).astype(np.int32)  # hand landmarks bias to left-top
        else:
            bias = bbox[0]  # hand landmarks bias to left-top
            # crop image
            image = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], :]
        self.image_height, self.image_width = image.shape[:2]
        self.bbox = None
        # 转换为灰度图像
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 使用高斯滤波平滑图像
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        # 使用霍夫变换检测圆形
        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100)
        # 如果检测到圆形，绘制边界
        if circles is not None:
            # 将圆形坐标转换为整数
            circles = np.round(circles[0, :]).astype("int")
            # 定义一个阈值，表示两个圆心之间的最大距离，用于判断是否合并
            threshold = 10
            # 定义一个列表，用于存储合并后的圆形
            merged_circles = []
            # 遍历每个圆形
            for (x1, y1, r1) in circles:
                # 定义一个标志，表示当前的圆形是否已经被合并过
                merged = False
                # 遍历已经合并过的圆形列表
                for i in range(len(merged_circles)):
                    # 获取已经合并过的圆形的坐标和半径
                    (x2, y2, r2) = merged_circles[i]
                    # 计算两个圆心之间的距离
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # 如果距离小于阈值，说明两个圆形可以合并
                    if distance < threshold:
                        # 将当前的圆形和已经合并过的圆形进行平均，得到新的圆形
                        if r1 >= r2:
                            merged_circles[i] = (x1, y1, r1)
                        else:
                            merged_circles[i] = (x2, y2, r2)
                        # 设置标志为True，表示当前的圆形已经被合并过
                        merged = True
                        # 跳出循环，不再遍历其他已经合并过的圆形
                        break
                # 如果当前的圆形没有被合并过，就将它添加到已经合并过的圆形列表中
                if not merged:
                    merged_circles.append((x1, y1, r1))
            # 遍历已经合并过的圆形列表，找到最大的圆，作为检测结果
            biggest = None
            biggest_r = -1
            for (x, y, r) in merged_circles:
                if r > biggest_r:
                    biggest = (x, y, r)
                    biggest_r = r
            # 将圆转换成bbox
            if biggest is not None:
                self.bbox = np.array([[biggest[0] - biggest[2], biggest[1] - biggest[2]],
                                      [biggest[0] + biggest[2], biggest[1] + biggest[2]]] + bias, np.int32)
                return self.__refine_bbox(self.bbox)
        return None

    def __refine_bbox(self, bbox):
        # refine bbox
        bbox[:, 0] = np.clip(bbox[:, 0], 0, self.image_width)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, self.image_height)
        w, h = bbox[1] - bbox[0]
        if w <= 0 or h <= 0 or w * h <= self.min_area:
            return None
        else:
            return bbox

    def resize_image(self, image, scale=0.6):
        # 获取图像的高度和宽度
        height, width = image.shape[:2]

        # 计算图像的中心点
        center_x, center_y = width // 2, height // 2

        # 计算缩放后的宽度和高度
        new_width, new_height = int(width * scale), int(height * scale)

        # 计算裁剪框的左上角和右下角坐标
        left = max(center_x - new_width // 2, 0)
        right = min(center_x + new_width // 2, width)
        top = max(center_y - new_height // 2, 0)
        bottom = min(center_y + new_height // 2, height)

        # 裁剪图像
        cropped_image = image[top:bottom, left:right]

        return cropped_image

    def get_status(self, image):
        self.dashboard_status = None
        if image is not None:
            # 取完整圆的60%部分
            # center = np.mean(self.bbox, axis=0)
            # scale_bbox = center + (self.bbox - center) * 0.5
            # scale_bbox = scale_bbox.astype(np.int32)
            # # 图片转换为灰度图
            # image = image[scale_bbox[0][1]:scale_bbox[1][1], scale_bbox[0][0]:scale_bbox[1][0], :]
            image = self.resize_image(image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 二值化，将黑色区域设为255，其他区域设为0
            _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
            # 寻找轮廓
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # 找到面积最大的轮廓
            max_area = 0
            max_contour = None
            for c in contours:
                area = cv.contourArea(c)
                if area > max_area:
                    max_area = area
                    max_contour = c

            # 如果找到了最大轮廓，绘制它并计算它的角度
            if max_contour is not None:
                # cv.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
                rect = cv.minAreaRect(max_contour)
                # 得到指针轮廓的最佳拟合直线，并获取其斜率
                vx, vy, x0, y0 = cv.fitLine(max_contour, cv.DIST_L2, 0, 0.01, 0.01)
                slope = vy / vx
                # 将斜率转换为角度，这个角度是指针相对于水平线的夹角
                angle = np.arctan(slope) * 180 / np.pi
                # 将这个角度转换为相对于竖直线的夹角，可以根据指针的方向进行调整，这里假设指针是顺时针旋转的
                angle = 90 + angle[0]
                # 如果指针中心的x值大于图片的一半，则角度+180。指针的角度范围是 [52, 308]，计算错误需要校准
                if rect[0][0] > image.shape[1] // 2:
                    angle += 180
                if angle < 45:
                    angle += 180
                if angle > 315:
                    angle -= 180
                score = (angle - 52) / 256  # 0-1指示数字，由256度划分

                # print(angle, score)

                if 0 <= score <= 0.3:
                    self.dashboard_status = "偏低"
                elif 0.3 < score <= 0.7:
                    self.dashboard_status = "正常"
                elif 0.7 < score <= 1.0:
                    self.dashboard_status = "偏高"
        else:
            self.dashboard_status = None
        return self.dashboard_status

    def cv2ImgAddText(self, img, text, left, top, textColor=(255, 0, 0), textSize=30):

        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型

            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(img)

            fontText = ImageFont.truetype(

                "font/simkai.ttf", textSize, encoding="utf-8")

            draw.text((left, top), text, textColor, font=fontText)

            return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    def visualize(self, image, thickness=1):
        if self.bbox is not None:
            # Draw bounding box on original image (in color)
            cv.rectangle(image, self.bbox[0], self.bbox[1], (0, 255, 0), 2)
            if self.dashboard_status is not None:
                image = self.cv2ImgAddText(image, self.dashboard_status, self.bbox[0][0], self.bbox[0][1] + -29 * thickness, )
                # cv.putText(image, self.dashboard_status, (self.bbox[0][0], self.bbox[0][1] + 22 * thickness),
                #            cv.FONT_HERSHEY_SIMPLEX, thickness, (0, 0, 255))
        return image

    def detect_and_match_features(self, image, template):
        # 转换为灰度图像
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        # SIFT特征检测器
        sift = cv.SIFT_create()

        # 检测特征点和描述符
        keypoints1, descriptors1 = sift.detectAndCompute(gray_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_template, None)

        # FLANN匹配器参数
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        # 进行特征点匹配
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # 绘制匹配结果
        matched_image = cv.drawMatches(image, keypoints1, template, keypoints2, good_matches, None,
                                        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return keypoints1, keypoints2, good_matches, matched_image

    def calculate_rotation_angle(self, keypoints1, keypoints2, matches):
        # 提取匹配点的坐标
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 计算变换矩阵
        M, mask = cv.estimateAffinePartial2D(points1, points2)

        # 打印变换矩阵用于调试
        # print("Transform matrix: ", M)

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

    def rotate_image(self, image, angle, center):
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated

