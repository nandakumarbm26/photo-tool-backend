import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
from ast import Str
from urllib.request import urlretrieve

import cv2
import dlib
import imutils
import mediapipe as mp
import numpy as np
import requests
import torch
from basicsr.utils import imwrite
from imutils import face_utils
from matplotlib.pyplot import axis
from PIL import Image
from scipy.spatial import distance as dist

# from gfpgan import GFPGANer


def rembgApi():
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(r'outeyeq.jpg', 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': '4sPAH5sEjbp7RuDLYK9DNpZh'},
    )
    if response.status_code == requests.codes.ok:
        with open('outeyeq.jpg', 'wb') as out:
            out.write(response.content)

        img=Image.open('outeyeq.jpg')
        img2=Image.open('white.jpg')
        img2=img2.resize(img.size,Image.ANTIALIAS)
        img2.paste(img,(0,0),mask=img)
        img2.save('outeyeq.jpg')
        return True
    else:
        print("Error:", response.status_code, response.text)
        return False

def getEyeQImage():
    t=subprocess.check_output('curl -H "X-API-KEY: 3584ff8a41351bdddafd96c7105d3f4f85f" -H "Content-Type: image/jpeg" -X PUT "https://api.perfectlyclear.io/v1/pfc" --upload-file in.jpg',shell=True,text=True)
    t=json.loads(t)
    t=subprocess.check_output('curl -H "X-API-KEY: 3584ff8a41351bdddafd96c7105d3f4f85f" -X GET "https://api.perfectlyclear.io/v1/pfc/'+t["imageKey"]+'"',shell=True,text=True)
    print(t)
    t=json.loads(t)
    urlretrieve(t['corrected_url'],'outeyeq.jpg')

def imageCheckFault(image):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose

  # mediapipe pose model
  pose =  mp_pose.Pose(
      static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True,
      min_detection_confidence=0.5)

  #convert image to RGB (just for input to model)
  image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  height,width,_ =image_input.shape
  # get results using mediapipe
  results =pose.process(image_input)

  if not results.pose_landmarks:
      return True
  else:
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  lsx=results.pose_landmarks.landmark[12].x*width
  lsy=results.pose_landmarks.landmark[12].y*height
  rsx=results.pose_landmarks.landmark[11].x*width
  rsy=results.pose_landmarks.landmark[11].y*height
  sholderLength=rsx-lsx
  sholderPercentatge=sholderLength/width
  if(sholderPercentatge>0.8):
    return True
  return False

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

def mouth_open(frame):
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	MOUTH_AR_THRESH = 0.79
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	(mStart, mEnd) = (49, 68)

	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rect = detector(gray, 0)

	# loop over the face detections

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
	shape = predictor(gray, rect[0])
	shape = face_utils.shape_to_np(shape)

	# extract the mouth coordinates, then use the
	# coordinates to compute the mouth aspect ratio
	mouth = shape[mStart:mEnd]

	mouthMAR = mouth_aspect_ratio(mouth)
	mar = mouthMAR
	# compute the convex hull for the mouth, then
	# visualize the mouth
	mouthHull = cv2.convexHull(mouth)

	cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
	cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# Draw text if mouth is open
	if mar > MOUTH_AR_THRESH:
		return True
	return False





def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_centers(img, landmarks):
    # 线性回归
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])

    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1)
    cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# 瞳距
    scale = desired_dist / dist # 缩放比例
    angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))

    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0) #高斯模糊

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) #y方向sobel边缘检测
    sobel_y = cv2.convertScaleAbs(sobel_y) #转换回uint8类型

    edgeness = sobel_y #边缘强度矩阵

    #Otsu二值化
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #计算特征长度
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)

    roi_1 = thresh[y:y+h, x:x+w] #提取ROI
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])

    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])#计算评价值
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])#计算评价值
    measure = measure_1*0.3 + measure_2*0.7

    # print(measure)

    #根据评价值和阈值的关系确定判别值
    if measure > 0.15:#阈值可调，经测试在0.15左右
        judge = True
    else:
        judge = False
    # print(judge)
    return judge






def specsdetection(img):
    #读取视频帧
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")


    #转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    rects = detector(gray, 1)

    # 对每个检测到的人脸进行操作
    for i, rect in enumerate(rects):
        # 得到坐标
        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face

        # 绘制边框，加文字标注
        cv2.rectangle(img, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # 检测并标注landmarks
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        # 线性回归
        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

        # 人脸对齐
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

        # 判断是否戴眼镜
        judge = judge_eyeglass(aligned_face)
    # print(judge)
    return judge


# def enhance(frame):
#     # ------------------------ set up background upsampler ------------------------
#     from basicsr.archs.rrdbnet_arch import RRDBNet
#     from realesrgan import RealESRGANer
#     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
#     bg_upsampler = RealESRGANer(
#         scale=2,
#         model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
#         model=model,
#         tile=0,
#         tile_pad=10,
#         pre_pad=0,
#         half=True)


#     # ------------------------ set up GFPGAN restorer ------------------------

#     arch = 'clean'
#     channel_multiplier = 2
#     model_name = 'GFPGANCleanv1-NoCE-C2'
#     # model_name = 'GFPGANv1.3'
#     # determine model paths
#     model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
#     # model_path = os.path.join('experiments\\pretrained_models', model_name + '.pth')
#     bg_upsampler = None
#     restorer = GFPGANer(
#         model_path=model_path,
#         upscale=2,
#         arch=arch,
#         channel_multiplier=channel_multiplier,
#         bg_upsampler=bg_upsampler)

#     # ------------------------ restore ------------------------

#     input_img = frame

#     # restore faces and background if necessary
#     cropped_faces, restored_faces, restored_img = restorer.enhance(
#         input_img, paste_back=True)
#     print(restored_img.shape)

#     imwrite(restored_img,'output2.png')

#     return restored_img


def face_alignment(img_or_path, face_index=0,border_color=(255,255,255)):
    img=img_or_path
    img =  cv2.copyMakeBorder(img, 100, 100, 100, 100, borderType=cv2.BORDER_CONSTANT, value=border_color[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    detections = detector(img, 1)
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # downlaod this file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    faces = dlib.full_object_detections()
    for det in detections:
        faces.append(sp(img, det))

    face_count = len(faces)
    if face_index >= face_count:
        raise ValueError("more than one face is detected")

    bb = [i.rect for i in faces]
    bb = [((i.left(), i.top()),
        (i.right(), i.bottom())) for i in bb]


    right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
    right_eyes = [[(i.x, i.y) for i in eye]
                for eye in right_eyes]
    right_points = []
    for eye in right_eyes:
        right_points.append((
            max(eye, key=lambda x: x[0])[0],
            max(eye, key=lambda x: x[1])[1],
            min(eye, key=lambda x: x[0])[0],
            min(eye, key=lambda x: x[1])[1]
        ))
    left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
    left_eyes = [[(i.x, i.y) for i in eye]
                for eye in left_eyes]
    left_points = []
    for eye in left_eyes:
        left_points.append((
            max(eye, key=lambda x: x[0])[0],
            max(eye, key=lambda x: x[1])[1],
            min(eye, key=lambda x: x[0])[0],
            min(eye, key=lambda x: x[1])[1]
        ))

    eye_1 = left_points[face_index]
    eye_2 = right_points[face_index]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1
    left_eye_center = (
        int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (
        int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=border_color[::-1])

    return rotated


def cropImage(img):
    img=face_alignment(img)
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    if(len(faces)!=1):
        raise "more than 1 face detected"
    landmarks=predictor(gray,faces[0])
    _27Minus8ForY=int(landmarks.part(8).y-landmarks.part(27).y)
    ratio= (_27Minus8ForY / img.shape[0])/0.36
    ymin = int(landmarks.part(27).y - (img.shape[0]*ratio*0.45))
    ymax = int(landmarks.part(27).y + (img.shape[0]*ratio*0.55))
    xmin=int(landmarks.part(27).x - (img.shape[0]*ratio*0.50))
    xmax=int(landmarks.part(27).x + (img.shape[0]*ratio*0.50))
    print(landmarks.part(27))
    print("(%d,%d),(%d,%d)"%(xmin,ymin,xmax,ymax))
    croppedImage=img[ymin:ymax,xmin:xmax]
    return croppedImage


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)
    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    return 1 if brightness == 255 else brightness / scale


def specsDetection(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    ### x_min and x_max
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    nose_bridge_x = []
    nose_bridge_y = []
    for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])
    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    ### ymin (from top eyebrow coordinate),  ymax
    y_min = landmarks[20][1]
    y_max = landmarks[31][1]
    img2 = Image.fromarray(img)
    img2 = img2.crop((x_min,y_min,x_max,y_max))

    img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
    edges_center = edges.T[(int(len(edges.T)/2))]
    if 255 in edges_center:
        return True
    else:
        return False


def bg(img):
    print("background called")
    height,width,channel=img.shape
    print(img.shape)
    bg_img=cv2.imread("white.jpg")
    mp_selfie_segmentation=mp.solutions.selfie_segmentation
    selfie_segmentation=mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    print("check1")
    # create videocapture object to access the webcam
    RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=selfie_segmentation.process(RGB)
    condition=np.stack((results.segmentation_mask,)*3,axis=-1)>0.5
    bg_img=cv2.resize(bg_img,(width,height))
    print("check2")
    output_image=np.where(condition,img,bg_img)
    return output_image

