import cv2

import dlib
import cv2



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


# img=cv2.imread("images/bg3.jpg")
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



