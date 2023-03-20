import base64
import copy
import time

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, send_file
from PIL import Image, ImageEnhance

from bgModule import bg_remover
from utils import calculate_brightness, cropImage, mouth_open, specsdetection

# from rembg import remove


main=Blueprint("main",__name__)

@main.route("/",methods=['GET'])
def helloWorld():
    return jsonify( {"data":"helloworld"})

@main.route("/download",methods=['GET'])
def download():
    return send_file(r'..\output.png')

@main.route('/passport',methods=['POST'])
def passport():
    try:
        fileName=time.time()
        face = request.get_json()
        face_jpg_original = base64.b64decode(face["face"])

        print("check1")

        f_jpg_as_np = np.frombuffer(face_jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(f_jpg_as_np, flags=1)
        print("check2")

        cv2.imwrite("inputCache/"+str(fileName)+".jpg",frame)
        im = Image.fromarray(np.uint8(frame)).convert('RGBA')
        light=calculate_brightness(im)
        print("check3")

        img=frame
        dim=(800,int((800/img.shape[1])*img.shape[0]))
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
        data =Image.fromarray(np.uint8(img)).convert('RGBA')
        data=bg_remover(data)
        img=cv2.cvtColor(np.array(data),cv2.COLOR_RGBA2BGRA)

        # dim=(800,int((800/img.shape[1])*img.shape[0]))
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # # img=enhance(img)
        # print("enhance done")

        print("image crop")
        dim=(800,int((800/img.shape[1])*img.shape[0]))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img=cropImage(img)



        # img=remove(img,alpha_matting=True)
        # cv2.imwrite('int.png',img)
        # os.system('rembg i -a -ae 15 int.png out.png')
        # img=cv2.imread('out.png',cv2.IMREAD_UNCHANGED)
        # print("bg remove done")



        # bg=Image.open('white.jpg')
        # print(bg.size)
        # bg=bg.resize((data.size[0],data.size[1]))
        # print(bg.size)
        # print("bg added")
        # bg.paste(data,(0,0),mask=data)

        # print("image crop")
        # dim=(1600,int((1600/img.shape[1])*img.shape[0]))
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # img=cropImage(img)
        cv2.imwrite('remove.png',img)
        print("img :",img.shape)

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)

        im = Image.fromarray(np.uint8(img)).convert('RGBA')
        brightness=ImageEnhance.Brightness(im)
        im=brightness.enhance(1.6-light)


        # fill_color = (255,255,255)  # your new background color

        # im = im.convert("RGBA")   # it had mode P after DL it from OP
        # if im.mode in ('RGBA', 'LA'):
        #     background = Image.new(im.mode[:-1], im.size, fill_color)
        #     background.paste(im, im.split()[-1]) # omit transparency
        #     im = background
        im.save('dpiImage.png',dpi=(300,300))
        im=Image.open('dpiImage.png')
        i=cv2.cvtColor(np.array(im),cv2.COLOR_RGBA2BGRA)
        # cv2.imwrite('covert.png',i)

        #i=cv2.copyMakeBorder(i, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # cv2.imwrite('out1.png',i)
        imgCopy=copy.deepcopy(i)
        specFlag=specsdetection(imgCopy)
        mouthOpen=mouth_open(imgCopy)
        print(specFlag)
        (flag, encodedImage) = cv2.imencode(".png", i)  # Encode Image

        base64_bytes = base64.b64encode(bytearray(encodedImage))
        shadow=False
        if(light<0.3):
            shadow=True
        # cv2.imwrite("outputCache/"+str(fileName)+".jpg",i)
        cv2.imwrite('output.png',i)
        return {"image":str(base64_bytes),"spectacles":specFlag,"shadow":shadow,"mouthopen":mouthOpen}

    except Exception as e:
        return {'err': str(e)}



# if (__name__=="__main__"):
#     main.run(port="3000")