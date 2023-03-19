import copy
from flask import Flask,jsonify,request,make_response,send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2
from PIL import Image,ImageEnhance
import utils
import time
import improve_quality
import bgModule
app=Flask(__name__)
CORS(app)

@app.route("/",methods=['GET'])
def helloWorld():
    return jsonify( {"data":"helloworld"})

@app.route("/download",methods=['GET'])
def download():
    return send_file(r'/home/azureuser/pass2Python/output.jpg')


@app.route('/passport',methods=['POST'])
def passport():
    try:
        fileName=time.time()
        print("Request processing "+str(fileName))

        face = request.get_json()
        face_jpg_original = base64.b64decode(face["face"])


        f_jpg_as_np = np.frombuffer(face_jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(f_jpg_as_np, flags=1)

        cv2.imwrite('in.jpg',frame)
        print("check1")
        #image quality improve code
        print("check1")
        


        utils.getEyeQImage()
        if not utils.rembgApi():
            raise Exception("Background removal failed")
        frame=cv2.imread('outeyeq.jpg')


        im = Image.fromarray(np.uint8(frame)).convert('RGBA')
        img=copy.deepcopy(frame)

        if(utils.imageCheckFault(copy.deepcopy(frame))):
                return {'err': "image not intact. please refer image guidlines. Your photo should be atleast waist level to head.",'error':True}

        dim=(800,int((800/img.shape[1])*img.shape[0]))
        img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
        data =Image.fromarray(np.uint8(img)).convert('RGBA')
        # data=bgModule.bg_remover(data)
        img=cv2.cvtColor(np.array(data),cv2.COLOR_RGBA2BGRA)

        dim=(800,int((800/img.shape[1])*img.shape[0]))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img=utils.enhance(img)

        dim=(800,int((800/img.shape[1])*img.shape[0]))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img=utils.cropImage(img)

       

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)

        im = Image.fromarray(np.uint8(img)).convert('RGB')
        # light=utils.calculate_brightness(im)
        # brightness=ImageEnhance.Brightness(im)
        # im=brightness.enhance(1.6-light)

        img=im.resize((600,600),Image.ANTIALIAS)
        img.save("o1.jpg")
        img.save('output.jpg',dpi=(300,300), optimize=True, quality=100, jfif_unit=1,  jfif_density=(300,300))

        i=cv2.cvtColor(np.array(im),cv2.COLOR_RGBA2BGRA)
        cv2.imwrite('imp.jpg',i)
        improve_quality.quality('imp.jpg')
        i=cv2.imread('input.jpg')
        #i=cv2.copyMakeBorder(i, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        imgCopy=copy.deepcopy(i)
        specFlag=utils.specsdetection(imgCopy)
        mouthOpen=utils.mouth_open(imgCopy)
        dim=(1600,int((1600/i.shape[1])*i.shape[0]))
        ihd = cv2.resize(i, dim, interpolation = cv2.INTER_CUBIC)



        iSd = cv2.imread('output.jpg',cv2.IMREAD_UNCHANGED)
        (flag, encodedImageHd) = cv2.imencode(".png", ihd)  # Encode Image
        (flag, encodedImageSd) = cv2.imencode(".png", iSd)  # Encode Image

        base64_bytesHd = base64.b64encode(bytearray(encodedImageHd))
        base64_bytesSd = base64.b64encode(bytearray(encodedImageSd))
        shadow=False
       

        return {"image":str(base64_bytesHd),"imageSd":str(base64_bytesSd),"spectacles":specFlag,"shadow":shadow,"mouthopen":mouthOpen}

    except Exception as e:
        print(e)
        return {'err': "error occured. Please check input photo credibility.",'error':True}

if (__name__=="__main__"):
    app.run(port="5050")
