import os
import requests
import json
API_TOKEN = "a9c85a24476b1aaecfa2ec81f468c78b"
API_BASE_URL = "https://api-service.vanceai.com/web_api/v1"

def upload_file(filepath):
    if not os.path.exists(filepath):
        raise ValueError("File does not exist")
    r = requests.post(f"{API_BASE_URL}/upload",
                      data={'api_token': API_TOKEN}, files={'file': open(filepath, 'rb')})
    if r.status_code == 200:
        jsondata = json.loads(r.text)
        uid = jsondata['data']['uid']
        print('UID generated ', uid)
        return uid
    return False


def improve_quality(uid):
    r = requests.post(f"{API_BASE_URL}/transform",
                      data={'api_token': API_TOKEN, 'uid': uid, 'jconfig': json.dumps({'job': 'real_esrgan','config':{"module":"real_esrgan","module_params":{"model_name": "RealEsrganStable","scale": "1x"}}})})
    if r.status_code == 200:
        jsondata = json.loads(r.text)
        print(r.text)
        trans_id = jsondata['data']['trans_id']
        print("Trans id generated ", trans_id)
        return trans_id

def download_file(trans_id, outputpath):
    r = requests.post(f"{API_BASE_URL}/download",
                      data={'api_token': API_TOKEN,'trans_id':trans_id})
    print('Code ',r.status_code)
    if r.status_code==200:
        with open(outputpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
    

if __name__ == "__main__":
    uid = upload_file('in.jpg')
    if uid:
        trans_id = improve_quality(uid)
        if trans_id:
            download_file(trans_id, "out.jpg")
        else:
            print('Failed to generate trans_id')
    else:
        print('failed to generate uid')
