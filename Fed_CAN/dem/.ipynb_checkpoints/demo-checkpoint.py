import tensorflow as tf
print(tf.__version__)
import os
import shutil
import zipfile
import json
#import pycognomotiv_prep as pc
#env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, save_model

INPUT_BUCKET = "cognomotiv-mb-chai"
OUTPUT_BUCKET = "cognomotiv-mb-pipeline-update"


def get_model_keys(session, bucket):
    s3_client = session.client("s3")
    next_token = None
    while True:
        if next_token is None:
            response = s3_client.list_objects_v2(Bucket=bucket)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket, ContinuationToken=next_token)
        next_token = response.get('NextContinuationToken')
        contents = response.get('Contents')
        if contents is not None:
            for content in contents:
                if content['Key'][-4:] == '.zip':
                    yield content['Key']
                    # print(f"s3://{bucket}/{content['Key']}")
                    # s3_client.download_file('MyBucket', 'hello-remote.txt', 'hello2.txt')
        if next_token is None:
            break

def download_model(session, bucket, key, tmpdir='./tmp'):
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    download_path = f"{tmpdir}/model.zip"
    s3_client = session.client("s3")
    s3_client.download_file(bucket, key, download_path)
    unzip_path = f"{tmpdir}/saved_model"
    if not os.path.isdir(unzip_path):
        os.mkdir(unzip_path)
    with zipfile.ZipFile(download_path, 'r') as zf:
        zf.extractall(unzip_path)
    return unzip_path

def wipe_tmpdir(tmpdir='./tmp'):
    shutil.rmtree(tmpdir)
    
    
base_dir="/opt/ml/processing/input/local/"
weights=None
nagg = 0

local_model_list=[]
for _,_,m_file_list in os.walk(base_dir):
    if m_file_list:
        m_file=m_file_list[0]
        print (m_file)
        local_model_list.append(m_file)
        
print(local_model_list)        
cnt=1

for each_model in sorted(local_model_list):
         
    model = load_model(base_dir+"local_"+str(cnt)+"/"+each_model)
    cnt+=1    
    if weights is None:
        weights = model.get_weights()
    else:
        weights = [x + y for x, y in zip(weights, model.get_weights())]
    nagg += 1
    
weights = [x / nagg for x in weights]    


# load base model
model = load_model("/opt/ml/processing/main_model/model_chai_larger.h5")
model.set_weights(weights)
save_model(model,"/opt/ml/processing/model/aggregated.h5")

import boto3
s3 = boto3.resource('s3')
s3.meta.client.upload_file("/opt/ml/processing/model/aggregated.h5", OUTPUT_BUCKET, 'aggregated.h5')