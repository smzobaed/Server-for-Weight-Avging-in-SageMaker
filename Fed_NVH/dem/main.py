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