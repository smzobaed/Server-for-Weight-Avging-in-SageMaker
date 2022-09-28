import os
import shutil
import zipfile
import json
from glob import glob
import shutil
#import pycognomotiv_prep as pc
#env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, save_model

INPUT_BUCKET = "cognomotiv-chai"
INPUT_BUCKET2 = "chai-aggregator"
OUTPUT_BUCKET = "cognomotiv-chai-output"

    
base_dir="/opt/ml/processing/input/local/"
main_model_dir="/opt/ml/processing/main_model"
weights=None
nagg = 0

local_model_list=[]
local_model_list=glob(base_dir+"*/", recursive = False)
print(local_model_list)        

numberof_client=len(local_model_list)

initial_client_model=[1]*numberof_client

for i in range(numberof_client):
    initial_client_model[i]=load_model(main_model_dir)
    print(initial_client_model[i].summary())


model_count=0
for each_folder in sorted(local_model_list):
    file_name=each_folder+"can-chai-v1.0.0.zip"
    extracted_dir=each_folder
    shutil.unpack_archive(file_name, extracted_dir)
    initial_client_model[model_count].load_weights(extracted_dir+"weights")
       
    if weights is None:
        weights = initial_client_model[model_count].get_weights()
    else:
        weights = [x + y for x, y in zip(weights, initial_client_model[model_count].get_weights())]
    nagg += 1
    model_count+=1 
weights = [x / nagg for x in weights]    


# load base model
aggregated_model = load_model(main_model_dir)
aggregated_model.set_weights(weights)
save_model(aggregated_model,"/opt/ml/processing/model/aggregated_new")

def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)
make_archive("/opt/ml/processing/model/aggregated_new", "/opt/ml/processing/model/aggregated_new.zip")

import boto3
s3 = boto3.resource('s3')
s3.meta.client.upload_file("/opt/ml/processing/model/aggregated_new.zip", OUTPUT_BUCKET, 'aggregated_new.zip')