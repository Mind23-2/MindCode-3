#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [[ $# -lt 3 || $# -gt 4 ]]; then 
    echo "Usage: bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [DATASET_NAME][DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

model=$(get_real_path $1)
data_path=$(get_real_path $2)

if [ $3 == 'Sintel' ] || [ $3 == 'TUM' ] || [ $3 == 'Kitti' ]; then
  dataset_name=$3
else
  echo "DATASET_NAME can choose from ['Sintel', 'TUM', 'Kitti']"
  exit 1
fi

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo $model
echo $data_path
echo $dataset_name
echo $device_id

function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    bash build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
     if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    cd result_Files
    mkdir Kitti
    mkdir Sintel
    mkdir TUM
    cd ..
    ../ascend310_infer/main --midas_file=$model --image_path=$data_path --dataset_name=$dataset_name --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    python ../postprocess.py --dataset_path=$data_path --result_path=./result_Files --dataset_name=$dataset_name &> acc.log
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}
compile_app
infer
cal_acc
