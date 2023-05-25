#!/bin/bash

runtime="--runtime=nvidia"
gpu_option="-e NVIDIA_VISIBLE_DEVICES=none"
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    runtime="--runtime=nvidia"
    gpu_option="-e NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,display,video,compute"
fi

NAME=ueda-egl-test

docker run \
  ${runtime} \
  --name ${NAME} \
  -i \
  --rm \
  ${gpu_option} \
  --shm-size=1024m \
  -e SIZEW=1920 \
  -e SIZEH=1080 \
  -e PASSWD=mypasswd \
  -e BASIC_AUTH_PASSWORD=mypasswd \
  -e NOVNC_ENABLE=true \
  -p 6081:8080 \
  ${NAME}
