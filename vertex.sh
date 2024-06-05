#!/bin/bash

# Set up the parameters
JOB_NAME="megaportrait-training-$(date +%Y%m%d%H%M%S)"
REGION="us-central1"
PYTHON_VERSION="3.11"
MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT=1

# Submit the training job
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --python-module=train \
  --python-version=$PYTHON_VERSION \
  --machine-type=$MACHINE_TYPE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --accelerator-count=$ACCELERATOR_COUNT \
  --args="--config=./configs/training/stage1-base.yaml" \
  --requirements=requirements.txt \
  --setup-script=setup.sh