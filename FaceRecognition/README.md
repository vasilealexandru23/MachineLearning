# Image Verification and Recognition using FaceNet and DeepFace architecture ideas

### This directory contains implementation of Image Verification and Recognition.

**NOTE: The FaceNet model takes a lot of data and a long time to train, so I use a pretrained Inception model using "channels last" convention.**

## `Structure of the directory:`
  * `images/` -> Directories with different anchor images of people.
  * `keras-facenet-h5/` -> Pre-trained model architecture and the precomputed weights.
  * `facerecognition.py` -> File that import the model pre-trained and implemnets face verification and recognition functions.

**NOTE: For more details about implementation check comments.**

Copyright 2024 Vasile Alexandru-Gabriel (vasilealexandru37@gmail.com)
