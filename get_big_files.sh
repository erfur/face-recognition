#!/usr/bin/bash

wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xf lfw.tgz

mkdir openface
wget https://github.com/krasserm/face-recognition/raw/master/weights/nn4.small2.v1.h5 -O openface/nn4.small2.v1.h5

mkdir models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/landmarks.dat