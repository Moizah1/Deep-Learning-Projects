# 🥚 Egg Detection System using Deep Learning and YOLOv11n

A deep-learning object detection system that **locates and counts eggs** in images, videos and live camera streams. A COCO-pretrained **YOLOv11n** convolutional neural network is fine-tuned on a Kaggle egg dataset (transfer learning), then used to draw bounding boxes, confidence scores and the total egg count.

---

## 1. Project Introduction

Counting eggs by hand on farms, in packing lines and in grocery inspection is slow and error-prone. This project trains a lightweight detector, **YOLOv11n** (the "nano" model of Ultralytics YOLO11), that runs fast enough for real-time use, even on a CPU or a modest GPU.

## 2. Problem Statement

Given an image or video frame, automatically find every egg, mark it with a bounding box and report how many eggs are present, robustly across lighting, camera angle, egg colour and partial occlusion.


