# bit-STED
Object Detection Model

Related paper: <b>Bit-STED: A Lightweight Transformer for Accurate Agave Counting with UAV Imagery </b>

This paper presented bit-STED, a novel and simplified transformer encoder architecture for efficient agave plant detection and accurate counting using unmanned aerial vehicle (UAV) imagery. Addressing the critical need for accessible and cost-efficient solutions in agricultural monitoring, this approach automates a process that is typically time-consuming, labor-intensive, and prone to human error in manual practices. The bit-STED model comprises a simplified backbone of a two-scale transformer, Dual Patch Norm (DPN) with activation, BitNet quantization, Multi-Query Attention (GQA), and circular bounding box (C-Bbox) predictions for the roughly circular shape of the agave rosettes. To complement the detection model, an algorithm for accurate plant counting in orthophotos was developed to effectively manage plants spanning multiple tiles using Non-Maximum Suppression based on Fractional Area (NMS-FA). The experimental results demonstrated that the bit-STED model outperformed the baseline models in terms of detection and agave plant count performance. Specifically, the bit-STED nano model achieved F1 scores of 96.66\% on a map with younger plants and 96.43\% on a map with larger, highly overlapping plants. These scores surpassed state-of-the-art baselines, such as YOLOv8 Nano (F1 scores of 96.42\% and 96.38\%, respectively) and DETR (F1 scores of 93.03\% and 85.61\%, respectively). Furthermore, the bit-STED nano model was significantly smaller, being less than one-eighth the size of the YOLOv8 nano model (1.4 MB compared to 12.0 MB), had fewer trainable parameters (0.35M compared to 3.01M), and was faster in average inference times (14.62 ms compared to 18.28 ms).

<p>Data can be available on request</p>
The data of the images are tif files with the multispectral data of the image
The image size is 224x224

DSTAdam optimizer used in training model was obtained from 
https://github.com/kunzeng/DSTAdam

## General overview of followed methodology.
<img width="2743" height="1125" alt="general" src="https://github.com/user-attachments/assets/e3473238-5048-4e0f-b270-ab29f6c12e62" />

We named the bit-simplified transformer encoder for detection (bit-STED) based on the following key points:

**1.** Simplified backbone with two-scale transformers. This is based on the proposal of a pyramid vision transformer (PVM) (Wang et al., 2021) for feature extraction to replace CNNs with transformers.

**2.** Dual patch norm (DPN) (Kumar et al., 2023) with activation for increased extraction of nonlinear features.

**3.** BitNet network (Ma et al., 2024) as a regularization method to improve transformer performance and increase the stability of the model.

**4.** Transformer encoder with multiquery attention (GQA) (Ainslie et al., 2023) to avoid overhead by diminishing internal cache utilization.

**5.** The circular bounding box (C-Bbox) (Liu et al., 2020) is used as an alternative to conventional bounding boxes (Bbox) (Redmon and Farhadi, 2016), but adapted for circular detection.

**6.** Simplified head based on RetinaNet head (Lin et al., 2018)

**7.** Two-component loss function: Box regressor and dense predictor class.

## Backbone overview for two-stage transformer.
<img width="3005" height="1520" alt="two_stages" src="https://github.com/user-attachments/assets/3a39b1ca-cafa-4fbb-bfed-7eca4610e88c" />

## Simplified Detection Head of STED Model.
<img width="3203" height="1390" alt="head" src="https://github.com/user-attachments/assets/6ff4d69a-a644-47ba-8dca-6cf873fe0597" />

## Plant detection on orthomap for testing. a) Detection on single tiles of 224x224. b) Orthophoto with merged detections of all tiles. c) Orthophoto with detections filtered by NMS-FA.
<img width="2280" height="814" alt="nmsArea1" src="https://github.com/user-attachments/assets/462fa2b8-baa3-44a3-8e6d-92dc97aba0e3" />

## Plant detection on orthomap for testing. a) Detection on single tiles of 224x224. b) Orthophoto with merged detections of all tiles. c) Orthophoto with detections filtered by NMS-FA.
<img width="2254" height="814" alt="nmsArea" src="https://github.com/user-attachments/assets/f66d73de-5305-4642-879e-ef98ba0df836" />


<details><summary>Trained weights</summary>

|                                  | <br>Image size | AP (%) | Inference Time (ms) | Size (mb) | <br>Trainable Params (M)<br> | <br><F1 (%)> | 
| -------------------------------------------------------------------------------------------- | ------------------- | -------------------- | --------------------- | ------------------------------- | ------------------------------------ | ------------------- |
| [bit-STED nano](https://drive.google.com/file/d/1SIg4AEdD-duo_-p6yZ5wCZmzn0ROtmtV/view?usp=drive_link) | 224x224                 | 91.17                 | 14.62                  | 1.4                      | 0.35                            | 96.43               | 
| [bit-STED small]([https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt](https://drive.google.com/file/d/1kyCclkaR1HPUMSHNZq4Yjkg-ZLel469h/view?usp=drive_link)) | 224x224                 | 85.81                 | 14.50                  | 4.9                     | 1.21                            | 94.72                | 
| [bit-STED medium]([https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt](https://drive.google.com/file/d/1cGVp7fCWQs2DxOpFvx4cWrNw1r4-IHa4/view?usp=drive_link)) | 224x224                 | 90.24                 | 22.26                  | 22.26                     | 6.62                            | 96.13               | 
</details>

