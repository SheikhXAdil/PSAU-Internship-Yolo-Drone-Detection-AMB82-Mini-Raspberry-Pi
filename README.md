# YOLO Drone Detection — AMB82-Mini & Raspberry Pi

A practical **Edge AI computer-vision project** developed during my internship at the **Research & Development Center, Prince Sattam bin Abdulaziz University (PSAU)**.

The project focused on training and deploying a custom **YOLOv7-based drone detection model** on resource-constrained edge hardware. The work progressed from deployment on the **AMB82-Mini** to deployment on a **Raspberry Pi Zero 2 W running Ubuntu**, followed by an optimization experiment using the smaller **YOLO11n** model to improve Raspberry Pi inference performance.

This repository contains the training experiments, embedded deployment work, Raspberry Pi inference implementation, converted models, and supporting test data from this part of the internship.

## Project Overview

The work followed an iterative Edge AI deployment process:

```text
Drone Detection Dataset
          ↓
      YOLOv7 Training
          ↓
     Model Evaluation
          ↓
   AMB82-Mini Deployment
          ↓
 Raspberry Pi Zero 2 W
          ↓
  Performance Evaluation
          ↓
      YOLO11n Training
          ↓
     NCNN Conversion
          ↓
 Raspberry Pi Optimization
```

The main objective was to investigate the practical challenges of taking a custom object-detection model from a training environment to increasingly resource-constrained edge platforms.

## Dataset

The drone detection models were trained using the **YOLO Drone Detection Dataset** by `muki2003`.

**Dataset:**
[YOLO Drone Detection Dataset — Kaggle](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset)

The same dataset was used for both the YOLOv7 and YOLO11n experiments, allowing the later model iteration to be evaluated under the same dataset conditions.

## YOLOv7 Training

The initial computer-vision model was based on **YOLOv7**.

The repository contains the training notebook used for the drone-detection model:

```text
yolo-drone-detection-amb82-mini/
└── training/
    └── yolo-training.ipynb
```

The training workflow covered:

* Dataset preparation
* YOLOv7 model training
* Model evaluation
* Model export
* Preparation for edge deployment

The resulting model was then taken through separate deployment paths for the AMB82-Mini and Raspberry Pi.

## AMB82-Mini Deployment

The first hardware deployment target for the custom drone detector was the **AMB82-Mini**.

Rather than developing the entire embedded object-detection application from scratch, I started from the **provided `ObjectDetectionLoop` example sketch** and customized it to run the trained **YOLOv7-tiny** model.

This provided a practical way to adapt an existing embedded AI example to a custom-trained object-detection workload.

### Deployment Approach

The working implementation involved:

* Starting from the provided AMB82-Mini `ObjectDetectionLoop` example
* Integrating the custom YOLOv7-tiny model
* Adapting the example for the drone-detection use case
* Running camera-based object detection
* Testing the detector directly on the AMB82-Mini
* Evaluating real-time inference performance

The repository contains the customized Arduino implementation under:

```text
yolo-drone-detection-amb82-mini/
└── arduino_amb82/
    └── ObjectDetectionLoop/
```

The project also contains a separate benchmark-oriented implementation:

```text
ObjectDetectionLoopWithBenchmark/
```

This was an **experimental attempt to add additional benchmarking functionality**. The implementation caused issues on the AMB82-Mini and was not used as the successful deployment path.

Therefore, the benchmark implementation should not be considered the final working AMB82-Mini deployment. The successful deployment was based on the customized **`ObjectDetectionLoop`** example.

## AMB82-Mini Testing

Sample test images are included in:

```text
yolo-drone-detection-amb82-mini/
└── test_images/
```

These were used during the development and testing process alongside the embedded camera-based inference.

The AMB82-Mini work provided practical experience with adapting an existing edge-AI example to a custom object-detection model and understanding the constraints of running computer vision directly on embedded hardware.

## Raspberry Pi Zero 2 W Deployment

The next stage of the project moved the drone detector to a **Raspberry Pi Zero 2 W running Ubuntu**.

This provided a different edge-computing environment compared with the AMB82-Mini. Unlike the microcontroller-oriented AMB82-Mini deployment, the Raspberry Pi provided a Linux environment and allowed the use of tools such as **Python, OpenCV, and NCNN**.

The Raspberry Pi implementation is contained in:

```text
yolo-drone-detection-raspberry-pi/
```

and currently includes:

```text
best.ncnn.bin
best.ncnn.param
best.pt
detector.py
```

## NCNN-Based Inference

The YOLOv7 model was converted to **NCNN** for inference on the Raspberry Pi Zero 2 W.

The implementation uses:

* NCNN
* Python
* OpenCV
* Camera input
* CPU-based inference
* YOLO object detection
* Confidence filtering
* Non-Maximum Suppression

The inference pipeline is:

```text
Camera Frame
     ↓
Image Preprocessing
     ↓
Letterbox Resize
     ↓
BGR → RGB
     ↓
NCNN Input
     ↓
YOLOv7 Inference
     ↓
Detection Output
     ↓
Confidence Filtering
     ↓
Non-Maximum Suppression
     ↓
Bounding Boxes
     ↓
Annotated Output
```

The detector also measures different stages of the processing pipeline so that inference performance can be investigated separately from other operations such as image preprocessing and output handling.

## Raspberry Pi Environment

The deployment target was:

* **Raspberry Pi Zero 2 W**
* **Ubuntu Linux**
* Python
* OpenCV
* NCNN
* Camera input

The Raspberry Pi environment provided a useful platform for investigating how a custom object detector performs under significant CPU and memory constraints.

## Performance Investigation

Performance was an important part of the Raspberry Pi deployment.

The investigation considered factors including:

* Model architecture
* Model size
* Input resolution
* Inference latency
* FPS
* Preprocessing overhead
* Postprocessing overhead
* Camera processing
* CPU limitations
* Frame skipping
* Runtime efficiency

The initial YOLOv7 deployment demonstrated that the model was relatively computationally expensive for the Raspberry Pi Zero 2 W.

This motivated the next stage of the project: investigating whether a substantially smaller model could provide better inference performance.

## YOLO11n Optimization

To improve the Raspberry Pi deployment, the next iteration moved from YOLOv7 to **YOLO11n**.

YOLO11n was trained using the **same drone detection dataset** and then converted to NCNN for deployment on the Raspberry Pi Zero 2 W.

The repository contains the YOLO11n experiment under:

```text
yolov11n-drone-detection/
```

including:

```text
metadata.yaml
model.ncnn.bin
model.ncnn.param
model_ncnn.py
yolo11n_detector.py
yolo11n_training.ipynb
```

The experiment focused on:

* Training YOLO11n
* Evaluating the model
* Converting the model to NCNN
* Deploying it on the Raspberry Pi Zero 2 W
* Camera-based inference
* Measuring inference performance
* Comparing the behavior of YOLOv7 and YOLO11n
* Investigating model efficiency on constrained hardware

## Why YOLO11n?

The move to YOLO11n was motivated by the computational limitations observed during the YOLOv7 Raspberry Pi deployment.

The experiment investigated whether selecting a smaller model could reduce inference latency and improve practical FPS while retaining useful drone-detection capability.

This demonstrated an important Edge AI principle:

```text
Larger / More Complex Model
            ↓
       Higher Compute
            ↓
      Higher Latency

Smaller / Efficient Model
            ↓
       Lower Compute
            ↓
      Better Edge Performance
```

The detailed measurements and conclusions from the YOLOv7 and YOLO11n experiments are documented in the broader PSAU internship documentation and presentation.

## Repository Structure

```text
PSAU-Internship-Yolo-Drone-Detection-AMB82-Mini-Raspberry-Pi/
│
├── yolo-drone-detection-amb82-mini/
│   │
│   ├── arduino_amb82/
│   │   ├── ObjectDetectionLoop/
│   │   │   └── Customized YOLOv7-tiny deployment
│   │   │
│   │   └── ObjectDetectionLoopWithBenchmark/
│   │       └── Experimental benchmarking attempt
│   │
│   ├── test_images/
│   │   └── Sample drone images
│   │
│   └── training/
│       └── yolo-training.ipynb
│
├── yolo-drone-detection-raspberry-pi/
│   ├── best.ncnn.bin
│   ├── best.ncnn.param
│   ├── best.pt
│   └── detector.py
│
└── yolov11n-drone-detection/
    ├── metadata.yaml
    ├── model.ncnn.bin
    ├── model.ncnn.param
    ├── model_ncnn.py
    ├── yolo11n_detector.py
    └── yolo11n_training.ipynb
```

The repository represents the progression:

```text
                    YOLOv7
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       AMB82-Mini        Raspberry Pi Zero 2 W
       YOLOv7-tiny              │
                                ↓
                            YOLO11n
                                │
                                ↓
                     Raspberry Pi Optimization
```

## Technologies

### Machine Learning

* **YOLOv7**
* **YOLOv7-tiny**
* **YOLO11n**
* PyTorch
* Object detection
* Model evaluation
* Custom drone dataset

### Edge AI

* **NCNN**
* Model conversion
* Edge inference
* Model optimization
* Performance benchmarking

### Computer Vision

* **OpenCV**
* Image preprocessing
* Letterbox resizing
* Bounding-box processing
* Non-Maximum Suppression
* Camera-based inference

### Hardware

* **AMB82-Mini**
* **Raspberry Pi Zero 2 W**
* Ubuntu Linux
* Camera modules

### Development

* Python
* C/C++
* Arduino
* Jupyter Notebook
* ESP/embedded development tools
* Kaggle

## Key Engineering Lessons

This project demonstrated that deploying an object-detection model on edge hardware requires considerably more than simply obtaining a trained model.

Practical deployment required consideration of:

* Model architecture
* Model size
* Hardware compute capability
* Available memory
* Runtime compatibility
* Input resolution
* Preprocessing
* Postprocessing
* Camera I/O
* Inference latency
* Real-world FPS

The progression from **YOLOv7 → Raspberry Pi → YOLO11n** provided practical evidence of the importance of choosing a model based not only on its detection capability but also on the computational constraints of the target hardware.

The AMB82-Mini work also demonstrated the value of adapting and extending existing embedded AI examples rather than unnecessarily rebuilding the complete application infrastructure from scratch.

## Internship Context

This project was developed during my internship at the **Research & Development Center, Prince Sattam bin Abdulaziz University (PSAU)**.

It formed one of the main computer-vision components of my broader investigation into **Edge AI deployment on resource-constrained hardware**.

The internship progressed through several experiments:

1. **ESP32 MNIST** — initial TinyML deployment using TensorFlow Lite Micro.
2. **YOLOv7-tiny on AMB82-Mini** — custom drone detection using a modified provided `ObjectDetectionLoop` example.
3. **YOLOv7 on Raspberry Pi Zero 2 W** — NCNN-based deployment under Ubuntu.
4. **YOLO11n on Raspberry Pi Zero 2 W** — model optimization and performance investigation.
5. Evaluation of the practical trade-offs between model complexity, inference performance, and hardware constraints.

This repository contains the main **YOLO drone-detection portion** of that internship work.

## Related PSAU Internship Archive

For the complete internship context, documentation, presentation, and related ESP32 experiment:

**[PSAU Internship — Edge AI Deployment Pipeline](https://github.com/SheikhXAdil/PSAU-Internship-Archive)**

The archive contains:

* Internship technical documentation
* Internship presentation
* ESP32 MNIST TensorFlow Lite Micro experiment
* YOLO drone-detection work
* Supporting implementations

The archive provides the broader research context, while this repository focuses specifically on the **YOLO-based drone-detection experiments and edge deployments**.

## Documentation

The detailed methodology, deployment findings, performance measurements, and analysis are available in the internship documentation and presentation maintained in the PSAU internship archive.

**[PSAU Internship Archive](https://github.com/SheikhXAdil/PSAU-Internship-Archive)**

The archive contains:

* `Edge AI Deployment Pipeline Documentation.pdf`
* `Edge AI Deployment Pipeline Presentation.pptx`

These documents provide the broader analysis of the hardware constraints, model-selection decisions, deployment process, optimization work, and performance findings.

## Outcome

This project provided practical experience taking a custom object-detection model through the edge deployment lifecycle:

```text
Dataset
   ↓
Model Training
   ↓
Evaluation
   ↓
Model Conversion
   ↓
Hardware Deployment
   ↓
On-Device Inference
   ↓
Performance Evaluation
   ↓
Model Optimization
```

The work demonstrated the practical differences between deploying a computer-vision workload on an embedded AI platform and running it on a low-resource Linux-based edge computer.

It also showed how **model selection, runtime choice, preprocessing, hardware limitations, and inference optimization** all influence the feasibility of real-time Edge AI applications.

## Status

**Completed internship project.**

This repository is preserved as part of my **PSAU Edge AI internship work** and contains the YOLOv7 and YOLO11n drone-detection experiments conducted across the **AMB82-Mini and Raspberry Pi Zero 2 W**.

It serves as the implementation repository for the computer-vision portion of the broader **PSAU Edge AI Deployment Pipeline**.
