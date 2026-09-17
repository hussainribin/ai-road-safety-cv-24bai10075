# AI Road Safety Analyzer Using Computer Vision

**Developed by:**
**HUSEN RIBINWALA**,
  **Registration No: 24BAI10075**,
  **B.Tech Computer Science (AI & ML)**

## Description

An academic project developed for B.Tech Computer Science (AI & ML). This system processes road images and video frames to detect lanes, identify vehicles using pretrained YOLO models, and estimate basic lane positioning to issue visual safety warnings.

## Architecture & Computer Vision Theory

### 1. Lane Detection

The system utilizes the following Computer Vision techniques:

**Grayscale Conversion → Gaussian Blur → Canny Edge Detection → ROI Masking → Hough Line Transform**

These stages are used to preprocess the road image, identify edges, isolate the relevant road region, and detect potential lane-line segments.

### 2. Vehicle Detection

The system implements **Ultralytics YOLOv8n (Nano)** for vehicle detection. It generates bounding boxes for selected vehicle categories using the pretrained MS COCO model.

The targeted vehicle classes include:

* Car
* Motorcycle
* Bus
* Truck

### 3. Lane Position Analysis

The system estimates the ego-vehicle position using the **camera/image center as a proxy**. This position is compared with the estimated center of the detected lane boundaries to generate a basic visual safety indication.

## Limitations

* Requires relatively clear lane markings and suitable lighting conditions.
* The system is not calibrated using physical camera intrinsics; therefore, position offsets are pixel-based estimations rather than actual metric distances.
* The ego-vehicle center is approximated using the camera/image center and assumes that the camera is reasonably centered on the dashboard.
* Detection performance may vary depending on road conditions, weather, visibility, and the underlying pretrained model.

## Future Improvements

* Integration of **Kalman Filters** for temporal lane tracking and smoother lane estimation across video frames.
* Application of **Perspective Transformation (Bird's Eye View)** for improved lane geometry and distance estimation.
* Development of a custom fine-tuned YOLO model for region-specific vehicle categories, such as **auto-rickshaws**.
* Improved camera calibration and real-world distance estimation for more accurate road-safety analysis.

## Academic Disclaimer

This project is developed for **academic and educational purposes** to demonstrate the application of Computer Vision techniques in road-scene analysis. It is not intended to function as a certified Advanced Driver Assistance System (ADAS) or autonomous-driving system.
