# Wildlife Detection & Comparison: Faster R-CNN vs. YOLOv8

This project implements an end-to-end object detection pipeline to identify and classify wildlife species (Oryx, Lion, and Ostrich) from camera trap imagery. It features a comparative analysis between a two-stage detector (**Faster R-CNN with ResNet-101**) and a one-stage detector (**YOLOv8-L**).

## 📌 Project Overview

The primary goal is to deploy high-resolution deep learning models for wildlife monitoring, optimized for accuracy and inference speed in remote deployments.

### Key Features

* **Stratified Data Splitting:** Prevents model bias by ensuring equal species proportions in training and testing sets.
* **Dual-Environment Architecture:** Isolated environments for TensorFlow and Ultralytics to prevent dependency conflicts.
* **Unified Dataset Alignment:** Both models were trained and evaluated on the exact same image splits for scientific rigor.

---

## 🖼️ Visual Comparison: Side-by-Side Detections

This section serves as a visual gallery to compare how each model localizes and classifies the same instance of wildlife.


| Wildlife Species | Faster R-CNN Detection | YOLOv8-L Detection |
| --- | --- | --- |
| **Oryx Gazella** | <img width="832" height="651" alt="image" src="https://github.com/user-attachments/assets/808b4095-ec16-4435-a310-642d96ebc9b5" /> |  <img width="795" height="610" alt="image" src="https://github.com/user-attachments/assets/145aa15d-8111-4ebd-86d0-7706a3b185a8" /> |
| **Panthera Leo** | <img width="828" height="601" alt="image" src="https://github.com/user-attachments/assets/c9d09e34-8929-457d-9604-c86442a63a5b" /> | <img width="843" height="620" alt="image" src="https://github.com/user-attachments/assets/4ade1983-599f-4026-9f72-880731562b26" /> |
| **Struthio Camelus** | <img width="827" height="626" alt="image" src="https://github.com/user-attachments/assets/a8aa8446-6467-4550-bd57-d108c4c17037" /> | <img width="826" height="624" alt="image" src="https://github.com/user-attachments/assets/aa710e0c-972b-4675-bd1c-f71481b65d0c" /> |

*Comparison of localization tightness and confidence scores across both architectures.*

---

## 🛠️ Data Configuration

### Label Map

* `OryxGazella`: 1 (Faster R-CNN) / 0 (YOLOv8)
* `PantheraLeo`: 2 (Faster R-CNN) / 1 (YOLOv8)
* `StruthioCamelus`: 3 (Faster R-CNN) / 2 (YOLOv8)

---

## 🚀 Model Details

### Faster R-CNN (Two-Stage) Optimization

* **Resolution:**  input with a **ResNet-101** backbone.
* **Anchor Strategy:** Set at `[0.25, 0.5, 1.0, 2.0]` to capture distant, small-bodied species.
* **Weighting:** Localization loss weight doubled to  for tighter bounding boxes in clusters.

### YOLOv8-L (One-Stage) Optimization

* **Architectural Choice:** The "Large" variant (YOLOv8-L) for high-capacity feature extraction.
* **Throughput:** Optimized for high-speed inference, reaching **82.44 FPS** on an RTX 4090.
* **Efficiency:** Lower memory overhead, allowing for larger batch sizes (Batch=8) at  resolution.

---

## 📊 Comparative Results Summary

| Performance Metric | Faster R-CNN (ResNet-101) | YOLOv8-L |
| --- | --- | --- |
| **mAP (0.50:0.95)** | 0.750 (75%) | **0.900+** |
| **Throughput (FPS)** | 7.17 FPS | **82.44 FPS** |
| **Inference Time** | 67.79 ms | **9.22 ms** |
| **Deployment Suitability** | Server-side / High Precision | **Edge / Solar-powered** |

### 🔍 Key Takeaways

1. **Speed Advantage:** YOLOv8-L is approximately **11.5x faster** than Faster R-CNN.
2. **Precision:** Faster R-CNN excels in localization precision (AP @ 0.50 = 95.5%), making it ideal for scientific census work where speed is secondary.
3. **Scalability:** YOLOv8-L can handle significantly more camera streams per GPU, making it the superior choice for wide-area surveillance.

---

## 📥 Getting Started

1. **Setup:** Create isolated conda/virtual environments for TF2 and Ultralytics.
2. **Run Pipeline:** Execute `Preprocesssing.ipynb` followed by the model-specific training notebooks.
3. **Benchmarking:** Use `Evaluation.ipynb` to generate the comparison metrics.
