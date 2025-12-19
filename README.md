[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)
# AI Hardware Project Template
ECE 4332 / ECE 6332 — AI Hardware  
Fall 2025

## 🗂 Folder Structure
- `docs/` – project proposal and documentation  
- `presentations/` – midterm and final presentation slides  
- `report/` – final written report (IEEE LaTeX and DOCX versions included)  
- `src/` – source code for software, hardware, and experiments  
- `data/` – datasets or pointers to data used


# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  

---

# AI Hardware Project Report

## 1. Title - OpenMV H7 Face Sentiment Analysis
Team Name: TinyVision 

Members: Maria Molchanova, Rohina Naqshbandi, & Jordan Ho 

## 2. Problem Definition
Our goal was to develop a real-time face sentiment analysis system that can detect emotions from facial expressions. This is particularly important in healthcare, where understanding a patient’s emotional state can significantly enhance care (e.x. in therapy and other treatments). By using TinyML, we can ensure our solution is efficient, low-latency, and energy-efficient, making it practical for everyday use. 

## 3. Chosen Hardware
We used the OpenMVH7 plus. The device features a Cortex M7 processor at 480 MHz and 32 MB SDRAM, with an OV5640 image sensor for taking images.
## 4. Technical Objectives
Initial objectives were to implement and optimize a facial emotion detection pipeline that efficiently maps to both software (ML model) and hardware (OpenMV H7 architecture) components, and fifferentiates between several classes of faces (happy, sad, neutral, etc) with >80 percent accuracy and at a rate of >20 frames per second.

Through testing of our application and further research (Asmara et al.), we determined that based on prior work we could more realistically achieve closer to 3 FPS with this accuracy.

## 5. Methodology

### OpenMV software
Software is written in MicroPython using the OpenMV IDE. Haar cascades are preloaded from those in the OpenMV firmware. Most models tested were deployed with Edge Impulse which provides library functions for loading in the model.


### CNN Training

### MobileNetV2 training
 In tandem with the CNN, we optimized a MobileNetV2 Transfer learning architecture. We trained this with the RAF-DB dataset, which has a good variety of lighting, age/gender/ethnicity, and poses. MobileNetV2 is a lightweight architecture that is efficient for edge devices. It consists of inverted residual blocks and depthwise separable convolutions. The training process involved loading in a base model pretrained with ImageNet. These pretrained model weights exist for a variety of alpha values, down to around 0.35, allowing us to apply transfer learning even to quite small models. The alpha value is a multiplier which sets the filters in each layer, allowing MobileNetV2 models to scale way down.
While training, we experimented with the tradeoffs between image size, RGB vs grayscale, and alpha parameters to see how to achieve the most accurate model that would fit into the crucial 4MB area.
We first trained in Edge Impulse, but training for MobileNetV2 is quite limited as the free tier compute is limited to around 1 hour which came out to 30 epochs. We then transitioned to training locally. While we were able to experiment some more, we ran into issues with quantization, where we had trouble getting my quantized model to fit. We transitioned to uploading the trained int32 model to edge impulse and quantizing from there. We were able to push validation accuracy to almost 80%.
We trained on three classes, positive-neutral-negative, as the goal was to push the accuracy.


## 6. Results Discussion
 

## 7. Team Responsibilities

| Name | Role | Responsibilities |
|------|------|------------------|
| Maria Molchanova | Hardware | Setup, integration |
| Jordan Ho | Software | Model training, inference |
| Rohina Naqshbandi | Testing | Optimization, performance |

## 10. References
https://docs.openmv.io/library/omv.tf.html 

https://docs.edgeimpulse.com/tutorials/end-to-end/image-classification 

https://www.edgeimpulse.com/blog/tensorflow-with-the-edge-impulse-python-sdk/ 

https://www.mdpi.com/2079-9292/11/8/1240 

https://github.com/7abushahla/Student-Engagement/blob/main/notebooks/train_and_quantize.ipynb

Asmara, R. A., Rosiani, U. D., Mentari, M., Syulistyo, A. R., Shoumi, M. N., & Astiningrum, M. (2024). An experimental study on deep learning technique implemented on low specification OpenMV CAM H7 device. JOIV International Journal on Informatics Visualization, 8(2), 1017. https://doi.org/10.62527/joiv.8.2.2299


## 🧾 Submissions
- Commit and push all deliverables before each deadline.
- Tag final submissions with:
   ```bash
   git tag v1.0-final
   git push origin v1.0-final
   ```

## 📜 License
This project is released under the MIT License.
