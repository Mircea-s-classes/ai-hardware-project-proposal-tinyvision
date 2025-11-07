# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal

## 1. Title - OpenMV H7 Face Sentiment Analysis
Team Name: TinyVision 

Members: Maria Molchanova, Rohina Naqshbandi, & Jordan Ho 

## 2. Platform Selection
We selected TinyML as it provides exciting possibilities for rapid lightweight detection and inference for many small applications. This technology allows us to process data in real-time, which is crucial for applications like sentiment analysis in healthcare. With TinyML, we can make devices smarter and more responsive while maintaining efficiency. Specifically, we will be using the OpenMV Cam H7 for its embedded camera and capabilities for machine vision. 

## 3. Problem Definition
Our goal is to develop a real-time face sentiment analysis system that can detect emotions from facial expressions. This is particularly important in healthcare, where understanding a patient’s emotional state can significantly enhance care (e.x. in therapy and other treatments). By using TinyML, we can ensure our solution is efficient, low-latency, and energy-efficient, making it practical for everyday use. 

## 4. Technical Objectives
Implement and optimize a facial emotion detection pipeline that efficiently maps to both software (ML model) and hardware (OpenMV H7 architecture) components. Differentiate between several classes of faces (happy, sad, neutral, etc) with >80 percent accuracy and at a rate of >20 frames per second---determined through some rough testing of our application. Further analysis after the initial implementation may involve: power consumption and hardware resource utilization compared to baseline CPU performance.

## 5. Methodology
We will use MicroPython to load our model and create our camera process loop. Edge Impulse can be used to train the model and organize our image datasets, then quantize to TensorFlow Lite. We will load the TensorFlow lite model into the OpenMV H7 and schedule inference. Facial detection via Haar cascade is already present in the firmware, so we can use this to first identify frames with faces to pass to our detection. 

## 6. Expected Deliverables
Working demo, GitHub repository, documentation, presentation slides, and final report.The working demo will consist of a small application which shows a live camera stream and the associated face detection and sentiment labels in real time. 

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| Maria Molchanova | Hardware | Setup, integration |
| Jordan Ho | Software | Model training, inference |
| Rohina Naqshbandi | Testing | Optimization, performance |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
1x OpenMV Cam H7
Datasets can be sourced from resources such as Kaggle https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition 

## 10. References
https://docs.openmv.io/library/omv.tf.html 

https://docs.edgeimpulse.com/tutorials/end-to-end/image-classification 

https://www.edgeimpulse.com/blog/tensorflow-with-the-edge-impulse-python-sdk/ 

https://www.mdpi.com/2079-9292/11/8/1240 

Asmara, R. A., Rosiani, U. D., Mentari, M., Syulistyo, A. R., Shoumi, M. N., & Astiningrum, M. (2024). An experimental study on deep learning technique implemented on low specification OpenMV CAM H7 device. JOIV International Journal on Informatics Visualization, 8(2), 1017. https://doi.org/10.62527/joiv.8.2.2299