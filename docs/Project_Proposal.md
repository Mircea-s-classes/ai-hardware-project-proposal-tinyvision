# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal Template

## 1. Project Title
TinyVision 

Maria Molchanova, Rohina Naqshbandi, & Jordan Ho 

## 2. Platform Selection
We selected TinyML as it provides exciting possibilities for rapid lightweight detection and inference for many small applications. This technology allows us to process data in real-time, which is crucial for applications like sentiment analysis in healthcare. With TinyML, we can make devices smarter and more responsive while maintaining efficiency. 

## 3. Problem Definition
Our goal is to develop a real-time face sentiment analysis system that can detect emotions from facial expressions. This is particularly important in healthcare, where understanding a patient’s emotional state can significantly enhance care. By using TinyML, we can ensure our solution is efficient, low-latency, and energy-efficient, making it practical for everyday use. 

## 4. Technical Objectives
Differentiate between several classes of faces (happy, sad, neutral, etc) with around 80 percent accuracy (determined through some rough testing of our application). Have detection occur in realtime (greater than 20 FPS). 

## 5. Methodology
We will use MicroPython to load our model and create our camera process loop. Edge Impulse can be used to train the model and organize our image datasets, then quantize to TensorFlow Lite. We will load the TensorFlow lite model into the OpenMV H7 and schedule inference. Facial detection via Haar cascade is already present in the firmware, so we can use this to first identify frames with faces to pass to our detection. 

## 6. Expected Deliverables
Working demo, GitHub repository, documentation, presentation slides, and final report.The working demo will consist of a small application which shows a live camera stream and the associated face detection and sentiment labels in real time. 

## 7. Team Responsibilities
List each member’s main role.

| Name | Role | Responsibilities |
|------|------|------------------|
| Jordan Ho | Software | Coordination, documentation |
| Maria Molchanova | Hardware | Setup, integration |
| Rohina Naqshbandi | Testing | Model training, inference |

## 8. Timeline and Milestones
Provide expected milestones:

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
Datasets can be sourced from resources such as Kaggle https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition 

## 10. References
https://docs.openmv.io/library/omv.tf.html 

https://docs.edgeimpulse.com/tutorials/end-to-end/image-classification 

https://www.edgeimpulse.com/blog/tensorflow-with-the-edge-impulse-python-sdk/ 

https://www.mdpi.com/2079-9292/11/8/1240 
