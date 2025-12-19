[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)
# AI Hardware Project Template
ECE 4332 / ECE 6332 — AI Hardware  
Fall 2025

## 🗂 Folder Structure
- `docs/` – project proposal and documentation  
- `presentations/` – midterm and final presentation slides  
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
Our goal was to develop a real-time face sentiment analysis system that can detect emotions from facial expressions. This is particularly important in healthcare, where understanding a patient’s emotional state can significantly enhance care (e.x. in therapy and other treatments). By using TinyML to create these Facial Expression Recognition (FER) models, we can ensure our solution is efficient, low-latency, and energy-efficient, making it practical for everyday use. 

## 3. Chosen Hardware
We used the OpenMVH7 plus. The device features a Cortex M7 processor at 480 MHz and 32 MB SDRAM, with an OV5640 image sensor for taking images.

## 4. Technical Objectives
&emsp;Initial objectives were to implement and optimize a facial emotion detection pipeline that efficiently maps to both software (ML model) and hardware (OpenMV H7 architecture) components, and differentiates between several classes of faces (happy, sad, neutral, etc) with >80 percent accuracy and at a rate of >20 frames per second.

&emsp;Through testing of our application and further research (Asmara et al., 2024), we determined that based on prior work we could more realistically achieve closer to 3 FPS with this accuracy. Our new goal for inference time was to beat 3 FPS. 

## 5. Methodology

### OpenMV software
&emsp;Software is written in MicroPython using the OpenMV IDE. Haar cascades are preloaded from those in the OpenMV firmware. Most models tested were deployed with Edge Impulse which provides library functions for conversion to TFLite models that were loaded on the Cam H7 Plus. Some of our models were trained in Google Colab for machine learning, which required custom TFLite model conversion scripts that performed proper int8 quantization.


### CNN Training
&emsp;Training in Edge Impulse and Google Colab, we first started off with training custom Convolutional Neural Networks (CNN) using several different datasets: FER2013, CK-extended, and RAF-DB. FER2013 is a collection of 35,685 48x48 pixel grayscale images of faces containing 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. CK+ is the Extended Cohn-Kanade dataset which contains sequences of image frames; the version we used included 981 labeled images taken from these videos for classifying 8 emotions---the same 7 emotions as FER2013 with an additional 'Contempt' class. RAF-DB stands for the Real-world Affective Faces Database, a dataset of 29672 real-world images downloaded from the Internet containing the same 7 facial expressions classes as FER2013; all images were independently labeled by 40 annotators. All of our training inputs were standardized to 48x48 greyscale image inputs. Data augmentation (random zoom, contrast, brightness, rotation, etc.) within the training process was also utilized.

&emsp; After researching and training different CNN architectures that would work well on the Cam H7 Plus for face expression recoginition, we encountered the following patterns in the attempt to achieve a highly accurate and well-fitted ML model: 
- overall accuracy improved when using fewer emotion classes since there was less emotional boundaries to differentiate 
- deeper architectures with larger quantities of convolutional blocks/layers increased model accuracy at the expense of long inference times and large cache memory usage in SRAM/DRAM of the Cam H7 Plus
- the chosen dataset with its quality in images and labels largely affected the achievable end accuracy, no matter how much tweaking we did with the learning rate, regularization strength, architectural depth, batch size, dropout, activation funcion, etc.
- across different datasets, Happy, Neutral, Angry, Surprise, and Sad had the least error most likely due to their distinct facial features (e.g. an open mouth for Surprise)
- accuracies seemed to peak around 70% in training epochs, 60% with validation sets, and 50% in testing
- converting ML to TFLite models involved int8 quantization to represent parameters in lower-bit representations; overall accuracy only dropped up to 5% at most which allowed deployment on the memory- and compute-constrained Cam H7 Plus
- real-world testing with these CNN models showed that models worked reliably well in practice, even if the training accuracy wasn't as high as 80%; however, good inferencing results depended on the framing and lighting conditions for the face 

&emsp;After familiarizing ourselves with the training process for FER models, we decided to continue training on the FER2013 dataset due to its size. To break past our plateuaing 70% training accuracy, we used Microsoft's FERPlus annotations that relabeled each FER2013 image using 10 crowd-sourced taggers to create a probability distribution among the same 8 classes as CK+. Using the new probability distribution, images were relabeled using majority vote to allow for one-hot encoding so that we can classify by emotion class. Additionally, bad samples were completely removed from the dataset. Preliminary training with this new model in Edge Impulse demonstrated the ability to reach 80% training accuracy.

&emsp;Through more research, a promising model shared on GitHub was found; it was created by a Master’s of Computer Science student in Poland by the name of Adam Wiącek for their thesis, and also implemented FERPlus for FER. We tested Wiącek's model claiming 87% testing accuracy on the Cam H7 Plus using a custom TFLite conversion script that required a representative dataset to perform int8 quantization. The model contained ~174k training params with a 244 KB size TFLite file and ecah inference on the Cam H7 Plus took ~334 ms which is ~3 FPS. Learning from Wiącek's architecture, we built our own CNN model while using optimized ML functions that would use as little computation and memory from the available ~1 MB of SRAM and 16/32 MB of SDRAM (synchronous DRAM) available. The optimizations implemented are as follows:
- SeparableConv2D replaces standard Conv2D by splitting convolution into depthwise and pointwise operations, significantly reducing parameters and computation.
- SpatialDropout2D replaces standard Dropout by dropping entire feature maps, improving feature independence and training stability.
- GlobalAveragePooling2D (GAP) replaces Flatten, reducing each feature map via averaging to lower memory usage with minimal accuracy loss.
- A subset of 5 emotion classes (Happy, Neutral, Angry, Sad, Surprised) was selected based on the lowest error in Wiącek's confusion matrix confirmed by observations in our own real-world testing
- The final FERplus-based model uses separable convolutions in all layers except the first, allowing full learning from the 48×48 input image.
- ReLU was used instead of LeakyReLU for faster convergence.
- An unconventional Dense layer after GAP was added to boost learning performance and attempt to surpass Wiącek's accuracy.
The final model size was cut down to ~77k parameters. The final accuracy was 83% in training, 80% in validation, and ~74% in test. Errors were balanced well across the 5 classes, which is a major improvement over earlier models which sometimes saw poor inferencing in Sad and Angry classes.

### MobileNetV2 training
&emsp;In tandem with the CNN, we optimized a MobileNetV2 Transfer learning architecture. We trained this with the RAF-DB dataset, which has a good variety of lighting, age/gender/ethnicity, and poses. MobileNetV2 is a lightweight architecture that is efficient for edge devices. It consists of inverted residual blocks and depthwise separable convolutions. The training process involved loading in a base model pretrained with ImageNet. These pretrained model weights exist for a variety of alpha values, down to around 0.35, allowing us to apply transfer learning even to quite small models. The alpha value is a multiplier which sets the filters in each layer, allowing MobileNetV2 models to scale way down.

&emsp;While training, we experimented with the tradeoffs between image size, RGB vs grayscale, and alpha parameters to see how to achieve the most accurate model that would fit into the crucial 4MB area. We first trained in Edge Impulse, but training for MobileNetV2 is quite limited as the free tier compute is limited to around 1 hour which came out to 30 epochs. We then transitioned to training locally. While we were able to experiment some more, we ran into issues with quantization, where we had trouble getting my quantized model to fit. We transitioned to uploading the trained int32 model to edge impulse and quantizing from there. We were able to push validation accuracy to almost 80%. We trained on three classes, positive-neutral-negative, as the goal was to push the accuracy.


## 6. Results Discussion
&emsp;The final CNN model was successfully deployed on the Cam H7 Plus with an average inference speed of ~5 FPS. Its TFLite model footprint contained ~160 KB RAM peak usage and ~160 KB flash, making it suitable for embedded deployment. Real-world testing demonstrated that it was highly accurate in predicting facial expressions across ALL 5 classes.
&emsp;The final MobileNetV2 training also successfully deployed on the Cam H7 Plus with an average inference speed of ~21 FPS, a big improvement over the final CNN model. Differentiation between positive, neutral, and negative facial expressions was demonstrated to be quite clean in real-world testing across all team members. Despite being 2287 KB in size for flash memory usage, the model is surprisingly fast, most likely due to the highly optimized and efficient architecture from the MobileNetV2's Inverted Residuals and Linear Bottlenecks.
&emsp;In conclusion, we successfully achieved our goals of 80%+ accuracy and >3 frames per second during inference for our final FER models. The final CNN achieved 83% in training, inferencing at 5 FPS. The final MobileNetV2 achieved almost 80% in validation, making up for its lower accuracy with a much faster 21 FPS during inference. Our future work includes defining set Cam H7 Plus testing standards for image lighting, contrast, and background so that we can better define a final accuracy for our models across different people
Additionally, we could explore hyperparameter optimizations for the learning rate, regularization strength, etc.---which for this project were mainly done using observation-based tweaks. This would allow us to achieve higher accuracy and faster inference times, all optimized to fit within the memory modules and computing capability of the OpenMV Cam H7 Plus.

## 7. Team Responsibilities

| Name | Role | Responsibilities |
|------|------|------------------|
| Maria Molchanova | Hardware | Setup, integration, model training |
| Jordan Ho | Software | Architecture research, model training, testing |
| Rohina Naqshbandi | Testing | Dataset research, model training |

## 10. References
https://openmv.io/products/openmv-cam-h7-plus 

https://docs.openmv.io/library/omv.tf.html 

https://docs.edgeimpulse.com/tutorials/end-to-end/image-classification 

https://www.edgeimpulse.com/blog/tensorflow-with-the-edge-impulse-python-sdk/ 

https://www.tensorflow.org/

https://github.com/vicksam/fer-model

https://github.com/7abushahla/Student-Engagement/blob/main/notebooks/train_and_quantize.ipynb

Asmara, R. A., Rosiani, U. D., Mentari, M., Syulistyo, A. R., Shoumi, M. N., & Astiningrum, M. (2024). An experimental study on deep learning technique implemented on low specification OpenMV CAM H7 device. JOIV International Journal on Informatics Visualization, 8(2), 1017. https://doi.org/10.62527/joiv.8.2.2299

Barsoum, E., Zhang, C., Ferrer, C. C., & Zhang, Z. (2016, September 24). Training Deep Networks for facial expression recognition with crowd-sourced label distribution. arXiv.org. https://doi.org/10.48550/arXiv.1608.01041 

Dukić, D., & Sovic Krzic, A. (2022). Real-Time Facial Expression Recognition Using Deep Learning with Application in the Active Classroom Environment. Electronics, 11(8), 1240. https://doi.org/10.3390/electronics11081240

## 🧾 Submissions
- Commit and push all deliverables before each deadline.
- Tag final submissions with:
   ```bash
   git tag v1.0-final
   git push origin v1.0-final
   ```

## 📜 License
This project is released under the MIT License.
