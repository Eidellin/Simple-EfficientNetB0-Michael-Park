# Simple-EfficientNetB0-Michael-Park

Version 0

## Project Title: Kaggle: Penguins vs Turtles object detection and classification using computer vision technologies 
Object detection and classification are the tasks of determining an object’s location and classifying it in an image with a predicted bounding box and label. In the field of computer vision, models, methods and strategies developed to solve this task are rapidly evolving. For many applications, the aim is to develop and evaluate pre-existing methods for the accurate and efficient detection and classification of objects in a dataset. The Penguins versus Turtles dataset available from Kaggle consists of a training set of 500 images and a validation set of 72 images containing either a penguin or a turtle. Our task was to localise and classify images as penguins or turtles by constructing a predicted bounding box and label. The purpose of this task was to evaluate and develop methods that can analyse the images of the dataset accurately and efficiently. In literature there exist many models and techniques to solve this task.

Two distinguishing paradigms of object classification models are based on the selection of machine learning or deep learning algorithms. The support vector machine (SVM) is a machine learning algorithm that is well-established in industry and is capable of respectable performance. Manual feature engineering is a must to obtain high model efficiency and accuracy in machine learning models [1]. Two important tasks in feature engineering are hyperparameter tuning and feature extraction. In this report, we evaluate the implemented Grid Search algorithm used to find optimal hyperparameters and a histogram of oriented gradients (HOG) feature extractor. On the other hand, convolutional neural networks (CNNs) such as EfficentNetB0 and Faster R-CNN take advantage of deep-learning to classify objects. The benefit of these deep learning frameworks is their capability to extract complex features from images automatically with little feature engineering required [1]. 

For object localisation, generating a set of bounding box proposals called anchor boxes is a fundamental approach. For this task, we developed a modified grid-based anchor box generation model. Combining this with class prediction probabilities from our SVM and EfficientNetB0 classification models, we adopted the non-maximum suppression (NMS) algorithm to produce the predicted bounding box. 

The objectives of this report are: 1)  to review our selected object detection and classification techniques in literature, 2) explain and motivate the selection of methods implemented, 3) explain the experimental setup used to evaluate the performance of developed methods and report the results obtained, 4) provide a discussion of results and method performance, and 5) recommend future work.  

\- from my team report

My model was EfficientNetB0. This is a simplified version and not perfectly implemented.

## EfficientNetB0

EfficientNet was chosen because it is known for its high efficiency and performance. The EfficientNet models were introduced by Mingxing Tan and Quoc V. Le in EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (2019). The paper demonstrated that EfficientNet models can achieve similar or sometimes better performance than existing CNN models with significantly fewer parameters and FLOPS. We selected the EfficientNet-B0 model which is the baseline model of the EfficientNet family to evaluate. This was achieved by creating a simplified version of the architecture to evaluate the performance of the model with our grid-based anchor boxes for object localisation. 

The unique aspect of EfficientNet is the use of a new compound scaling method. This network introduced a method to scale up all three dimensions of the network (depth, width and resolution) based on a compound coefficient by adding more layers, adding more neurons to existing layers and increasing input image size. The coefficient is found by a Grid Search that seeks the optimal relationships between three dimensions under a fixed resource constraint, helping to maintain a balance and ensuring that each aspect is being scaled proportionally. Hence, the architecture is more resource-efficient.

The model was developed by scaling down the baseline network to make it as small and efficient as possible. Especially when computational resources or power is limited, this model is a strong choice due to its efficient use of computational resources and strong performance.

The Cross Entropy loss and the Lion optimizer have been chosen to improve its performance. The Cross Entropy loss is a famous loss function for multi-class classification and the Lion optimizer is one of the famous modern optimizers especially in the Computer Vision field.

## Results

### Localisation Performance

| Model           | Mean IoU | SD IoU | Mean Distance (px) | SD Distance (px) |
|------------------|----------|--------|---------------------|-------------------|
| EfficientNetB0  | 0.030    | 1.275  | 289.97              | 129.67            |
| SVM             | -0.258   | 1.930  | 207.80              | 123.34            |
| Faster R-CNN    | 0.89     | 0.79   | 12.3                | 11.6              |

### Classification Performance

| Model           | Accuracy | Precision           | Recall             | F1-Score          |
|------------------|----------|---------------------|---------------------|-------------------|
| EfficientNetB0  | 64%      | 57%                 | 81%                | 68%              |
| SVM             | 75%      | 71%                 | 83%                | 77%              |
| Faster R-CNN    | 94%      | Penguins: 100%      | Penguins: 89%      | Penguins: 94%    |
|                  |          | Turtles: 90%       | Turtles: 100%      | Turtles: 95%     |

