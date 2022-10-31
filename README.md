# ISIC_2019_Benchmark

This Project aims to benchmark various commonly used models for the task of classifying dermoscopic images among nine different diagnostic categories using the ISIC 2019 Challenge dataset

- Melanoma
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis
- Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
- Dermatofibroma
- Vascular lesion
- Squamous cell carcinoma
- None of the others

The dataset has 25,331 images available for training across eight different categories of diseases. But the model must identify the 9th condition where none of the eight categories are formed.

Distribution-

![image](https://user-images.githubusercontent.com/72119231/198967851-37998036-2a20-4815-915a-986853c13c60.png)


We have used various backbones followed by a fully connected layer and a sigmoid layer to get predictions for eight classes. If no types are found, all classes' predictions should be zero.

We have used vgg19,resnet50, densenet121, and vit_32_b as backbones, bceloss, and ASL loss as two variants of loss functions, Adam optimizer with learning rate scheduler and mean of classwise AUC-roc score and accuracy for evaluation.


## Pre-Processing: 
It is a mandatory task to be done on the images acquired using dermoscopy. It is required because the captured image may not be clear in resolution. Human skin surface will be accumulated by hairs, scars and skin tone differences. Hence, the images should be preprocessed to access the affected skin lesion accurately. An image can be preprocessed in multiple ways. They are: Hair removal, black frame removal, circle removal, and more techniques which are narrated as below:
### Image Resizing
The resolution of the dermoscopic images is usually very high. So, to reduce the computational complexity, the input images are scaled down to resize it to size 224x224.
### Hair Removal
The RGB image is converted to a grayscale image. 
After conversion, the contrast enhanced image is constructed using the contrast-limited adaptive histogram equalization method.
CLAHE is the upgraded version of adaptive histogram equalization. In CLAHE, the image is split up into small blocks called tiles. Histogram equalization is applied to each of these. To avoid amplification of noise, contrast limiting is applied. Finally, the neighboring tiles are merged using bilinear interpolation to remove artefacts in the boundaries. The contrast enhanced L channel is then merged to form a Lab image and then converted into an RGB image.
After computing the contrast enhanced image, the averaged image is constructed by applying the average filter. 
To create the initial hair mask, the contrast and average are subtracted. 
The hair mask is converted to a binary image by applying the thresholding method. Here, pixels within a defined range are selected as foreground, and pixels outside the range are selected as background.
The resulting thresholded image has small objects which are removed by applying the morphological opening operation. Using morphological opening operation, all the connected components that have fewer than “p” pixels are removed from the binary image, i.e. from the thresholded image.The value of p chosen is 50.
To achieve a hair free image, the bi-harmonic inpainting technique is used. 
### Segmentation
The main purpose of a segmentation step is to obtain the region of interest (ROI). The ROI is expected to have relevant information in the form of different features that can be used for lesion classification and diagnosis. 
The image is first converted to a grayscale image. Following this, we obtain the elevation map of the given image by calculating the sobel gradients on the grayscale image.
We create a binary image from the gray scale image via thresholding. We apply the watershed algorithm, fill the binary holes to create an intermediate image. We then remove the small objects the are left over in the intermediate stage via the use of morphological operations. Finally, we extract the foreground and leave the background to get the final image.

## Evaluation-Metrics:
 - F1 score
 - Sensitivity
 - Specificity
 - Precision
 - Recall
 - Balanced Accuracy
 - Average AUC across all diagnoses


### Project Objectives:
To work on multi class classification of diagnostic categories of Skin Lesion in Dermoscopic-Images

 - Solve the class-imbalance problem to achieve better mean-AUC.
 - Implement various baseline methods based on Literature Review.
 - Implement attention based CNN model to focus on lesion for correct prediction
 - Implement Weighted Cost function and Focal Loss to solve the issue of Class Imbalance.
 - Implement XAI method - gradcam to visualize features based on which model is predicting.

### Models Shortlisted:
#### Baseline Models:
 - VGG-19 (Ref: https://arxiv.org/abs/1409.1556 )
 - ResNet50 (Ref: https://arxiv.org/abs/1512.03385 )
 - DenseNet-121(Ref: https://arxiv.org/abs/1608.06993 )
 - VIT-32b (Ref: https://arxiv.org/abs/2010.11929 )

The models have been shortlisted after reviewing the papers:
 - 2102.01284.pdf (arxiv.org)
 - https://www.sciencedirect.com/science/article/pii/S1361841521003509 

##### VGG-19
Ref: https://arxiv.org/abs/1409.1556 

Pros:

Increases the number of layers of the model.
Reduces the kernel size
As a consequence of the previous 2, non linearity has been introduced into the model.

Cons:

Vanishing and Exploding gradients.
Slower than Renets which were introduced at the same time.
##### ResNet50
Ref: https://arxiv.org/abs/1512.03385 

Pros:

With residual blocks, one can construct networks of any depth with the hypothesis that new layers are actually helping to learn new underlying patterns in the input data

Cons:

Vanishing and exploding gradients still persist in this model.
##### DenseNet-121
Ref: https://arxiv.org/abs/1608.06993

Pros:

By connecting this way DenseNets require fewer parameters than an equivalent traditional CNN, as there is no need to learn redundant feature maps
DenseNets layers are very narrow (e.g. 12 filters), and they just add a small set of new feature-maps.
Another problem with very deep networks was the problems to train, because of the mentioned flow of information and gradients. DenseNets solve this issue since each layer has direct access to the gradients from the loss function and the original input image.

Cons:

Compared with ResNet, DenseNet uses a lot more memory, as the tensors from different are concatenated together.
The disadvantage of DenseNet is that the feature maps of each layer are spliced with the previous layer, and the data is replicated multiple times. As the number of network layers increases, the number of model parameters grows linearly, eventually leading to explosive growth in computation and memory overhead during training.

##### VIT32B
Ref: https://arxiv.org/abs/2010.11929

Pros:

The core mechanism behind the Transformer architecture is Self-Attention. It gives the capability to understand the connection between inputs.
The model calculates self-attention for all the pixels in the image with each other. It segments images into small patches (like 16x16) as the atom of an image instead of a pixel to efficiently tease out patterns.It has Multi-Head Attention. 
The model is able to encode the distance of patches in the similarity of position embeddings.It integrates information across the entire image even in the lowest layers in Transformers.

Cons:

Cannot be fine-tuned with a small number of images available in task-specific datasets. 
Information learned cannot be transferred as the model trained for one task does not adapt well to other related tasks.
Difficult to interpret visual transformer models
### Loss Function - 
#### FocalLoss -
(Ref: https://arxiv.org/abs/1708.02002 )

A Focal Loss function addresses class imbalance during training in tasks like object detection. Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified examples. It is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases. Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples.
#### Model Structure:
After running the baseline models, we will use them as backbones for our ensemble based architecture. The ensemble methods that will be experimented with are:
Mean (Average Voting)


## Results


![image](https://user-images.githubusercontent.com/72119231/198964834-3460a219-106c-4317-9ce7-74192e82c40e.png)


#### Confusion Matrix of DenseNet-121 - Hair Preprocessed
![image](https://user-images.githubusercontent.com/72119231/198966827-93650e30-fdaf-4d9f-ba02-84c5bf762d09.png)


#### Gradcam++ XAI of DenseNet-121 - Hair Preprocessed
![image](https://user-images.githubusercontent.com/72119231/198967092-5055a2a5-607f-4c16-8990-3e5daf83b72c.png)



#### Confusion Matrix of DenseNet-121 - Hair+Segmentation Preprocessed
![image](https://user-images.githubusercontent.com/72119231/198966934-6d9dba95-a293-4006-8734-5e43686afb0d.png)




#### Gradcam++ XAI of DenseNet-121 - Hair+Segmentation Preprocessed
![image](https://user-images.githubusercontent.com/72119231/198967140-eb612899-6cec-4905-ac36-5cd76f37b16d.png)


## Analysis:
We tried out various baselines models along with an ensemble of the baselines and a combination of different baselines using focal-loss as the loss function for optimizing our models .

Among individual models DenseNet121 seems to be the best based on the metrics followed by ResNet50.

VGG19 seems to be the worst performer with metrics showing the model was unable to capture essential features

We hoped using focal loss will help us in overcoming the effect of class-imbalance in the dataset however , we were unable to fine-tune focal-loss properly leading to poor results.

This is highlighted with the results of VIT as it was expected to give best results but is unable to stand on expectations.

We tried out Ensemble(using Averaging of Predictions) various baseline models with a model of the same type and multiple models of different type.

Among Same Models Ensemble on models trained on different seed the same pattern continues as in baseline models with ResNet50 and DenseNet121 leading followed by VIT32b and last being VGG-19.

Ensembles of Multiple DIfferent Models perform decently as compared to baselines.

## Conclusion:
We tried to solve the problem of multi-class classification of skin lesions using focal loss as a methodology to optimize models to mitigate the effect of class imbalance . However there seems to be some issue in the selection of Hyper-parameters in focal loss functions leading to non-optimization of all the models.

## Future Works
We would have to use NAS and try out various hyper-parameters using some algorithms to optimize focal-loss and optimize all the models. Alongside use XAI to see the models learning to evaluate qualitatively the learning of model.









<br>
<hr>









## Misc

To evaluate the robustness and generalizability of the models, we have used the DDI dataset — the first publicly available, expertly curated, and pathologically confirmed image dataset with diverse skin tones. The motivation behind the creation of this dataset was that it was found that most dermatology datasets lack essential information about dataset diversity and have noisy diagnostic labels. Currently, publicly available datasets lack biopsy-proven skin lesions in dark skin tones.

Since there was a difference in labels for DDI and ISIC19, we did manual grouping under the advice of an expert to create a version of DDI with eight classes.

To explain our model predictions, we have used GradCam++ to create class-activation maps to see models focusing on various categories.



About ISIC - The International Skin Imaging Collaboration (ISIC)  is an international effort to improve melanoma diagnosis, sponsored by the International Society for Digital Imaging of the Skin (ISDIS). The ISIC Archive  contains the most extensive publicly available collection of quality-controlled dermoscopic images of skin lesions.

About DDI Dataset - Diverse Dermatology Images (DDI) dataset provides a publicly accessible dataset with diverse skin tones and pathologically confirmed diagnoses for AI development and testing. DDI enables model evaluation with stratification between light skin tones (Fitzpatrick skin types (FST) I-II) and dark skin tones (FST V-VI) and includes pathologically confirmed uncommon diseases (with incidence less than 1 in 10,000), which are usually lacking in AI datasets. The authors used this dataset to demonstrate three critical issues for AI algorithms developed for detecting cutaneous malignancies: 1) significant drop-off in performance of AI algorithms developed from previously described data when benchmarked on DDI 2) skin tone and rarity of disease as contributors to performance drop-off in previously described algorithms and 3) the inability of state-of-the-art robust training methods to correct these biases without diverse training data.


RESULTS-
Overall Results in both ISIC19 and DDI Dataset-

![image](https://user-images.githubusercontent.com/72119231/181760598-88ac93a9-6b61-4b84-8057-56ec3ed8f5dd.png)

Classwise ROC-AUC Score for all models-

![image](https://user-images.githubusercontent.com/72119231/181761302-64607e97-7403-431b-8e3c-790c5882dd94.png)


Classwise Accuracy Score for all models-

![image](https://user-images.githubusercontent.com/72119231/181761823-2177bee9-0aae-46a4-887d-cd887af26fb4.png)


CAM for Explainibilty of model predictions-
![Page 1](https://user-images.githubusercontent.com/72119231/181905501-02ad8a8b-0418-4f9c-bdfc-56396a73d84a.png)

![Copy of Page 1](https://user-images.githubusercontent.com/72119231/181905504-bd9ec1dc-04ff-4630-83a1-90cc4fe6156a.png)
