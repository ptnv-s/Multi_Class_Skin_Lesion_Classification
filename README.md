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

We have used various backbones followed by a fully connected layer and a sigmoid layer to get predictions for eight classes. If no types are found, all classes' predictions should be zero.

We have used resnet18, densenet121, and vit_32_b as backbones, bceloss, and ASL loss as two variants of loss functions, Adam optimizer with learning rate scheduler and mean of classwise AUC-roc score and accuracy for evaluation.

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
