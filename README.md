**Shuffle Vision Transformer for Driver Emotion Recognition**
<br>
This repository hosts the official implementation of the paper:
"Shuffle Vision Transformer: Lightweight, Fast, and Efficient Recognition of Driver Facial Expressions."

**Dataset**
<br>
We use the KMU-FED dataset for training and evaluation.
You can find the dataset here: KMU-FED Dataset
![KMU-FED](https://user-images.githubusercontent.com/12345678/KMU-FED.png)



The dataset is preprocessed and saved in .h5 format using the script:
python KMU.py

**Data Augmentation**
<br>
To increase the diversity of the dataset, we apply the mirror trick as a data augmentation technique — enhancing model robustness by simulating left-right facial variations.

**Confusion Matrix**
<br>
To evaluate the model's classification performance to plot the confusion matrix


python confusion_matrix.py --model Ourmodel
This visualizes the predicted vs. actual expressions, helping assess class-wise accuracy.

**Grad-CAM Visualization**
<br>
We use Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the most influential regions in the input image that contribute to the model's predictions.

Grad-CAM helps us:

Interpret the model’s decisions

Understand which facial regions influence specific expression predictions

You can generate Grad-CAM visualizations using:

python GradCAm.py --GradCAM.jpg --Test_model.t7
