# LungCancerClassification
Developed a deep learning model that can accurately classify chest CT-Scan images into different categories representing lung cancer types (adenocarcinoma, large cell carcinoma, squamous cell carcinoma). Early and accurate detection of lung cancer can significantly improve patient outcomes and guide appropriate treatment strategies.
This project is a deep learning-based approach to classify lung cancer types in CT scan images using the AlexNet model. The goal of the project is to assist in the early detection and classification of different types of lung cancer, which can aid in providing more personalized and effective treatments for patients.

Here is a breakdown of the different components in the project:

1. **Data Preparation**: The project utilizes a dataset containing CT scan images of the chest, specifically focused on lung cancer cases. The dataset is divided into training, validation, and test sets. Data augmentation techniques are applied to the training set to increase the diversity of the data and improve the model's generalization.

2. **Model Architecture**: The model architecture used in this project is AlexNet. AlexNet is a classic convolutional neural network (CNN) architecture that gained significant attention after winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It consists of several convolutional and fully connected layers, making it well-suited for image classification tasks.

3. **Transfer Learning**: The AlexNet model is loaded with pre-trained weights. Pre-trained weights are learned from a large dataset (e.g., ImageNet) and then used as an initialization for the lung cancer classification task. This approach allows the model to leverage knowledge from ImageNet for feature extraction and adapt it to the new lung cancer dataset.

4. **Training and Evaluation**: The model is trained on the training data, and the validation set is used to monitor its performance during training. Several metrics are used for evaluation, such as accuracy, precision, recall, and F1 score. The model is evaluated on the test set to assess its performance on unseen data.

5. **Class Weighting**: Since the dataset may be imbalanced (i.e., some lung cancer types may have fewer samples than others), class weights are computed and applied during training. Class weights help in giving more importance to underrepresented classes, preventing the model from being biased towards the majority class.

6. **Prediction**: After the model is trained and evaluated, it is used to make predictions on new CT scan images. The user can upload an image, which is then preprocessed, fed into the trained model, and classified into one of the four lung cancer classes: adenocarcinoma, large cell carcinoma, normal, or squamous cell carcinoma.

Overall, this project demonstrates the use of transfer learning to leverage pre-trained models for medical imaging tasks and highlights the importance of proper evaluation metrics and class weighting to deal with imbalanced datasets in real-world applications.
