Facial Emotion Recognition (FER2013)


This project implements a Hybrid CNN–Transformer model for facial emotion recognition using the FER2013 dataset.
The model combines local feature extraction from CNNs with global contextual understanding from Vision Transformers.
 
Model Architecture
The proposed model uses a hybrid architecture:
•	ResNet18 (CNN backbone)
Extracts local facial features such as edges, textures, and muscle movements.
•	Vision Transformer (ViT-Base/16)
Captures global spatial relationships across the face using self-attention.
•	Feature Fusion Layer
CNN and ViT features are projected into a shared embedding space and concatenated.
•	Classification Head
A fully connected neural network predicts one of the seven facial emotions.
Emotion classes:
angry, disgust, fear, happy, neutral, sad, surprise
 
Dataset Description
The model is trained and evaluated on the FER2013 dataset, a widely used benchmark for facial emotion recognition.
Dataset characteristics:
•	35,887 grayscale facial images
•	48 × 48 resolution
•	7 emotion classes
Images are converted to 3 channels to match pretrained CNN and ViT models.

Dataset structure used in this project:

FER2013
 ├── train
 │    ├── angry
 │    ├── disgust
 │    ├── fear
 │    ├── happy
 │    ├── neutral
 │    ├── sad
 │    └── surprise
 └── test
      ├── angry
      ├── disgust
      ├── fear
      ├── happy
      ├── neutral
      ├── sad
      └── surprise
 
Training Improvements
The following training strategies were used to improve performance:
•	Label Smoothing – reduces overconfidence and handles noisy labels
•	Mixup Augmentation – improves generalization by blending training samples
•	Class-Weighted Loss – compensates for class imbalance
•	Cosine Learning Rate Scheduler – stable training convergence
•	Test-Time Augmentation (TTA) – improves final prediction robustness
 
Final Performance (FER2013)
Metric	Score
Test Accuracy	71.48%
Macro F1 Score	69.96%
Evaluation includes Test-Time Augmentation (TTA).
 
How to Run the Code
1. Install Dependencies
pip install torch torchvision timm scikit-learn
 
2. Download FER2013 Dataset
Dataset can be obtained from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013
Place the dataset in the following structure:
dataset/
   train/
   test/
 
3. Train the Model
Run the training script:
python train.py
Training outputs:
•	best model checkpoint
•	training logs
•	validation metrics
 
4. Evaluate the Model
After training, the script automatically reports:
•	Test Accuracy
•	Macro F1 Score
•	Confusion Matrix
•	Classification Report 
