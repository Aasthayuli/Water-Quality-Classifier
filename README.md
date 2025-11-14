# 💧 Water Quality Classifier using CNN

Welcome to my internship project repository!
This project focuses on classifying water quality images into three categories — Clean, Muddy, and Polluted — using Deep Learning (CNN).
The goal was to build a full pipeline: data preprocessing, model design, optimization, and deployment through a Streamlit web app.

---

## 📅 WEEK 1 – Data Collection & Preprocessing

**Objective**: Prepare a structured dataset and clean input for model training.

**Dataset Link:** [Click here to download the dataset](https://drive.google.com/drive/folders/1W5GTIHEfQVhZsacwTnReRYffUiE4D2RU?usp=sharing)

**_Tasks_**:

- Collected water-quality images across three classes: Clean, Muddy, and Polluted.
- Organized dataset into training, testing, and validation folders.
- Performed image preprocessing:

  - Resized images for uniformity.
  - Applied normalization and data augmentation (rotation, flipping, brightness).

- Visualized sample images from each class for quality check.
- Created efficient DataLoader pipelines using PyTorch.

**_Deliverables:_**

- data_preprocessing.py
- Folder structure: train/, test/
- Preprocessed dataset ready for model training.

---

## ⚙️ WEEK 2 – Model Design & Training

**_Objective_**: Build and train a CNN model to classify water images effectively.

**_Tasks_**:

- Designed a Convolutional Neural Network (CNN) architecture using PyTorch.
- Defined:
  - Loss Function: CrossEntropyLoss
  - Optimizer: Adam
  - Trained the model across multiple epochs using training and validation datasets.
  - Monitored model performance via accuracy and loss curves.
  - Saved trained model weights as water_quality_classifier.pth.

**_Deliverables:_**

- water_cnn_model.py
- Saved model: water_quality_classifier.pth
- Training accuracy and loss graphs.

---

## 🚀 WEEK 3 – Model Optimization & Streamlit Deployment(ongoing)

**Objective**: Improve model performance and deploy it as a user-friendly app.

**_Tasks_**:

- Tuned hyperparameters (batch size, learning rate) for better generalization.
- Added Batch Normalization and Dropout layers.
- Evaluated performance using:
- Confusion Matrix
- Precision, Recall, and F1-score
- Built a Streamlit-based Web App that:
- Lets users upload water images.
- Displays prediction results.
- Final deployment integrates the model backend with a simple front-end UI.

**_Deliverables:_**

- optimized_model_and_visualization.py
- app.py – Streamlit app file
- Final optimized model (water_quality_classifier.pth)

---

### 🔗 Trained Model

Download [water_quality_classifier.pth](https://drive.google.com/file/d/1pXSDJB2EfMUPfJeeNkQcAKc2YEE3jDeh/view?usp=sharing)

---

## 🧠 Tech Stack

- Programming Language: Python
- Libraries: PyTorch, torchvision, NumPy, Matplotlib, Streamlit
- Tools: Jupyter Notebook, VS Code
- Version Control: Git & GitHub

---

## 🎯 Project Highlights

- Built a complete AI pipeline from data to deployment.
- Used CNN for image classification with custom dataset.
- Deployed an interactive web app for real-time predictions.
- Strong focus on clean code, reproducibility, and explainability.

---

## 📌 Repository Structure

Water_Quality_Classifier/

│

├── data/

│ └── water_dataset/

│ ├── train/

│ └── test/

│

├── week1_data_preprocessing/

│ ├── data_preprocessing.py

│

├── week2_model_training/

│ ├── water_cnn_model.py

│ ├── train_model.py

│

├── week3_model_optimization/

│ ├── app.py

│ ├── evaluated_model.py

│ ├── visualize_results.py

│

├── outputs/

│ ├── graphs/

│ ├── logs/

│ └── predictions/

│

├── requirements.txt

├── README.md

├── .gitignore

└── Problem_Statement.pdf

└── water_quality_classifier.ipynb

> **Note:**
>
> - `outputs/` stores model visualizations, logs, and predictions.
> - `app.py` runs the Streamlit-based user interface for the final optimized model.
> - `water_quality_classifier.ipynb` is the Jupyter Notebook file for research, experimentation and rapid prototyping.

---

## 🏁 Conclusion

This internship project enhanced my understanding of Deep Learning, Computer Vision, and Model Deployment.

It provided a hands-on experience with real-world data, model optimization techniques, and end-to-end integration using Streamlit.
