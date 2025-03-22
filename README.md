# Wine Quality Prediction using Machine Learning

This project leverages machine learning to predict the quality of wine based on its chemical composition. Using features such as alcohol content, volatile acidity, residual sugar, pH, and more, the goal is to determine whether a wine is of **high quality** (score of 7 or higher) or **low quality** (score below 7). The application utilizes a **Random Forest Classifier** to make predictions based on these features.

The project provides an interactive **web application** using **Streamlit**, allowing users to input wine features and get predictions in real-time. In addition, the app visualizes various aspects of the data to provide deeper insights into the relationship between different chemical properties and wine quality.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [How to Run the Application](#how-to-run-the-application)
7. [Features](#features)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

The **Wine Quality Prediction** project utilizes machine learning to predict the quality of wines. The dataset consists of **1599 wine samples**, each with 11 chemical attributes that help in determining its quality. 

The goal of the project is to:
- Predict whether the wine is of high quality (score of 7 or higher).
- Provide insights into how different chemical properties of the wine affect its quality.
- Implement a **Random Forest Classifier** to make predictions based on these attributes.
- Allow users to interact with a web application that makes it easy to input wine attributes and get a prediction of its quality.

---

## Technologies Used

This project employs various Python libraries and tools:

- **Python 3**: The primary programming language for implementing machine learning models and the Streamlit application.
- **Streamlit**: A fast and easy framework to create web applications for data science.
- **Scikit-learn**: A library used for building and training the Random Forest Classifier model.
- **Pandas**: A powerful data analysis library used for handling and manipulating the dataset.
- **NumPy**: Used for efficient numerical operations, particularly when handling arrays and performing mathematical calculations.
- **Plotly**: A graphing library used to create interactive visualizations for the data.
- **SHAP**: A library for model explainability and understanding the impact of features on the prediction.
- **Kaggle API**: Utilized to download the **Wine Quality Dataset** from Kaggle.
  
---

## Dataset

The **Wine Quality Dataset** used in this project can be found on [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset). It consists of **1599 rows** and **12 columns** that describe the properties of wine. Each sample corresponds to a particular wine, and the dataset includes the following columns:

### Features:
- **Fixed acidity**: The fixed acids in wine, such as tartaric, malic, citric, and others.
- **Volatile acidity**: The amount of acetic acid in the wine, which can affect its taste.
- **Citric acid**: A naturally occurring organic acid found in grapes and wine.
- **Residual sugar**: The amount of sugar left in the wine after fermentation.
- **Chlorides**: The amount of salt in the wine.
- **Free sulfur dioxide**: Sulfur dioxide that is not bound to other compounds and helps prevent oxidation.
- **Total sulfur dioxide**: Total amount of sulfur dioxide in the wine.
- **Density**: The density of the wine.
- **pH**: The acidity of the wine.
- **Sulphates**: The concentration of sulfur-based preservatives in the wine.
- **Alcohol**: The alcohol content of the wine.
- **Quality**: The target variable representing the quality score of the wine, ranging from 0 to 10.

---

## Setup and Installation

Follow the steps below to set up the project locally.

### 1. **Clone the Repository**

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/ngoubimaximillian12/Wine-Quality-Prediction-using-Machine-Learning.git
