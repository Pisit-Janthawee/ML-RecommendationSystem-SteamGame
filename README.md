# ML-RecommendationSystem-SteamGame

# **Steam game Recommendation System - Content-Based**

<img src="https://github.com/Pisit-Janthawee/ML-RecommendationSystem-SteamGame/blob/main/Deploy.png" align="center">

## **Machine Learning Steps**

1. **Data Collection and Preprocessing**:

- 1.1 Import dataset
- 1.2 Merge data frames
- 1.3 Normalize text data

2. **Feature Extraction**:

- 2.1 TF-IDF Vectorization
- 2.2 Count Vectorization

3. **Exploratory Data Analysis (EDA)**:

- 3.1 Conduct descriptive analysis
- 3.2 Identify data patterns and trends

4. **Modeling**:

- 4.1 Similarity Calculation:
  - 4.1.1 Linear Kernel for similarity measurement
  - 4.1.2 Cosine Similarity for similarity measurement
- 4.2 Benchmark
- 4.3 Pipeline

5. **Recommendation Algorithm**:

- Develop a content-based recommendation system based on analyzed data

6. **Deployment** (Gradio)

Now, let's dive into each of these steps in detail.

## Dataset (Kaggle)

- Combined data of 27,000 games scraped from Steam and SteamSpy APIs
- Source: https://www.kaggle.com/datasets/nikdavis/steam-store-games/data

## Overview & Objective:

**Overview**:
In this project, we will be working with a dataset containing information about different Steam games, including their attributes such as name, developer, genres, category, steam tags, and description of the game. By utilizing natural language processing (NLP) techniques to normalize these text data for feeding to Vectorizer the result can be used to find similarity values. and recommend!

**Objective**:
to suggest games based on their similarities with the user's preferences.

- **Business Type**: Gaming and Entertainment

- **Business Objective**:

  - Enhance user engagement and satisfaction within the gaming community. By providing personalized game recommendations,
  - Aim is to increase user retention, drive game exploration, and ultimately contribute to increased game sales and customer loyalty.

- **Learning Problems**: Item similarity

- **Reason for Choosing this Project**: Recommendation systems that can help users navigate through the extensive game options.

- **Expected Result**: similarity(game attributes + users' interests/user's preferences) = match score

- **Utilization of Results**: recommendation system will be integrated into the Steam platform to provide users with personalized game suggestions directly on their dashboard.

- **Benefits of this Project**:

  - Improved user experience through personalized game recommendations
  - Increased user engagement and retention within the Steam platform
  - Enhanced game exploration, leading to increased game sales and user satisfaction
  - Efficient utilization of the vast library of Steam games, ensuring that users find relevant and enjoyable content tailored to their preferences
  - Insights into user preferences and trends that can inform future game development and marketing strategies within the gaming industry.

  -

# File Description

## Folder

1. **repository**
   - _Explanation_: This folder contains raw data, such as steam games, steam description, steam media, etc.

## 01-02 .ipynb Files

1. **01_init_notebook.ipynb**
   - _Explanation_: This initial notebook is used for exploring the data and performing data preprocessing tasks as outlined in the "Data Preprocessing" section. Also, perform Vectorizer to feed to model or calculate Similarity measurements, and recommend the similar items
2. **02_deployment.ipynb**
   - _Explanation_: This notebook focuses on model deployment using Gradio. It provides an easy-to-use interface for displaying input Text and similar items

## .py Files

1. pipeline.py

   - _Explanation_: The file contains functions and classes that define the workflow of the data processing pipeline.

2. recommendation.py

   - _Explanation_: Class of Recommendation system's functionality, find the closet name and recommend it!

3. text_normalization.py
   - _Explanation_: Class of text preprocessing and normalization tasks. It contains functions and utilities that handle tasks such as tokenization, stemming, lemmatization, and other text-cleaning operations. for to feed text-based models and algorithms.
