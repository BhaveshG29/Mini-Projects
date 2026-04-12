
# TF-IDF + Bag of Words Sentiment Analysis (From Scratch)

## Overview

This project is a learning-focused NLP implementation that builds a classical sentiment analysis pipeline from first principles. The main emphasis is on implementing **Bag of Words (BoW)** and **TF-IDF** manually instead of relying on high-level libraries.

The objective was to understand how raw text is converted into numerical features and how those features can be used by a neural network classifier.

## Core Features

-   Custom text preprocessing pipeline
    
-   Manual vocabulary creation
    
-   Custom **Bag of Words** matrix generation
    
-   Custom **TF computation**
    
-   Custom **IDF computation**
    
-   Custom **TF-IDF encoding**
    
-   Support for transforming new/custom text using learned vocabulary + IDF
    
-   Sparse matrix usage for memory efficiency during vectorization
    
-   Custom built **2 Hidden Layer Feedforward Neural Network (FNN)**
    
-   Custom built **RMSProp optimizer**
    
-   Sentiment prediction on unseen reviews
    

## Dataset

This project uses the **IMDb Dataset of 50K Movie Reviews**.

Source:  
[https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)

## Model Architecture

After converting reviews into TF-IDF vectors, the encoded features are passed into a fully connected neural network implemented from scratch.

Architecture:

-   Input Layer = TF-IDF feature size
    
-   Hidden Layer 1
    
-   Hidden Layer 2
    
-   Output Layer (Binary Sentiment Classification)
    

Optimizer used:

-   RMSProp (implemented from scratch)
    

## Why This Project?

Most NLP tutorials use ready-made tools such as:

-   `TfidfVectorizer`
    
-   `CountVectorizer`
    
-   High-level deep learning frameworks
    

This project intentionally avoids that path for the feature engineering stage, so the internal mechanics of text vectorization become clear.

## Learning Goals

This notebook helps practice:

-   Tokenization
    
-   Vocabulary indexing
    
-   Sparse feature representation
    
-   Frequency statistics
    
-   TF-IDF weighting
    
-   Neural network fundamentals
    
-   Optimizer mechanics
    
-   End-to-end NLP workflow
    

## Test Data Note

In addition to the IMDb dataset split, a separate custom test set of reviews was created using an LLM to evaluate predictions on fresh examples.

## Project Structure

-   `Main.ipynb` — Full notebook implementation
    
-   `cache/tfidf_matrix.npz` — Saved sparse TF-IDF matrix cache
    
-   `data/IMDb_Dataset.csv` — Raw IMDb dataset
    

## Limitations

This is an educational project, not a production NLP system. It prioritizes clarity and understanding over maximum speed, scalability, and framework-level abstractions.

## Future Improvements

-   Add n-grams
    
-   Compare with Logistic Regression / Naive Bayes / SVM
    
-   Model serialization
    
-   Better preprocessing pipeline
    
-   Benchmark against sklearn TF-IDF
    
-   Replace BoW features with embeddings
    

## Summary

A complete sentiment analysis pipeline built for learning purposes, with the main focus on **custom from-scratch TF-IDF + Bag of Words**, followed by a custom neural network classifier and optimizer.
