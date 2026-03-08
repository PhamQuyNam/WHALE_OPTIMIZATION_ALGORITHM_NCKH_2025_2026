# WHALE_OPTIMIZATION_ALGORITHM_NCKH
# Movie Recommendation Optimization using Whale Optimization Algorithm

## Overview

This project presents a research implementation of a movie recommendation system optimized using the Whale Optimization Algorithm (WOA).  
The goal is to improve the prediction accuracy of recommendation models by automatically optimizing the hyperparameters of Matrix Factorization.

Traditional recommender systems often rely on manual tuning or exhaustive search methods such as Grid Search, which are computationally expensive and prone to local optima. To address this limitation, this research integrates Whale Optimization Algorithm (WOA) into the training process of Matrix Factorization to efficiently search for optimal hyperparameters.

This repository contains the implementation used in our research paper on improving movie recommendation accuracy.

---

# Research Objective

The main objectives of this research are:

- Improve the prediction accuracy of movie recommendation systems.
- Automatically optimize hyperparameters of Matrix Factorization models.
- Compare the performance of WOA with traditional optimization methods.
- Evaluate recommendation quality using multiple metrics.

---

# Proposed Method

The proposed framework integrates **Matrix Factorization (MF)** with the **Whale Optimization Algorithm (WOA)** to optimize model parameters.

## System Workflow

## Optimization Strategy

In the proposed method:

- Each **whale** represents a candidate solution containing a set of MF hyperparameters.
- The **fitness function** is defined as the RMSE on the validation dataset.
- The algorithm iteratively updates whale positions to search for the optimal parameter configuration.

Optimized parameters include:

- Learning rate
- Regularization coefficient
- Number of latent factors

---

# Dataset

This project uses the widely adopted benchmark dataset:

- MovieLens 100K
- MovieLens 1M

Dataset characteristics:

- User–movie rating matrix
- Sparse interaction data
- Explicit ratings (1–5 scale)

---

# Evaluation Metrics

To evaluate the performance of the recommendation system, the following metrics are used:

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- NDCG@K

These metrics measure both prediction accuracy and recommendation quality.

---

# Baseline Models

To validate the effectiveness of the proposed method, we compare the following approaches:

1. Matrix Factorization with manual tuning  
2. Matrix Factorization with Grid Search  
3. Matrix Factorization optimized by Particle Swarm Optimization (PSO)  
4. Matrix Factorization optimized by Whale Optimization Algorithm (WOA) – Proposed Method

---

# Experimental Setup

Typical experimental configuration:

- Train/Test split: **80% / 20%**
- Population size (WOA): 20–30
- Maximum iterations: 30–50
- Random seed fixed for reproducibility

---

# Expected Results

The proposed **MF-WOA model** is expected to:

- Achieve lower RMSE and MAE compared with baseline methods
- Converge faster than traditional search methods
- Provide more stable hyperparameter optimization

---

# Project Structure

---

# Future Work

Future improvements may include:

- Adaptive Whale Optimization Algorithm
- Hybrid recommender systems
- Integration with deep learning models
- Application to large-scale streaming platforms

---

# Citation

If you use this work in your research, please cite our paper:
