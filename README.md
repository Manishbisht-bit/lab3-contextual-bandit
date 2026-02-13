# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Manish Bisht  
**Roll Number:** U20230021  

---

## 1. Introduction

This project implements a Contextual Multi-Armed Bandit (CMAB) based News Recommendation System.

Unlike traditional Multi-Armed Bandits (MAB), where the agent selects the globally best arm, a Contextual Bandit observes additional information (context) before selecting an action. The objective is to learn a policy that selects the optimal arm conditioned on the current context.

In this assignment:

- Context → User Category (User1, User2, User3)  
- Arm → News Category (Entertainment, Education, Tech, Crime)  
- Total Arms → 3 contexts × 4 categories = 12 arms  

The system integrates supervised learning for user classification and reinforcement learning for category selection to maximize expected reward (user engagement).

---

## 2. Dataset Description

### User Data
Files used:
- train_users.csv
- test_users.csv

Each row represents a user with multiple features.
The label column represents user category:
- User1
- User2
- User3

These categories correspond to contexts in the CMAB formulation.

### News Articles Data
File used:
- news_articles.csv

Each article contains a category:
- Entertainment
- Education
- Tech
- Crime

These categories correspond to arms of the bandit.

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Missing values were handled appropriately.
2. Categorical features were encoded.
3. Numerical features were scaled where necessary.
4. The train_users.csv dataset was split into:
   - 80% Training
   - 20% Validation

This prepared the dataset for classification and bandit training.

---

## 4. User Classification (Context Detector)

A supervised classification model was trained to predict the user category (User1, User2, User3).

The dataset was split into training and validation sets (80/20). The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Evaluation was performed using `sklearn.metrics.classification_report`.

The trained classifier serves as the Context Detector in the final recommendation pipeline.

---

## 5. Contextual Bandit Algorithms

Three contextual bandit strategies were implemented. Each context was trained independently with 4 arms corresponding to news categories.

All reinforcement learning simulations were run for:

T = 10,000 steps

---

### 5.1 Epsilon-Greedy

Strategy:
- With probability ε → Explore
- With probability (1 − ε) → Exploit

Hyperparameters tested:
- ε = 0.01
- ε = 0.1
- ε = 0.3

Observations:
- Small ε converges faster but may not explore sufficiently.
- Large ε explores more but slows convergence.
- Moderate ε provided balanced performance.

---

### 5.2 Upper Confidence Bound (UCB)

Strategy:

UCB(a) = Q(a) + C * sqrt(ln(t) / N(a))

Hyperparameters tested:
- C = 0.5
- C = 1
- C = 2

Observations:
- Larger C increases exploration.
- Smaller C may cause premature exploitation.
- UCB demonstrated strong convergence behavior and stable performance.

---

### 5.3 Softmax (τ = 1)

Strategy:

P(a) = exp(Q(a)/τ) / Σ exp(Q(b)/τ)

Temperature parameter τ was fixed at 1 as required.

Observations:
- Provides smooth probabilistic exploration.
- More stable compared to ε-greedy.
- Slightly slower convergence in some contexts compared to UCB.

---

## 6. RL Simulation and Analysis

For each algorithm and context:

- Cumulative rewards were computed.
- Average reward over time was plotted.
- Hyperparameter comparison plots were generated.

All plots include:
- Proper axis labels
- Legends
- Descriptive titles

The analysis shows the importance of balancing exploration and exploitation. UCB generally demonstrated faster and more stable convergence, while epsilon-greedy and softmax provided competitive performance depending on hyperparameters.

---

## 7. Final Recommendation Engine

The complete end-to-end system works as follows:

1. Input user features from test_users.csv.
2. Predict user context using the trained classifier.
3. Select the optimal news category using the trained contextual bandit model.
4. Randomly sample an article from the selected category.
5. Output:
   - Recommended news category
   - Sampled article

This completes the Contextual Bandit-based News Recommendation workflow.

---

## 8. Conclusion

This project demonstrates the integration of supervised learning and reinforcement learning in a contextual bandit framework.

Key insights:

- Context-aware decision-making improves recommendation quality.
- Proper hyperparameter tuning significantly affects reward convergence.
- UCB showed strong empirical performance.
- Context detection accuracy is crucial for overall system performance.

The final system successfully recommends news articles tailored to predicted user categories while maximizing expected engagement.

---

## 9. How to Run

Install required package:

pip install rlcmab-sampler

Run the notebook:

lab3_results_U20230021.ipynb

Python version required: 3.12 or higher.

---

## 10. Repository Structure

- lab3_results_U20230021.ipynb
- README.md

The notebook contains all code, experiments, plots, and results.
