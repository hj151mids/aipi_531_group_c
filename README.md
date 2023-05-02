# Improving GRU4Rec with the Addition of Feature Network
Author: Ya-Yun Huang, Haoliang Jiang, Yufei Lei, Chloe Liu
## 00 Summary
This project focuses on improving the GRU4Rec + SNQN recommender with the help of taking user or item features into account. We used two different offline evaluation metrics (NDCG and HR) for benchmarking. Our findings suggest that the inclusion of a feature network can improve the results of product recommendation. 
## 01 Introduction
## 02 Literature Review
Reinforcement Learning is a type of machine learning that allows an agent to learn how to behave in an environment by trial and error. In the context of recommender systems, RL can be used to learn how to recommend items to users in a way that maximizes their satisfaction.  
One of the challenges of applying RL to recommender systems is that the 3environment is often very large and complex. This makes it difficult to train an RL agent to learn the optimal policy. Another challenge is that the reward signal is often delayed, which can make it difficult for the agent to learn the correct association between its actions and rewards it receives.  
To address these challenges, Xin Xin et al have proposed a number of methods for combining RL with supervised learning. One such method is called supervised negative Q-learning (SNQN). SNQN works by first training a supervised learning model to predict the reward for each possible action. The RL agent then uses this model to estimate the Q-values for each action. The Q-values are then used to train the RL agent to learn the optimal policy. In their paper, the experimental results have shown that SNQN can achieve significantly better performance than supervised learning methods. This is because SNQN is able to learn the optimal policy more quickly and efficiently. It is also able to generalize better to new users and items than other methods.  

## 03 Methodology
### 031 Data Cleaning and Feature Engineering
### 032 Modeling
## 04 Results
## 05 Conclusion and Limitations
## 06 Instructions
