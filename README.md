# Improving GRU4Rec with the Addition of Feature Network
Author: Ya-Yun Huang, Haoliang Jiang, Yufei Lei, Chloe Liu
## 00 Summary
This project focuses on improving the GRU4Rec + SNQN recommender with the help of taking user or item features into account. We used two different offline evaluation metrics (NDCG and HR) for benchmarking. Our findings suggest that the inclusion of a feature network can improve the results of product recommendation. 
## 01 Introduction
## 02 Literature Review
Reinforcement Learning is a type of machine learning that allows an agent to learn how to behave in an environment by trial and error. In the context of recommender systems, RL can be used to learn how to recommend items to users in a way that maximizes their satisfaction.  
One of the challenges of applying RL to recommender systems is that the 3environment is often very large and complex. This makes it difficult to train an RL agent to learn the optimal policy. Another challenge is that the reward signal is often delayed, which can make it difficult for the agent to learn the correct association between its actions and rewards it receives.  
To address these challenges, Xin Xin et al. have proposed a number of methods for combining RL with supervised learning. One such method is called supervised negative Q-learning (SNQN). SNQN works by first training a supervised learning model to predict the reward for each possible action. The RL agent then uses this model to estimate the Q-values for each action. The Q-values are then used to train the RL agent to learn the optimal policy. In their paper, the experimental results have shown that SNQN can achieve significantly better performance than supervised learning methods. This is because SNQN is able to learn the optimal policy more quickly and efficiently. It is also able to generalize better to new users and items than other methods.  
In our project, we would like to make improvements on the aforementioned SNQN model by including a feature network such that the recommender is more personalized.  
Personalized real-time recommendation has had a profound impact on retail, media, entertainment, and other industries. However, developing recommender systems for every use case is costly, time-consuming, and resource-intensive. To fill this gap, Yifei Ma et al. have proposed a number of black-box recommender systems that can adapt to a diverse set of scenarios without the need for manual tuning. The structure that Yifei proposed allowed inclusion of item features and cold start. We will adopt this methodology and apply it to modify the backbone of Xin Xin's SNQN model.
## 03 Methodology
### 031 Data Cleaning and Feature Engineering
For the Retail Rocket dataset, we largely followed the same data cleaning processes as that proposed by Xin Xin's paper. We first removed the users or items that have lower than 2 interactions. We then label encoded the session_ids and behavior. Once we created a sorted data table, we read in the two item properties files and cancated these 2 files. Items without category_ids were removed. Last but not least, we one-hot encoded the merged dataset of event plus item features. We used the one-hot encoded item_category dataset for training the model. 
### 032 Modeling
## 04 Results
## 05 Conclusion and Limitations
## 06 Instructions
