# Improving GRU4Rec with the Addition of Feature Network
Author: Ya-Yun Huang, Haoliang Jiang, Yufei Lei, Chloe Liu

## Introduction
In this class project for AIPI 531, we used E-Commerce datasets Retail Rocket and H&M provided by instructors. Our task was to train different session based product recommenders with the help of SNQN reinforcement learning for E-commerce use cases and evaluate if the inclusion of side information such as item features for cold start will improve the performance of the recommender. We trained our baseline supervised learning recommender, which is GRU4Rec with Q-Learning being turned off, on the Retail Rocket dataset. After that, on each dataset, we trained two Deep-RL SNQN recommender (Q-Learning on), that are GRU4Rec mixing feature network and GRU4Rec alone. Not least, we used two different offline evaluation metrics (NDCG and HR) to evaluate how well our recommenders were doing. Our findings suggest that the inclusion of a feature network can potentially improve the performance of a product recommender when adequate training time and computation capacity is provided.  

## Literature Review
Reinforcement Learning is a type of machine learning that allows an agent to learn how to behave in an environment by trial and error. In the context of recommender systems, RL can be used to learn how to recommend items to users in a way that maximizes their satisfaction.  

One of the challenges of applying RL to recommender systems is that the 3environment is often very large and complex. This makes it difficult to train an RL agent to learn the optimal policy. Another challenge is that the reward signal is often delayed, which can make it difficult for the agent to learn the correct association between its actions and rewards it receives.  

To address these challenges, Xin Xin et al. have proposed a number of methods for combining RL with supervised learning. One such method is called supervised negative Q-learning (SNQN). SNQN works by first training a supervised learning model to predict the reward for each possible action. The RL agent then uses this model to estimate the Q-values for each action. The Q-values are then used to train the RL agent to learn the optimal policy. In their paper, the experimental results have shown that SNQN can achieve significantly better performance than supervised learning methods alone. This is because SNQN is able to learn the optimal policy more quickly and efficiently. It is also able to generalize better to new users and items than other methods.  

In our project, we would like to make improvements on the aforementioned product recommender by including a feature network such that the recommender is more personalized.  
Personalized real-time recommendation has had a profound impact on retail, media, entertainment, and other industries. However, developing recommender systems for every use case is costly, time-consuming, and resource-intensive.  

To fill this gap, Yifei Ma et al. have proposed a number of black-box recommender systems that can adapt to a diverse set of scenarios without the need for manual tuning. The structure that Ma proposed allowed inclusion of item features and cold start. The use of item features in HRNNs can be easily achieved by drawing a connection between RNN decoders and factorization models. Traditional RNN decoders calculate the score of each item; however, Ma combined both views from a decoder and a factorization model by creating a mixed score and assigning a mixing parameter lambda. This novel method shows promise for optimizing HRNNs and improving recommendation systems overall. For our project, we will adapt this method and modify Xin Xin's GRU4Rec model by adding another feature network.  

## Data Cleaning and Feature Engineering
For the Retail Rocket dataset, we largely followed the same data cleaning processes as that proposed by Xin Xin's paper. We first removed the users or items that have lower than or equal to 2 interactions. We then label encoded the session_ids and behavior. Once we created a sorted data table, we read in the two item properties files and cancated these 2 files. Items without category_ids were removed. We named the label-encoded dataset event_with_prop. Last but not least, we one-hot encoded the merged dataset of event plus item features. We one-hot encoded the categoryid and made a dataframe with each item's corresponding one-hot encoded category id and named the table item_category.  

Data processing is done slightly different from that conducted on the Retail Rocket dataset, however, the model outcome should not be largely affected. We first sampled 3000 items and users to generate a "not buy" dataset. We then merged this generated negative feedback dataset with the positive feedback dataset and form one single master data. We renamed some columns and cleaned the timestamp column by redefining the date format. We then follow the same steps as those for the Retail Rocket dataset. We removed users with less than or equal to 2 interactions. We then label encoded the item_id and categoryid. Last but not least, we one-hot encoded the categoryid and made a dataframe with each item's corresponding one-hot-encoded category id and named the table item_category.  

## Model Building

We used the source code from Xinxin where they implemented the joint training of supervised learning and SNQN reinforcement learning loss on the replay buffer. We added a dense layer `output_phi` to calculate the score of an item at the end of the GRU4Rec path. We also added another dense layer to embed the one-hot encoded item features. To calculate the `output_phi_prime` score based on item features, we concatnated the bias with the product of matrix multiplication of feature embeddings and hidden state. We then multiply each score by their weight and sum up to get our mixed score `output2`, which is phi tilda in the chart below. The modified structure of our model resembles the HRNN model proposed by Ma.  

## Model Results and Performance Evaluation
We used normalized discounted hit ratio (hr) and cumulative gain (ndcg) as our performance evaluation metrics. The quality of the recommendation list is evaluated by ndcg@10, which assigns higher scores to items that are ranked higher in the top-10 positions of the list. hr@10 evaluates if the model-generated recommendation list has the ground-truth item in its top-10 positions.  

Here are the result table for the four RL based models that we trained in 1 epoch:  

| Model | Data | Clicks hr@10 | Clicks ncdg@10 | Purchase hr@10 | Purchase ncdg@10 |
|:--------------|:-------------|:-------------|:-------------|:-------------|:-------------
| GRU4Rec + Feature Network, SNQN On | Retail Rocket | 0.001454 | 0.000540 | 0.000000 | 0.000000 |
| GRU4Rec, SNQN On | Retail Rocket | 0.000169 | 0.000068 | 0.000378 | 0.000176 |
| GRU4Rec + Feature Network, SNQN On | H&M | 0.004000 | 0.001574 | 0.000401 | 0.000162 |  
|GRU4Rec, SNQN On | H&M | 0.00000 | 0.000000 | 0.000368 | 0.000179 |

## Conclusion and Limitations
In this project, we trained different session based product recommendation recommenders for E-commerce use cases and compared the performances of these recommenders. We also included item features as side information to evaluate if doing so will help boost the performance of our recommender. We have discovered that when the item features are considered, the hr@10 an ncdg@10 for clicking are improving. However, the hr@10 and ncdg@10 for purchasing are not necessarily improving. We suspect that this may be caused by the following two reasons:  
1. The fully connected feature network might too simple to encode item features
2. Due to the large size of the datasets, we only had the adequate time and computation capacity to train 1 epoch for each model. In the future, we can increase the training time in order to boost the performance.

## Instructions to Run Code
1. We have created notebooks for you to run our models in Google Colab. You will need to upload this repository, along with the H&M and Retail Rocket datasets to Google Colab. Please refer to the following table for the model and driver notebook that you are looking for:  

| Model | Model File Path| Driver Notebook File Path | Data Source |
|:--------------|:-------------|:-------------|:-------------
| GRU4Rec + Feature Network, SNQN On| ~./HM_Chloe/SNQN_v1.py | ~./AIPI531_Project_SNQN_itemfeatures_retailrocket.ipynb | Retail Rocket |
| GRU4Rec, SNQN On| ~./HM_Chloe/SNQN_v1.py | ~./AIPI531_Project_SNQN_retailrocket.ipynb | Retail Rocket |
| GRU4Rec, SNQN Off (Baseline) | ~./Kaggle/SNQN_v3.py | ~./Kaggle/GRU4REC.ipynb | Retail Rocket |
| GRU4Rec + Feature Network, SNQN On | ~./HM_Chloe/SNQN_v1.py | ~./HM_Chloe/AIPI531_Project_SNQN_itemfeatures_HM.ipynb | H&M |
| GRU4Rec, SNQN On | ~./HM_Chloe/SNQN.py | ~./HM_Chloe/AIPI531_Project_SNQN_HM.ipynb | H&M |


2. Make sure that you have uploaded the code files and datasets to Google Drive and mounted your drive correctly. After mounting, modify the project directory path if necessary   

3. Open up terminal in the selected driver notebook. Here is an example photo of how you can access it: ![image](https://user-images.githubusercontent.com/90075179/236083065-e42fad95-ac6f-4b74-9651-c3de9728d20a.png)  

4. To install the requirements in a requirements.txt file, you can use the following command: `pip install -r requirements.txt`. This will install all of the packages listed in the requirements.txt file, along with their dependencies  

5. If you are using a virtual environment, you will need to activate it before installing the requirements: type `source venv/bin/activate` and then `pip install -r requirements.txt`  

6. Once the requirements are installed, you can start using them in the project. If you have never subscribed to Colab Pro, then you will not have access to the terminal. In this case, simply run the `!pip install...` lines in the driver notebook that you selected  

7. Once you have successfully installed all the packages, you can move to the third section of your selected notebook, which is to train and evaluate the model.  

8. To do so, just run the cell `! python "model.py" --model=GRU --epoch=1` to select the model you want to train, initiate the training, and view the losses and metrics. 

## Reference
1. Supervised Advantage Actor-Critic for Recommender Systems | X. Xin, A. Karatzoglou, I. Arapakis, and J. M. Jose | Proceedings of ACM Conference (Conference’17), 2021.  
2. Temporal-Contextual Recommendation in Real-Time | Y. Ma, H. Lin, B. Narayanaswamy, H. Ding | The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2020.
