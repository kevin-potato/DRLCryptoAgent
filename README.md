# Artefacts

This folder contains all the code and files to run this project, as well as the agent for the training results in the report.



The overall structure of the project:

![structure](D:\Y4\FYP\s2\structure.png)



The structure of the model:

![model structure](D:\Y4\FYP\s2\model structure.png)



To run the project, make sure you have the following python libraries installed:

- matplotlib
- numpy
- pandas
- gym, shimmy
- torch
- stable_baselines3



How to run this project:

1. ***'BTC-USD.csv'*** contains price data for Bitcoin and is the starting point for this project.
2. run ***'preprocess_data.py'***, it will perform data preprocessing and add technical indicators to produce a new *.csv* file for subsequent training and learning.
3. run ***'train_model.py'***, it will start the training of two models using PPO and DQN algorithms respectively. ***'PPO.zip'*** and ***'DQN.zip'*** are trained models that were used in the report.
4. run ***'benchmark_strategies.py'***, it will start backtesting and comparing it to the benchmark strategy, by default it uses the two already trained models in the file, you can test your own model by modifying the file address in the code.



There is also a jupyter notebook file called ***'FYP.ipynb'*** containing all the code and visualisation results of the project."# DRLCryptoAgent" 
