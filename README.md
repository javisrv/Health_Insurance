# Health_Insurance
Machine Learning automation models to predict whether customers would subscribe to a car insurance.

# Repository rationale
    • We selected this repository in order to practice some tools beyond data science fundamental skills, like deploying Machine Learning Models or using python pickle library for time saving. 

    • We aim to show how databases from a particular field can be transferred to other business strategies. 

    • In this project, our potential client is an Insurance company that has provided Health Insurance to its customers. They need our help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

    • Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimize its business model and revenue.
    • To measure the performance of the model we use the auc_score metric. However, in order to find the best threshold, we decided to give weight to the errors and to the successes at the time of the prediction. False positives will count a loss of $100 and true positives will count a gain of $350. (we have decided this after exploring different business approaches). The threshold that maximizes this gain will be chosen.
What is the source of the data?
The datasets were obtained from the following link https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction?select=test.csv

# What files does this repository contain?
. 
    ├── data_raw
           ├── test.csv	<--- Original test set
           ├── train.csv	<--- Original train set
    ├── documents
           ├── train_script.csv             <--- Clean test set, for run into model_search.py
           ├── val_script.csv               <--- Clean test set, for run into model_search.py
           ├── test_script.csv	            <--- Clean test set, for run into clients_predict.py
           ├── test_model_notebook.csv	    <--- Clean test set, for run into Models.ipynb
           ├── train_model_notebook.csv     <--- Clean train set, for run into Models.ipynb	
           ├── val_model_notebook.csv	      <--- Clean validation set, for run into Models.ipynb 
           ├── clients_list.xlsx	          <---List of customers likely to purchase the insurance
    ├── notebooks
           ├── EDA.ipynb          <--- Makes data cleaning, wrangling and EDA 
           ├── Models.ipynb       <--- Makes a training model and predict Clients
    ├── scripts          
           ├── cleaning.py              <--- Creates train, val and test csv for the scripts files 
           ├── model_search.py          <--- Creates the trained and the threshold pkls files
           ├── clients_predict.py       <--- Creates the clients list
           ├── Workflow.pdf             <--- Explain how to run the scripts
    ├── src
           ├── How to create the environment.txt
           ├── insurance_env.yml                   <---  You need it to create the environment
           ├── metrics_description.py	             <---  Show metrics into Models.ipynb
           ├── model.pkl                           <--- Model trained, ready to make predictions
           ├── threshold.pkl                       <--- The threshold that the model needs
    ├── README.md

# How should I use this repository?
The repository is basically divided into two parts:

- notebooks:

    • EDA.ipynb: this file performs a cleanup of the training and test data set. In addition, he performs an EDA in order to understand who and how the company's customers are, as well as to understand the relationships between the variables. This notebook includes the pre-processing of the data.

    • Models.ipynb: this file performs a detailed analysis of the Machine Learning models (Random Forest, Logistic Regression, XGBoost, and ADABoost) to find the best predictive model. An analysis of the costs that prediction errors can have (in our case we take into account only the false positives) and the successes (in our case we take into account only the true positives) was carried out.

- scripts:

    • This folder contains 3 .py files: cleaning.py, model_search.py and clients_predict.py
 
    • As described in the Workflow.pdf file, these 3 scripts are combined to be executed sequentially and finally return an excel file, with the list of clients that probably take out insurance for their vehicle.

References
    • How to Avoid Data Leakage When Performing Data Preparation. Jason Brownlee. https://machinelearningmastery.com/data-preparation-without-data-leakage/
    • Pickle - Python object serialization Definition.  https://docs.python.org/3/library/pickle.html
