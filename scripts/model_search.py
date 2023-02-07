
# Basic libraries
from datetime import datetime
import datetime
import pandas as pd
import numpy as np
import pickle
import scipy.stats as st
import argparse
import warnings
warnings.filterwarnings('ignore')

# Data preprocessing libraries
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


# Machine Learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Metrics libraries
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report, ConfusionMatrixDisplay

def model(p_input_train, p_input_val, iterations, p_outcomes_model, p_outcomes_threshold):
    # Uploading dataframes
    print("###" * 30)
    print("1/5")
    print("Importing dataframes:")   
    train = pd.read_csv(p_input_train)
    X_train = train.drop(["Response"], axis = 1)
    y_train = train["Response"]
    val = pd.read_csv(p_input_val)
    X_val = val.drop(["Response"], axis = 1)
    y_val = val["Response"]
    print("Importing dataframes done.")
     
    # Model dictionaries and hyperparameters building
    print("###" * 30)
    print("2/5")
    print("Creating a dictionary of models and hyperparameters:")
    models = {"Random Forest": RandomForestClassifier(),
               "Logistic Regression": LogisticRegression(),
               "XGBoost": XGBClassifier(),
               "AdaBoost": AdaBoostClassifier()}
    
    hyperparameters = {"Random Forest": {"classifier__n_estimators": [100, 200],
                                         "classifier__class_weight": ["balanced"],
                                         "classifier__max_features": ["auto", "sqrt", "log2"],
                                         "classifier__max_depth" : [3, 5, 7, 8, 10],
                                         "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10, 0.25],
                                         "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10, 0.25],
                                         "classifier__criterion" :["gini", "entropy"]},
                       "Logistic Regression": {"classifier__penalty" : ["l1", "l2"], 
                                               "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                               "classifier__solver": ["liblinear"],
                                               "classifier__class_weight": ["balanced", {0: 0.10, 1: 0.90}, 
                                                                                        {0: 0.20, 1: 0.80},
                                                                                        {0: 0.30, 1: 0.70},
                                                                                        {0: 0.40, 1: 0.60},
                                                                                        {0: 0.50, 1: 0.50},
                                                                                        {0: 0.60, 1: 0.40},
                                                                                        {0: 0.70, 1: 0.30},
                                                                                        {0: 0.80, 1: 0.20},
                                                                                        {0: 0.90, 1: 0.10}]},
                       "XGBoost": {"classifier__n_estimators" : st.randint(20,40), 
                                   "classifier__max_depth": st.randint(3, 12), 
                                   "classifier__learning_rate": st.uniform(0.05, 0.4),
                                   "classifier__colsample_bytree": st.beta(10, 1), 
                                   "classifier__subsample": st.beta(10, 1), 
                                   "classifier__gamma": st.uniform(0, 10),
                                   "classifier__reg_alpha": st.uniform(0.05,10), 
                                   "classifier__min_child_weight": st.uniform(1,20)},
                       "AdaBoost": {"classifier__n_estimators": [10, 50, 100, 200, 500],
                                    "classifier__learning_rate": [0.0001, 0.01, 0.1, 1.0, 1.1, 1.2]}
                                    }
    print("Creating dictionary done.")
    print("Keys of the dictionary:", list(models.keys()))
    
    # Results dataframe  building
    print("###" * 30)
    print("3/5")
    print("Creating a dataframe with the results of each model:")
    model_results = pd.DataFrame({"Model": [], 
                                  "Validation Recall": [], 
                                  "Validation Precision": [],
                                  "Val TP": [], "Val FP": [], "Val TN": [], "Val FN": [],
                                  "Validation AUC": [], "Threshold": [], "Profit": [], "Execution Time": []})
    print("Creating dataframe done.")
    print("Columns of the dataframe:", model_results.columns.tolist())
    
    # Training of all models begins
    print("###" * 30)
    print("4/5")
    print("Training of all models begins")
    start2 = datetime.datetime.now()
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current time:", current_time)
        
    best_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraning {model_name}:")
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("\tCurrent time", current_time)
        
        # Building pipeline
        print("\tPipeline building")
        scaler = StandardScaler()
        steps = [("scaler", scaler), ("classifier", model)]
        pipeline = Pipeline(steps = steps)
        print("\tPipeline Builded")
        
        # Selecting the hyperparameter grid
        print("\tSelecting the hyperparameter grid")
        param_grid = hyperparameters[model_name]
        print("\tHyperparameter grid selected")
        
        # Instantiating RandomizedSearchCV
        print("\tInstantiating RandomizedSearchCV")
        rscv = RandomizedSearchCV(pipeline, param_grid, n_iter = iterations, n_jobs = -1, cv = 3, verbose = 0, scoring = "roc_auc")
        print("\tRandomizedSearchCV Instantiated")
        
        # Beginning the training
        print(f"\tBeginning the training of {model_name}...")
        rscv.fit(X_train, y_train)
        best_model = rscv.best_estimator_
        print("\tTraining finished")
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("\tCurrent time", current_time)
        
        best_models.update({model_name: best_model})
        
        # Predicting
        print(f"\tPredicting from {model_name}")
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i in threshold:
            start = datetime.datetime.now()
            y_pred_proba = best_model.predict_proba(X_val)[:, 1]
            y_pred = [1 if y > i else 0 for y in y_pred_proba]
            print(f"\t\tPredictiong for threshold {i} generated")
            pred_val = pd.DataFrame({"y_pred": y_pred, "y_val": y_val})
            pred_val["profit"] = (pred_val["y_pred"] * (-100)) + (pred_val["y_pred"] * pred_val["y_val"] * 350)
            
            print(f"\t\tGenerating scores for threshold {i}")
            recall = round(recall_score(y_val, y_pred), 2)
            precision = round(precision_score(y_val, y_pred), 2)
            auc = round(roc_auc_score(y_val, y_pred_proba), 2)
            confusion = confusion_matrix(y_val, y_pred)
            end = datetime.datetime.now()
            delta = round((end - start).seconds / 60, 2)
            print(f"\t\tScores for threshold  {i} generated")
            results = pd.DataFrame({"Model": [model_name], "Validation Recall": [recall], "Validation Precision": [precision], 
                                    "Val TP": [confusion[1,1]], "Val FP": [confusion[1,0]], "Val TN": [confusion[0,0]], "Val FN": [confusion[0,1]],
                                    "Validation AUC": [auc], "Threshold": [i], "Profit": [pred_val["profit"].sum()], "Execution Time": [delta]})

            model_results = pd.concat([model_results, results], axis = 0, ignore_index = True)
    print("\nShowing the first 10 rows of the dataframe:\n", model_results.sort_values(by = "Profit", ascending = False).head(10))
    print("\nPredictions finished.")
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current time:", current_time)
    end2 = datetime.datetime.now()
    delta = round((end2 - start2).seconds / 60, 2)
    print(f"Process took: {delta} minutes.")

    # Saving final model and threshold as pkl
    print("###" * 30)
    print("5/5")
    print("Saving final model and threshold  as pkl:")
    model_max_profit = model_results[model_results["Profit"] == model_results["Profit"].max()]
    model_name_final = model_max_profit.loc[model_max_profit["Execution Time"].idxmin(), "Model"]
    threshold_final = model_max_profit.loc[model_max_profit["Execution Time"].idxmin(), "Threshold"]
    model_final = best_models[model_name_final]
    
    with open(p_outcomes_model, "wb") as file:
        pickle.dump(model_final, file)
        
    with open(p_outcomes_threshold, "wb") as file:
        pickle.dump(threshold_final, file)       
    
    print("Saving pkl files done.")
    
    return model_name_final, threshold_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_input_train", type = str, required = True, help = "Insert the path of the 'train_script.csv' file. It ends with ../Health_Insurance/documents/train_script.csv")
    parser.add_argument("--path_input_val", type = str, required = True, help = "Insert the path of the 'val_script.csv' file. It ends with ../Health_Insurance/documents/val_script.csv")
    parser.add_argument("--n_iterations", type = int, required = True, help = "Insert the number of iterations to search for hyperparameters with RandomizedSearchCV")
    parser.add_argument("--path_outcomes_model", type = str, required = True, help = "Insert the path where the model.pkl file will be saved. It ends with ../Health_Insurance/src/model.pkl")
    parser.add_argument("--path_outcomes_threshold", type = str, required = True, help = "Insert the path where the threshold.pkl file will be saved. It ends with ../Health_Insurance/src/threshold.pkl")
    args = parser.parse_args()
    
    inputs_train = args.path_input_train
    inputs_val = args.path_input_val
    n_iterations = args.n_iterations
    outcomes_model = args.path_outcomes_model
    outcomes_threshold = args.path_outcomes_threshold
    
    now = datetime.datetime.now()    
    current_time = now.strftime("%H:%M:%S")
    print("\nCurrent time:", current_time)
    print("Building the predictive model with its corresponding threshold:") 
    model_name, threshold_value = model(inputs_train, inputs_val, n_iterations, outcomes_model, outcomes_threshold)
    print("###" * 30)
    print("Predictive model with its corresponding threshold created.")
    print("\tBest model:", model_name)
    print("\tBest threshold:", threshold_value)
    print() 
    
    
    
    
