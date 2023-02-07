# Basic libraries
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

def client_prediction(p_test, p_model, p_threshold, p_output):
    print("###" * 30)
    print("1/6")    
    print("Importing dataframe...")
    path = p_test
    data = pd.read_csv(path)
    data = data.set_index("id") 
    print("Importing dataframe done.")
    
    print("###" * 30)
    print("2/6")       
    print("Importing model...")
    with open(p_model, "rb") as file:
        model_from_disk = pickle.load(file)
    print("Importing model done.")
    
    print("###" * 30)
    print("3/6")       
    print("Importing threshold...")       
    with open(p_threshold, "rb") as file:
        threshold_from_disk = pickle.load(file)
    print("Importing threshold done.")

    print("###" * 30)
    print("4/6")     
    print("Making predictions...")
    y_pred = model_from_disk.predict(data)
    y_pred_proba = model_from_disk.predict_proba(data)[:, 1]
    y_pred = [1 if y > threshold_from_disk else 0 for y in y_pred_proba]
    print("Making predictions done.")

    print("###" * 30)
    print("5/6")     
    print("Creating client list...")
    data["Response"] = y_pred
    data = data.reset_index()
    mask_response = data["Response"] == 1
    final_list = data[mask_response].reset_index()[["id"]].rename(columns = {"id": "Clients-to-call list"})
    print("Client list done.")
    
    print("###" * 30)
    print("6/6")       
    print("Creating 'clients_list.xlsx'...")
    final_list.to_excel(p_output, sheet_name = "Clients")
    print("'clients_list.xlsx' created.")
    
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    end = datetime.datetime.now()
    delta = round((end - start).seconds / 60, 2)
    print("###" * 30)
    print("Performing the prediction of the clients and saving the results in 'documents/clients_list.xlsx' finished.")
    print(f"Time it took to run this script: {delta} minutes.\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_inputs", type = str, required = True, help = "Insert the path of the 'test_script.csv' file. It ends with ../Health_Insurance/documents/test_script.csv")
    parser.add_argument("--path_model", type = str, required = True, help = "Insert the path of the 'model.pkl' file. It ends with ../Health_Insurance/src/model.pkl")
    parser.add_argument("--path_threshold", type = str, required = True, help = "Insert the path of the 'threshold.pkl' file. It ends with ../Health_Insurance/src/threshold.pkl")
    parser.add_argument("--path_outcome", type = str, required = True, help = "Insert the path where the final list of clients will be saved. It ends with ../Health_Insurance/documents/clients_list.xlsx")
    args = parser.parse_args()
    
    test = args.path_inputs
    model = args.path_model
    threshold = args.path_threshold
    outcome = args.path_outcome
    
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    start = datetime.datetime.now()
    print("\nCurrent time", current_time)
    print("Performing the prediction of the clients and saving the results in 'clients_list.xlsx':")    
    client_prediction(test, model, threshold, outcome) 
