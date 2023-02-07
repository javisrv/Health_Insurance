# Basic libraries
from datetime import datetime
import pandas as pd
import numpy as np
import datetime
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

def correct_dtypes(dataframe):
    print("\t1/2- Running correct_dtypes()...")
    #dataframe["id"] = dataframe["id"].astype("int")
    dataframe["Gender"] = dataframe["Gender"].astype("object")
    dataframe["Age"] = dataframe["Age"].astype("int")
    dataframe["Driving_License"] = dataframe["Driving_License"].astype("object")
    dataframe["Region_Code"] = dataframe["Region_Code"].astype("object")
    dataframe["Previously_Insured"] = dataframe["Previously_Insured"].astype("object")
    dataframe["Vehicle_Age"] = dataframe["Vehicle_Age"].astype("object")
    dataframe["Vehicle_Damage"] = dataframe["Vehicle_Damage"].astype("object")
    dataframe["Annual_Premium"] = dataframe["Annual_Premium"].astype("float")
    dataframe["Policy_Sales_Channel"] = dataframe["Policy_Sales_Channel"].astype("object")
    dataframe["Vintage"] = dataframe["Vintage"].astype("int")
    try:
        dataframe["Response"] = dataframe["Response"].astype("int")
    except:
    	None
    print("\t2/2- correct_dtypes() done.")
    
    return dataframe 
    
def correct_Annual_Premium(dataframe):
    """
    Receive the dataframe and replace the Annual_Premium variable with the log of the feature. From the distribution 
    	we choose the data with less dispersion. Outliers are interpreted as errors in data entry. 

    Given the distribution showned in EDA script, it is decided to use log transformation. Log transformation  de-emphasizes outliers and allows 
    	us to potentially obtain a bell-shaped distribution. The idea is that taking the log of the data can restore symmetry to the data. 
    Data with a high dispersion interpreted as erroneous are eliminated.
    
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
           
    Returns
    -------
    pandas.core.series.Series: 
        Returns the modified dataframe.
    """
    print("\t1/4- Running correct_Annual_Premium()...")
    print("\t2/4- Initial shape:", dataframe.shape)
    dataframe = dataframe[dataframe["Annual_Premium"] > 3500]
    dataframe['Annual_Premium_log'] = np.log(dataframe['Annual_Premium'])
    print("\t3/4- Final shape:", dataframe.shape)
    print("\t4/4- correct_Annual_Premium() done.")
        
    return dataframe

def correct_policy(dataframe):
    print("\t1/6- Running correct_policy()...")
    print("\t2/6- Number of unique categories for 'Policy_Sales_Channel':", dataframe["Policy_Sales_Channel"].nunique())
    top_policy = dataframe["Policy_Sales_Channel"].value_counts().head().index.tolist()
    print("\t3/6- Performing the transformation...")
    for i in dataframe.index:
        if dataframe.loc[i, "Policy_Sales_Channel"] not in top_policy:
            dataframe.loc[i, "Policy_Sales_Channel"] = "Other"
    dataframe["Policy_Sales_Channel"] = dataframe["Policy_Sales_Channel"].astype("str")   
    print("\t4/6- Transformation done.")
    print("\t5/6- Number of unique categories for 'Policy_Sales_Channel':", dataframe["Policy_Sales_Channel"].nunique())
    print("\t6/6- correct_policy() done.")
        
    return dataframe
    
def correct_region(dataframe):
    print("\t1/6- Running correct_region()...")
    print("\t2/6- Number of unique categories for 'Region_Code':", dataframe["Region_Code"].nunique())
    top_region = dataframe["Region_Code"].value_counts().head().index.tolist()
    print("\t3/6- Performing the transformation...")
    for i in dataframe.index:
        if dataframe.loc[i, "Region_Code"] not in top_region:
            dataframe.loc[i, "Region_Code"] = "Other" 
    dataframe["Region_Code"] = dataframe["Region_Code"].astype("str") 
    print("\t4/6- Transformation done.")
    print("\t5/6- Number of unique categories for 'Region_Code':", dataframe["Region_Code"].nunique())
    print("\t6/6- correct_region() done.")
      
    return dataframe

def correct_Vehicle_Age(dataframe):
    print("\t1/2- Running correct_Vehicle_Age()...")
    dataframe['Vehicle_Age'] = dataframe.Vehicle_Age.map({"< 1 Year": "Vehicle_Age_lower 1 Year", 
    							  "> 2 Years": "Vehicle_Age_higher 2 Years", 
    							  "1-2 Year" : "1-2 Year"})
    print("\t2/2- correct_Vehicle_Age() done.")
   							  
    return dataframe
    
def drop_variables(dataframe, list):
    print("\t1/2- Running drop_variables()...")
    dataframe =  dataframe.drop(list, axis = 1)
    print("\t2/2- drop_variables() done.")
    
    return dataframe

def train_val_split(dataframe): 
    print("\t1/9- Running train_val_split()...")
    X = dataframe.drop("Response", axis = 1)
    y = dataframe ["Response"]
    print("\t2/9- X.shape:", X.shape)
    print("\t3/9- y.shape:", y.shape)
    print("\t4/9- Making train and validation split..")
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 123)
    print("\t5/9- X_train shape:", X_train.shape)
    print("\t6/9- y_train shape:", y_train.shape)
    print("\t7/9- X_val shape:", X_val.shape)
    print("\t8/9- y_val shape:", y_val.shape)
    print("\t9/9- End train_val_split()")
    train_dataframe = pd.concat([X_train, y_train], axis = 1)
    val_dataframe = pd.concat([X_val, y_val], axis = 1)
    
    return train_dataframe, val_dataframe

def numerical_categorical_split(dataframe): 
    print("\t1/4- Running numerical_categorical_split()...")
    cat_dataframe = dataframe.select_dtypes("object")
    num_dataframe = dataframe.select_dtypes(exclude = "object")
    print("\t2/4- cat_dataframe columns:", list(cat_dataframe.columns))
    print("\t3/4- num_dataframe columns:", list(num_dataframe.columns))  
    print("\t4/4- numerical_categorical_split() done.")
    
    return cat_dataframe, num_dataframe

def create_dummy(cat_dataframe): 
    print("\t1/2- Running create_dummy()...")
    enc = OneHotEncoder(drop = "first", handle_unknown = "ignore")
    enc_fit= enc.fit_transform(cat_dataframe[["Gender", "Vehicle_Age", "Vehicle_Damage", "Region_Code", "Policy_Sales_Channel"]]).toarray()
    enc_fit_df = pd.DataFrame(enc_fit, columns = enc.get_feature_names_out(), index = cat_dataframe.index)
    print("\t2/2- create_dummy() done.")
    
    return enc_fit_df
 
def scaler(num_dataframe, index_dataframe, **kwargs):
    print("\t1/2- Running scaler()")
    std = StandardScaler()
    try:
    	y = num_dataframe["Response"]
    	num_dataframe = num_dataframe.drop("Response", axis = 1)
    except:
    	None
    #print("Print num_dataframe:\n", num_dataframe.head())
    if kwargs.get('train') == True:
        print("Inside the if statement")
        fit = std.fit(num_dataframe)
        num_dataframe_std = fit.transform(num_dataframe)
    else:
        print("Inside the else statement")
        fit = kwargs.get('fit')
        num_dataframe_std = fit.transform(num_dataframe)
    num_dataframe_std_df = pd.DataFrame(num_dataframe_std, columns = num_dataframe.columns, index = index_dataframe.index)
    print("\t2/2- End of scaler()")

    if kwargs.get('train') == True:
        print("Return inside the if statement")
        return num_dataframe_std_df, y, fit
    elif kwargs.get('val') == True:
        print("Return inside the elif statement")
        return num_dataframe_std_df, y
    else:
        print("Return inside the else statement")
        return num_dataframe_std_df
        

def main(p_input_train, p_input_test, p_outcome_train, p_outcome_val, p_outcome_test):
    # Uploading dataframe					
    print("###" * 30)
    print("1/12")
    print("Uploading dataframe...")
    df_train = pd.read_csv(p_input_train)
    df_test = pd.read_csv(p_input_test)
    df_test_id = df_test["id"]
    df_test.drop("id", axis = 1, inplace = True)
    print("Dataframe uploaded.")				

    # Fixing dtypes						
    print("###" * 30)
    print("2/12") 
    print("Fixing train dtypes:")
    df_train = correct_dtypes(df_train)
    print("Train dtypes fixed.")
    print("\n")
    print("Fixing test dtypes:")
    df_test = correct_dtypes(df_test)
    print("Test dtypes fixed.")					
    
    # Fixing Annual_Premium					
    print("###" * 30)
    print("3/12")
    print("Fixing train 'Annual_Premium':")
    df_train = correct_Annual_Premium(df_train)
    print("Train 'Annual_Premium' fixed.")
    print("\n")
    print("Fixing test 'Annual_Premium':")
    df_test = correct_Annual_Premium(df_test)
    print("Test 'Annual_Premium' fixed.")			
   
    # Fixing Policy_Sales_Channel				
    print("###" * 30)
    print("4/12")
    print("Fixing train 'Policy_Sales_Channel':")
    df_train = correct_policy(df_train)
    print("Train 'Policy_Sales_Channel' fixed.")
    print("\n")
    print("Fixing test 'Policy_Sales_Channel':")
    df_test = correct_policy(df_test)
    print("Test 'Policy_Sales_Channel' fixed.")			
    
    # Fixing Region_Code					
    print("###" * 30)
    print("5/12")
    print("Fixing train 'Region_Code':")
    df_train = correct_region(df_train)
    print("Train 'Region_Code' fixed.")
    print("\n")
    print("Fixing test 'Region_Code':")
    df_test = correct_region(df_test)
    print("Test 'Region_Code' fixed.")				

    # Fixing Vehicle_Age					
    print("###" * 30)
    print("6/12")
    print("Fixing train 'Vehicle_Age':")
    df_train = correct_Vehicle_Age(df_train)
    print("Train 'Vehicle_Age' fixed.")
    print("\n")
    print("Fixing test 'Vehicle_Age':")
    df_test = correct_Vehicle_Age(df_test)
    print("Test 'Vehicle_Age' fixed.")					

    # Dropping useless variable					
    print("###" * 30)
    print("7/12")
    print("Dropping train useless variables:")
    df_train = drop_variables(df_train, ["id", "Annual_Premium"])
    print("Drop done.")
    print("\n")
    print("Dropping test useless variables:")    
    df_test = drop_variables(df_test, ["Annual_Premium"])
    print("Drop done.")					
    
    # Making train_validation split
    print("###" * 30)
    print("8/12")
    print("Running train/validation split:")			
    train_df, val_df = train_val_split(df_train) 		
    print("Train/validation split executed.")			
    
    # Making variables split					
    print("###" * 30)
    print("9/12")
    print("Running categorical/numerical variables split of train/val:")
    cat_train_df, num_train_df = numerical_categorical_split(train_df) 
    cat_val_df, num_val_df = numerical_categorical_split(val_df)
    print("Categorical/numerical variables split of train/val executed.")
    print("\n")
    print("Running categorical/numerical variables split of test:")
    cat_test_df, num_test_df = numerical_categorical_split(df_test)
    print("Categorical/numerical variables split of test executed.")	
    
    # Making dummies variables					
    print("###" * 30)
    print("10/12")
    print("Running create_dummy() for train/val:")   
    cat_dummy_train = create_dummy(cat_train_df) 
    cat_dummy_val = create_dummy(cat_val_df) 
    print("create_dummy() for train/val executed.")
    print("\n")
    print("Running create_dummy() for test:")
    cat_dummy_test = create_dummy(cat_test_df)
    print("create_dummy() for test executed.")				
    
    # Normalizing numerical variables				
    print("###" * 30)
    print("11/12")
    print("Running scaler() for train/val:")      
    num_train_std, y_train, fit_std = scaler(num_train_df, train_df, train = True)				 
    num_val_std, y_val = scaler(num_val_df, val_df, fit = fit_std, train = False, val = True)[:2]				
    print("scaler() train/val executed.")
    print("\n")
    print("Running scaler() for test:")
    num_test_std = scaler(num_test_df, df_test, fit = fit_std, train = False, val = False)
    print("scaler() test executed.")
    
    # Exporting dataframes
    print("###" * 30)
    print("12/12")
    print("Concatenating categorical and numerical variables...")
    train_final = pd.concat([num_train_std, cat_dummy_train, y_train], axis = 1)
    val_final = pd.concat([num_val_std, cat_dummy_val, y_val], axis = 1)
    test_final = pd.concat([df_test_id, num_test_std, cat_dummy_test], axis = 1)
    print("Concatenating done.")
    print("Exporting training, validation and test data...")
    train_final.to_csv(p_outcome_train, index = False)
    val_final.to_csv(p_outcome_val, index = False)
    test_final.to_csv(p_outcome_test, index = False)
    print("Exporting dataframes done.")
    print("\n")
    print("Train shape:", train_final.shape)
    print("First 5 rows of training set:\n", train_final.head())
    print("\n")
    print("Validation shape:", val_final.shape)
    print("First 5 rows of validation set:\n", val_final.head())
    print("\n")
    print("Test shape:", test_final.shape)
    print("First 5 rows of test set:\n", test_final.head())     
    	
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    end = datetime.datetime.now()
    delta = round((end - start).seconds / 60, 2)
    print("###" * 60)
    print("Data cleanup and preprocessing of training and test set finished.")
    print(f"Time it took to run this script: {delta} minutes.\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_input_train", type = str, required = True, help = "Insert the path of the 'train.csv' file. It ends with ../Health_Insurance/data_raw/train.csv")
    parser.add_argument("--path_input_test", type = str, required = True, help = "Insert the path of the 'test.csv' file. It ends with ../Health_Insurance/data_raw/test.csv")
    parser.add_argument("--path_outcome_train", type = str, required = True, help = "Insert the path where the training set file will be saved. It ends with ../Health_Insurance/documents/train_script.csv")
    parser.add_argument("--path_outcome_val", type = str, required = True, help = "Insert the path where the validation set file will be saved. It ends with ../Health_Insurance/documents/val_script.csv")
    parser.add_argument("--path_outcome_test", type = str, required = True, help = "Insert the path where the test set file will be saved. It ends with ../Health_Insurance/documents/test_script.csv")
    args = parser.parse_args()
    
    inputs_train = args.path_input_train
    inputs_test = args.path_input_test
    outcomes_train = args.path_outcome_train
    outcomes_val = args.path_outcome_val
    outcomes_test = args.path_outcome_test
    
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    start = datetime.datetime.now()
    print("\nCurrent time", current_time)
    print("Starting data cleanup and preprocessing of training and test set:") 
    main(inputs_train, inputs_test, outcomes_train, outcomes_val, outcomes_test)
    
    
