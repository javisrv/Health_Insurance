import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report

def conf_matrix_train_val(Y_train, Y_train_pred, Y_test, Y_test_pred):
    # Train Confusion Matrix 
    cf_matrix_train = confusion_matrix(Y_train, Y_train_pred)
    group_names_train = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts_train = ["{0:0.0f}".format(value) for value in cf_matrix_train.flatten()]
    labels_train = [f"{v1}\n{v2}" for v1, v2 in zip(group_names_train, group_counts_train)]
    labels_train = np.asarray(labels_train).reshape(2,2)
    
    # Validation Confusion Matrix     
    cf_matrix_test = confusion_matrix(Y_test, Y_test_pred)
    group_names_test = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts_test = ["{0:0.0f}".format(value) for value in cf_matrix_test.flatten()]
    labels_test = [f"{v1}\n{v2}" for v1, v2 in zip(group_names_test, group_counts_test)]
    labels_test = np.asarray(labels_test).reshape(2,2)
    
    # Graphics
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4))
    sns.heatmap(cf_matrix_train, annot = labels_train, annot_kws = {'size': 14}, fmt = "", cmap = ListedColormap(["white"]), cbar = False, linewidths = 4, linecolor = 'black', ax = ax[0])
    sns.heatmap(cf_matrix_test, annot = labels_test, annot_kws = {'size': 14}, fmt = "", cmap = ListedColormap(["white"]), cbar = False, linewidths = 4, linecolor = 'black', ax = ax[1])
    
    # Titles, labels, ticks, etc
    ax[0].set_title("Train Confusion Matrix", fontsize = 20)
    ax[1].set_title("Validation Confusion Matrix", fontsize = 20)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()
    
    print("---" * 34)
    
def roc_curve_graph_train_val(y_test, y_test_pred_proba):
    fpr, tpr, thr = roc_curve(y_test, y_test_pred_proba) 

    plt.figure(figsize = (8,8))

    plt.plot(fpr, tpr, color = "black")
    plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01), color = "darkgreen", linestyle = "--")

    plt.title("Validation ROC Curve", fontsize = 18) 
    plt.xlabel("1 - Specificty", fontsize = 16)
    plt.ylabel("Sensitivity", fontsize = 16)

    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.yticks(ticks = [0,0.2,0.4,0.6,0.8,1], labels = ["",0.2,0.4,0.6,0.8,1], fontsize = 14)
    plt.xticks(fontsize = 14)

    style = dict(size = 14, color = "black", fontstyle = "oblique")
    props = dict(boxstyle = "round", facecolor = "grey", alpha=0.5)

    auc_score = round(auc(fpr, tpr), 2)

    plt.text(0.85, 0.1, f"AUC = {auc_score}", ha = "center", va = "bottom", **style, bbox = props)

    plt.show()
    
    print("---" * 34)
    
def metrics_train_val(Y_train, Y_test, Y_train_pred, Y_test_pred, Y_train_pred_proba, Y_test_pred_proba):
    #scoring = {"recall" : "recall", "precision": "precision", "roc_auc": "roc_auc"}
    #recalls = cross_validate(estimator = model, X = X_train_std, y = Y_train, cv = 3, verbose = 1, scoring = scoring)
    print(f"\tTrain Precision Score:{precision_score(Y_train, Y_train_pred):.3f}")
    print(f"\tValidation Precision Score:{precision_score(Y_test, Y_test_pred):.3f}")
    print("\t---" * 4)
    print(f"\tTrain Recall Score:{recall_score(Y_train, Y_train_pred):.3f}")
    print(f"\tValidation Recall Score:{recall_score(Y_test, Y_test_pred):.3f}")
    print("\t---" * 4)
    print(f"\tTrain ROC AUC Score:{roc_auc_score(Y_train, Y_train_pred_proba):.3f}")
    print(f"\tValidation ROC AUC Score:{roc_auc_score(Y_test, Y_test_pred_proba):.3f}")
    print("---" * 34)
    
def fit_train_val(model, X_train, y_train_arg, X_val, y_val_arg):
    print("For", str(model).split("(")[0],":")
    #global fit
    fit = model.fit(X_train, y_train_arg.values.reshape(-1,))
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics_train_val(y_train_arg, y_val_arg, y_train_pred, y_val_pred, y_train_pred_proba, y_val_pred_proba)
    
    conf_matrix_train_val(y_train_arg, y_train_pred, y_val_arg, y_val_pred)
    roc_curve_graph_train_val(y_val_arg, y_val_pred_proba)
    print(classification_report(y_val_arg, y_val_pred))
