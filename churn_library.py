# library doc string
"""
This library is to do basic exploratory data analysis and train a logistic \
regression model and random forest model,Then do shap analysis

Author: Jie Lyu
Creation Date: 2/12/2023
"""

# import libraries
import os
import shap
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

os.environ["QT_QPA_PLATFORM"] = "offscreen"


sns.set()


cat_columns = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

quant_columns = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

keep_cols = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            bank_data: pandas dataframe
    """
    bank_data = pd.read_csv(pth)
    return bank_data


def perform_eda(bank_data):
    """
    perform eda on bank_data and save figures to images folder
    input:
            bank_data: pandas dataframe

    output:
            None
    """
    bank_data["Churn"] = bank_data["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # fig1
    fig1 = plt.figure(figsize=(20, 10))
    bank_data["Churn"].hist()
    fig1.savefig("./images/eda/churn.png")
    # NOTE fig2
    fig2 = plt.figure(figsize=(20, 10))
    bank_data["Customer_Age"].hist()
    fig2.savefig("./images/eda/customer_age.png")

    # NOTE fig3
    fig3 = plt.figure(figsize=(20, 10))
    bank_data.Marital_Status.value_counts("normalize").plot(kind="bar")
    fig3.savefig("./images/eda/normalize.png")

    # NOTE fig4
    fig4 = plt.figure(figsize=(20, 10))
    sns.distplot(bank_data["Total_Trans_Ct"])
    fig4.savefig("./images/eda/total_trans_ct.png")

    # NOTE fig5
    fig5 = plt.figure(figsize=(20, 10))
    sns.heatmap(bank_data.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    fig5.savefig("./images/eda/corr.png")


def encoder_helper(bank_data, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            bank_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used \
                      for naming variables or index y column]

    output:
            bank_data: pandas dataframe with new columns for
    """

    for k in category_lst:
        res = []
        for val in bank_data[k]:
            res_groups = bank_data.groupby(k).mean()["Churn"]
            res.append(res_groups.loc[val])
        bank_data[k + "_" + response] = res
    return bank_data


def perform_feature_engineering(bank_data, response):
    """
    input:
              bank_data: pandas dataframe
              response: string of response name \
                        [optional argument that could be used for \
                        naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    x_bank = pd.DataFrame()
    x_bank[keep_cols] = bank_data[keep_cols]
    y_bank = bank_data[response]
    x_train, x_test, y_train, y_test = train_test_split(
        x_bank, y_bank, test_size=0.3, random_state=42
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    fig1 = plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(
        0.01,
        1.25,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    fig1.savefig("./images/results/cls_report_rfc.png")

    # NOTE lr
    fig2 = plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(
        0.01,
        1.25,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    fig2.savefig("./images/results/cls_report_lr.png")


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    fig.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    # plots
    # lr
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()
    fig.savefig("./images/results/lr_roc.png")

    # rfc
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax, alpha=0.8)
    plt.show()
    fig.savefig("./images/results/rfc_roc.png")
    # save best model
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    # NOTE feature importance
    feature_importance_plot(
        cv_rfc.best_estimator_, x_train, "./images/results/feature_rfc.png"
    )
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )


if __name__ == "__main__":
    bank_data_bank_data = import_data("./data/bank_data.csv")
    perform_eda(bank_data_bank_data)
    selected_bank_data = encoder_helper(bank_data_bank_data, cat_columns, "Churn")
    bank_x_train, bank_x_test, bank_y_train, bank_y_test = perform_feature_engineering(selected_bank_data, "Churn")
    train_models(bank_x_train, bank_x_test, bank_y_train, bank_y_test)
