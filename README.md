# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is to do basic exploratory data analysis and save the results for data save in `data/bank_data.csv`.

Then train a random classifier and logistic regression model respectively based on the bank data that is read from the local. 

The outputs including feature importance, roc curve and sharp figures evaludate the model's performance from difference aspects. Also some functions are missing, we are supposed to fill them out and make the code clean, well documented the same time.

We have already provided you the churn_notebook.ipynb file containing the solution to identify the customer churn, but without implementing the engineering and software best practices.

You need to refactor the given churn_notebook.ipynb file following the best coding practices to complete these files:

- churn_library.py
- churn_script_logging_and_tests.py
- README.md

## Project Dependencies
Before you started, please make sure you have these packages installed
- scikit-learn==0.22       
- shap==0.40.0     
- joblib==0.11
- pandas==0.23.3
- numpy==1.19.5 
- matplotlib==2.1.0      
- seaborn==0.11.2
- pylint==2.7.4
- autopep8==1.5.6

Also all the packages can be installed by the following command:
```
pip install -r requirements.txt
```
## Files and data description
The whole directory and files are listed as follows: 

```
├── Guide.ipynb
├── README.md
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn.png
│   │   ├── corr.png
│   │   ├── customer_age.png
│   │   ├── normalize.png
│   │   └── total_trans_ct.png
│   └── results
│       ├── cls_report_lr.png
│       ├── cls_report_rfc.png
│       ├── feature_rfc.png
│       └── roc.png
├── logs
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.6.txt
├── requirements_py3.8.txt
└── test_churn_script_logging_and_tests.py
```


data/bank_data.csv: the place where the raw datapoints are saved
logs: where test cases's logs are saved

## 1. churn_library.py 
The churn_library.py is a library of functions to find customers who are likely to churn. You may be able to complete this project by completing each of these functions, but you also have the flexibility to change or add functions to meet the rubric criteria.

The document strings have already been created for all the functions in the churn_library.py to assist with one potential solution. In addition, for a better understanding of the function call, see the Sequence diagram in the classroom.

After you have defined all functions in the churn_library.py, you may choose to add an if __name__ == "__main__" block that allows you to run the code below and understand the results for each of the functions and refactored code associated with the original notebook.

## 2. churn_script_logging_and_tests.py
This file should:

Contain unit tests for the churn_library.py functions. You have to write test for each input function. Use the basic assert statements that test functions work properly. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run.

Log any errors and INFO messages. You should log the info messages and errors in a .log file, so it can be viewed post the run of the script. The log messages should easily be understood and traceable.

Also, ensure that testing and logging can be completed on the command line, meaning, running the below code in the terminal should test each of the functions and provide any errors to a file stored in the /logs folder.




## Running Files
To run the main script and save the well-trained model you can run
```
python churn_library.py

```

To run functionality test cases, you can run
```
pytest test_churn_script_logging_and_tests.py
```




