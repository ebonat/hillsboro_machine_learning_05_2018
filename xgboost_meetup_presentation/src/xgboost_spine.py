

import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
#     read dataset_spine.csv file
    df_diabetes = pd.read_csv(filepath_or_buffer="dataset_spine.csv")
     
#     define x features and y label (target)
    X = df_diabetes.drop(labels="class", axis=1)    
    y = df_diabetes["class"]
    
#     data split to select train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     standard scaler for x features
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

#     create xgboost classifier model using default hyperparameter
    xgboost_classifier = XGBClassifier()
    print(xgboost_classifier)
    print()
    
#     fit the model
    xgboost_classifier.fit(X_train, y_train)
    
#     get y predict
    y_predict = xgboost_classifier.predict(X_test)    
    
#     classification  metrics
    accuracy_score_value = accuracy_score(y_test, y_predict) * 100
    accuracy_score_value = float("{0:0.2f}".format(accuracy_score_value))    
    print("Accuracy Score: {} %".format(accuracy_score_value))
    print()
        
    confusion_matrix_result = confusion_matrix(y_test, y_predict)
    print("Confusion Matrix:")
    print(confusion_matrix_result)
    print()
    
    classification_report_result = classification_report(y_test,y_predict)
    print("Classification Report:")    
    print(classification_report_result)
    print()  
    
if __name__ == '__main__':
    main()