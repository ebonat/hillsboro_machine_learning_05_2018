
# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def main():

#     each datapoint is a 8x8 image of a digit.
#     classes     10
#     samples per class     ~180
#     samples total     1797
#     dimensionality     64
#     features     integers 0-16

    digits = datasets.load_digits()   
#     flatten the image is required
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

#     data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    
#     create xgboost classifier model using default hyperparameter
    xgboost_classifier = XGBClassifier()
    print(xgboost_classifier)
    print()
    
#     fit the model
    xgboost_classifier.fit(X_train, y_train)
    
#     get y predict
    y_predict = xgboost_classifier.predict(X_test)    
    
#     classification metrics
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