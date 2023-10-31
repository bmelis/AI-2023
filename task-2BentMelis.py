import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import category_encoders as ce
from sklearn.metrics import accuracy_score

header = st.container()

with header:
    st.title("Taak 1 AI - Bent Melis (r0831245)")

selection = st.selectbox(
    'Select your sklearn technique:',
    ('Decision Tree Classifier', 'Support vector machine (SVM)','Gaussian Processes')
)

heart_failure_df = pd.read_csv("resources/heart_failure_clinical_records_dataset.csv", sep=',')


print(heart_failure_df.shape)
heart_failure_df.info()

feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase','diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'sex', 'smoking','time']

X = heart_failure_df[feature_cols]
y = heart_failure_df[['DEATH_EVENT']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

if selection == 'Decision Tree Classifier':

    # DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    ce_ord = ce.OrdinalEncoder(cols = feature_cols)
    X_cat = ce_ord.fit_transform(X_train)

    ce_oh = ce.OneHotEncoder(cols = feature_cols)
    X_cat_oh = ce_oh.fit_transform(X_train)


    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)
    
elif selection == 'Support vector machine (SVM)':
    # SVC
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)

    ce_ord = ce.OrdinalEncoder(cols=feature_cols)
    X_cat = ce_ord.fit_transform(X_train)

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

else:
    #Guassian processes:
    gp_classifier = GaussianProcessClassifier(kernel=RBF())
    gp_classifier.fit(X_train, y_train)

    ce_ord = ce.OrdinalEncoder(cols=feature_cols)
    X_cat = ce_ord.fit_transform(X_train)


    y_pred = gp_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)