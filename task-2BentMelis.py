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
from sklearn import metrics

header = st.container()

with header:
    st.title("Taak 2 AI - Bent Melis (r0831245)")

selection = st.selectbox(
    'Select the ML techniques from the scikitLearn library you want to test:',
    ('Decision Tree Classifier', 'Support vector machine (SVM)','Gaussian Processes')
)

heart_failure_df = pd.read_csv("resources/heart_failure_clinical_records_dataset.csv", sep=',')

feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase','diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'sex', 'smoking','time']

X = heart_failure_df[feature_cols]
y = heart_failure_df[['DEATH_EVENT']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

if selection == 'Decision Tree Classifier':
    st.text('')

    st.text('Some extra info:')
    st.text('A Decision Tree Classifier is a type of supervised learning algorithm that is')
    st.text('mostly used for classification problems. It works for both categorical and')
    st.text('continuous input and output variables. In this technique, we split the population')
    st.text(' or sample into two or more homogeneous sets (or sub-populations) based on the most')
    st.text('significant splitter/differentiator in input variables')

    st.text('')

    st.text("I have used this technique and after testing the accuracy was:")

    # DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    ce_ord = ce.OrdinalEncoder(cols = feature_cols)
    X_cat = ce_ord.fit_transform(X_train)

    ce_oh = ce.OneHotEncoder(cols = feature_cols)
    X_cat_oh = ce_oh.fit_transform(X_train)


    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.text(str(round(accuracy),2))
    st.text("Root Mean Squared Error =", str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)))


elif selection == 'Support vector machine (SVM)':
    st.text('')

    st.text('Some extra info:')
    st.text('SVMs are a set of supervised learning methods used for classification, regression') 
    st.text(' and outliers detection. They are effective in high dimensional spaces and best') 
    st.text('suited for problems with complex domains where there are clear margins of')
    st.text('separation in the data. To correctly classify the data, this method finds the')
    st.text('hyperplane in an N-dimensional space that distinctly classifies the data points.')

    st.text('')

    st.text("I have used this technique and after testing the accuracy was:")
    # SVC
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)

    ce_ord = ce.OrdinalEncoder(cols=feature_cols)
    X_cat = ce_ord.fit_transform(X_train)

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.text(str(round(accuracy),2))
    st.text("Root Mean Squared Error =", str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)))

else:
    st.text('')

    st.text('Some extra info:')
    st.text('Gaussian Processes (GP) are a generic supervised learning method designed to solve')
    st.text('regression and probabilistic classification problems. They are highly flexible,') 
    st.text('and can model various kinds of data. Gaussian Processes provide a principled,')
    st.text('practical, probabilistic approach to learning in kernel machines.')

    st.text('')

    st.text("I have used this technique and after testing the accuracy was:")
    #Guassian processes:
    gp_classifier = GaussianProcessClassifier(kernel=RBF())
    gp_classifier.fit(X_train, y_train)

    ce_ord = ce.OrdinalEncoder(cols=feature_cols)
    X_cat = ce_ord.fit_transform(X_train)


    y_pred = gp_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.text(str(round(accuracy),2))
    st.text("Root Mean Squared Error =", str(round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)))