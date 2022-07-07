import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


st.title('Model Deployment : Logistic Regression')
print('Model : Logistic Regression')
st.sidebar.header('User input Parameters')

def user_input_features():
    Industrial_Risk=st.sidebar.radio('Industrial_Risk',('0','0.5','1.0'))
    Management_Risk=st.sidebar.radio('Management_Risk',('0','0.5','1.0'))
    Financial_Flexibility=st.sidebar.radio('Financial_Flexibility',('0','0.5','1.0'))
    Credibility=st.sidebar.radio('Credibility',('0','0.5','1.0'))
    Competitiveness=st.sidebar.radio('Competitiveness',('0','0.5','1.0'))
    Operating_Risk=st.sidebar.radio('Operating_Risk',('0','0.5','1.0'))
    data={'Industrial_Risk':Industrial_Risk, 
          'Management_Risk':Management_Risk,
          'Financial_Flexibility':Financial_Flexibility,
          
          'Credibility':Credibility,
          'Competitiveness':Competitiveness,
          'Operating_Risk':Operating_Risk}
    features=pd.DataFrame(data,index={1})
    return features
df=user_input_features()
st.subheader('User input Parameters')
st.write(df)

bankruptcy = pd.read_csv("C:/Users/Anjali/Downloads/bankruptcy-prevention.csv",delimiter=';')
#Rename Column
bankruptcy = bankruptcy.rename({' class': 'Class',
                          'industrial_risk':'Industrial_Risk',
                          ' management_risk':'Management_Risk',
                          ' financial_flexibility':'Financial_Flexibility',
                          ' credibility':'Credibility',
                          ' competitiveness':'Competitiveness',
                          ' operating_risk':'Operating_Risk'}, axis=1)

labelencoder = LabelEncoder()
bankruptcy.iloc[:,-1] = labelencoder.fit_transform(bankruptcy.iloc[:,-1])

X = bankruptcy.iloc[:,0:6]
Y = bankruptcy.iloc[:,6]

classifier = LogisticRegression()
classifier.fit(X,Y)

y_pred=classifier.predict(df)
y_proba=classifier.predict_proba(df)

st.subheader('Predicted Result')
st.write('Non-Bankrupt'if y_proba[0][1]>0.5 else 'Bankrupt')
print('Non-Bankrupt'if y_proba[0][1]>0.5 else 'Bankrupt')   
print(y_proba)

st.subheader('Additional Details')
#st.subheader('Prediction Probability')
if st.button('Click here for Prediction Probability'):
   st.write(y_proba)


