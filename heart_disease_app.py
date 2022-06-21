# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:09:53 2022

@author: User
"""

import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os

BEST_MODEL_PATH = os.path.join(os.getcwd(),'model','best_model_svc.pkl')
with open(BEST_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# output : 0 = less chance of heart attack 
        #: 1 = more chance of heart attack
        
# X_new = ['cp','thall','age','thalachh','oldpeak']

#%%
with st.form("Patient's Info"):
    st.header("This app is to predict if a person has Heart Disease :smiley:")
    st.caption('Have you ever wonder how close you are to Heart Disease?')

    image = Image.open('heart.jpg')
    st.image(image, caption='Keep Smiling',use_column_width=True)
    
    age = st.number_input('Age')
    thalachh = int(st.number_input('Maximum Heart Rate achieved (thalachh)'))
    oldpeak = int(st.number_input('ST depression (oldpeak)'))
    cp = int(st.radio("What's your Chest Pain type(cp)? 0: typical angina \
                      1: atypical angina 2: non-anginal pain \
                       3: asymptomatic",(0,1,2,3)))    

    thall = int(st.radio("What's your blood disorder:thalassemia(thall)? \
                         1: fixed defect 2:normal blood flow,\
                          3: reversible defect",(1,2,3)))
    
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        
        st.write("Age", age, "MaxHeartRate",thalachh,
                 "ST Depression",oldpeak, "Chest Pain",cp,
                 "Blood Disorder(Thalassemia)",thall)
        
        X_new = [age,thalachh,oldpeak,cp,thall]
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
                
        outcome_dict = {0:'Less chance of heart attack ',
                        1:'High chance of heart attack'}
        
        st.write(outcome_dict[outcome[0]])      
        if outcome == 1:
            st.snow()
            st.write('You better take care of your health as \
                     you have a higher chance of getting Heart Disease')

        else:
            st.balloons()
            st.write('Good you have lesser chance, but according to statistics,\
                     you still have a chance of getting it')








