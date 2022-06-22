![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)


# MachineLearning_Classification_to predict the Heart Disease

### Descriptions
 1) To predict if someone have heart disease or have potential to have heart attack
 2) The datasets contains 303 train dataset with 10 test set
 3) This dataset contains continuous and categorical data
 4) The target is categorical either 0 or 1, less potential to have heart attack, high potential to have heart attack
 5) There are different ML approaches to select the best features in order to have an accurate prediction
 6) For Continuos Data Versus Categorical Data, used : `Logistic Regression`
 7) For Categorical Data Versus Categorical Data, used : `Cramer's V`
 
### Results
`Model`: When tune the model using `Standard Scaler` & `SVC`

![Fine_tune](https://user-images.githubusercontent.com/106902414/174779358-a9dcc302-1fac-4c98-9755-587bc32f3e7a.PNG)

![Fine tune](https://user-images.githubusercontent.com/106902414/174779383-5a33597f-31c2-472a-bc4d-36925a4f8710.PNG)


`Classification_Report` without fine tune the model

![Model Analysis](https://user-images.githubusercontent.com/106902414/174779132-11793148-e54c-4c70-bf27-c08f7b44c1fe.PNG)




`Confusion_Matrix` without fine tune the model


<img src="plot/ConfusionMatrixDis.png" alt="model" style="width:300px;height:250px;">


`Training Data` example for continuous data:

<img src="plot/age.png" alt="model" style="width:300px;height:200px;">


`Training Data` example for categorical data:

<img src="plot/cp.png" alt="model" style="width:300px;height:200px;">

### Discussion
1) Machine Learning best pipeline for this dataset is: `Standard Scaler` + `SVC`
2) When tuning the model the accuracy decreases, thus we choose the best model without tuning it
3) The prediction is quite acuurate during deployment
4) For model improvement: Try with different Scalling and Classifier, also different the random_state when train-test-split the data

### Credits
`You can load the dataset from here`
https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
