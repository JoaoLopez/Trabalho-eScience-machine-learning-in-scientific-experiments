Artificial Intelligence for Driving Scientific Experiments
——THE COMPARISON BETWEEN ACTIVE MACHINE LEARNING AND RANDOM SELECTION OF EXPERIMENTS

Code creater: Shikun Luo, Beijing University of Technology

Source code:
data_prepare.py
decision_tree.py
random_forest.py
KNN.py
OneVsRestClassifier_with_ROC.py
SVM_activeLearning.py

Introduction:
Before the use of the code, it's necessary to download the dataset (URL: https://www.kaggle.com/javi2270784/gas-sensor-array-temperature-modulation) and put it in the folder.

The data_prepare.py is used to organize the original dataset and create a new dataset which can be used directly in the later predictive model. It can create a new file named MOX Conclusion.csv. Since it's a time-consuming job, the MOX Conclusion.csv is directly offered in the folder. By manually revising the CO concentration in MOX Conclusion.csv from 0, 2.22, 4.44, 6.67, 8.89, 11.11, 13.33, 15.56, 17.78, 20 to 0 to 9 and keep others unchanged, we can create MOX Conclusion_test.csv.

The decision_tree.py uses the function of DecisionTreeClassifier. The mean_absolute_error and Classifier accuracy of different max_leaf_nodes can be acquired.

Similarly, random_forest.py and KNN.py can calculate the accuracy of prediction with the use of RandomForestClassifer and KNeighborsClassifier.

OneVsRestClassifier_with_ROC.py uses the function of OneVsRestClassifier and it can also print a ROC curve.

SVM_activeLearning.py uses the function of Support Vector Machine. It prints the graphs of 30 times active learning. It also prints the learning curves of active learning and traditional machine learning of the experiments in the end. Part of this file makes some references to the mind of the code (URL: https://www.kaggle.com/akhileshravi/active-learning-tutorial).

Picture sample folder provides an example of the picture created in SVM_activeLearning.py.

