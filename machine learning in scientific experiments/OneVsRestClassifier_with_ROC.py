import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from numpy import interp

# to acquire the dataset
MOX_path="./MOX Conclusion_test.csv"
MOX_data=pd.read_csv(MOX_path)

#ESSE PROCESSAMENTO ATÉ O FINAL DO ARQUIVO PODERIA SER FEITO EM UMA FUNÇÃO DETERMINÍSTICA
#BASTARIA REMOVER AS MENSAGENS DE PRINT
#É NECESSÁRIO TESTAR POIS UTILIZA ALGUNS MODELOS PREDITIVOS QUE NÃO TENHO CERTEZA SE SÃO DETERMINÍSTICOS
MOX_features=["Humidity"
              ,"R1_Up_Slope","R1_Down_Slope","R2_Up_Slope","R2_Down_Slope","R3_Up_Slope","R3_Down_Slope"
              ,"R4_Up_Slope","R4_Down_Slope","R5_Up_Slope","R5_Down_Slope","R6_Up_Slope","R6_Down_Slope"
              ,"R7_Up_Slope","R7_Down_Slope","R8_Up_Slope","R8_Down_Slope","R9_Up_Slope","R9_Down_Slope"
              ,"R10_Up_Slope","R10_Down_Slope","R11_Up_Slope","R11_Down_Slope","R12_Up_Slope","R12_Down_Slope"
              ,"R13_Up_Slope","R13_Down_Slope","R14_Up_Slope","R14_Down_Slope"]
X=MOX_data[MOX_features] # X as feature
Y=MOX_data.CO_Concentration # Y as prediction target
label_Y=label_binarize(Y, classes=[0,1,2,3,4,5,6,7,8,9]) # CO concentration 0, 2.22, 4.44, 6.67, 8.89, 11.11, 13.33, 15.56, 17.78, 20 are replaced by 0 to 9
CO_classes = label_Y.shape[1] # to set all the classes for CO concentration

#train_test_split NÃO É DETERMINÍSTICO POIS DIVIDE AS AMOSTRAS DE TREINAMENTO E TESTE ALEATORIAMENTE
#svm.SVC TAMBÉM PARECE NÃO SER DETERMINÍSTICO
train_X, valid_X, train_Y, valid_Y = train_test_split(X, label_Y)
MOX_classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=np.random.RandomState(0)))
y_score = MOX_classifier.fit(train_X, train_Y).decision_function(valid_X)

# to get the fpr and tpr
#ESSE PROCESSAMENTO ATÉ A CRIAÇÃO DO GRÁFICO (plt.figure()) PARECE SER DETERMINÍSTICO
#MAS É NECESSÁRIO TESTAR
fpr=dict()
tpr=dict()
roc_auc=dict()
for one_class in range(CO_classes):
    fpr[one_class], tpr[one_class], thresholds = roc_curve(valid_Y[:, one_class], y_score[:, one_class])
    roc_auc[one_class] = auc(fpr[one_class], tpr[one_class])
fpr_assemble = np.unique(np.concatenate([fpr[one_class] for one_class in range(CO_classes)]))
tpr_sum = np.zeros_like(fpr_assemble)
for one_class in range(CO_classes):
    tpr_sum+= interp(fpr_assemble, fpr[one_class], tpr[one_class])
tpr_average=tpr_sum/CO_classes
fpr["macro"] = fpr_assemble
tpr["macro"] = tpr_average
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# to create the ROC figure
plt.figure(figsize=[10,6])
ROC_Color=['red', 'darkorange', 'yellow',"limegreen","darkgreen","cyan","dodgerblue","blue","darkviolet","magenta"]
for one_class, one_color in zip(range(CO_classes), ROC_Color):
    plt.plot(fpr[one_class], tpr[one_class], color=one_color, lw=1,label='ROC fold {0} (AUC = {1:0.2f})'.format(one_class, roc_auc[one_class]))
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()