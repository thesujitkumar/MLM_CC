import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



df=pd.read_csv("test_predictions.csv")
print(df.columns)
#df['output'].replace(label_mapping, inplace=True)
df1=pd.read_csv("FNC_Prompt_Tuning_Results.csv")
print(df1.columns )
#df1['label'] #.replace(label_mapping, inplace=True)


gt=df1['predictions']
print("the lenght of ground truth",len(gt))

pred=df['Label']
print("the lenght of predictions",len(pred))

f1=f1_score(gt, pred, average='macro')
print("the f1 macro score is ",f1)


f1score=f1_score(gt, pred, average=None)
print("classwise f1 score of [ fake,true] is",f1score)


accuracy=accuracy_score(pred, gt)
print("accuracy of classifier is",accuracy)

a=confusion_matrix(gt, pred, labels=[0, 1])
print("confusion matrix",a)

#print(gt)
#print(pred)
b=precision_score(gt, pred, average=None)
print("precision is",b)
c=recall_score(gt, pred, average=None)
print("recall is",c)



fpr, tpr, thresholds = metrics.roc_curve(gt, pred, pos_label=2)
metrics.auc(fpr, tpr)
