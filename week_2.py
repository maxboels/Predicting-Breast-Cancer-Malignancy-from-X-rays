# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:07:22 2020

@author: boels
"""

'---------------------------------EVALUATION-----------------------------------'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import util


class_labels = ['malignancy'] * 10
pred_labels = ['predictions'] * 10

thresholds = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]



test_results = pd.read_csv("Output/df_test.csv")



test_results.head()

# ground truth
y = test_results[class_labels].values
# predicted labels
pred = test_results[pred_labels].values


test_results[np.concatenate([class_labels, pred_labels])].head()

df = util.get_performance_metrics(y, pred, class_labels, 
                                  acc=util.get_accuracy, 
                                  prevalence=util.get_prevalence, 
                                  sens=util.get_sensitivity, 
                                  spec=util.get_specificity, 
                                  ppv=util.get_ppv, 
                                  npv=util.get_npv, 
                                  auc=util.roc_auc_score,
                                  f1=f1_score, 
                                  thresholds=thresholds)




util.get_curve(y, pred, class_labels)


#Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced.
util.get_curve(y, pred, class_labels, curve='prc')

util.plot_calibration_curve(y, pred, class_labels)

# plot historam
plt.hist(pred, bins=13)
plt.legend(['Malignancy'], loc='upper right',)
plt.ylabel('Count')
plt.xlabel('Class Probabilities')
plt.title('Predicted Class Probabilities Histogram')
plt.tight_layout()



# save to csv
df.to_csv("Output/df_Evaluation_Metrics_10.csv", index=None)


