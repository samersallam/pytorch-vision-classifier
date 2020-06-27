from sklearn import datasets, svm
import numpy as np
import pandas as pd
import copy
import pickle as pk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from imblearn.metrics import specificity_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy import interp
from matplotlib import pyplot as plt
import seaborn as sns
from classification_analysis.classification_analysis import ClassificationAnalysis


# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters â€˜macroâ€™ to account for label imbalance; it can result in an F-score that is not between precision and recall.

class ClassifierReport:

    def __init__(self, y_true, y_pred, y_score , number_of_classes, average_type = 'macro', 
                 digits_count_fp = 3,classes_labels = None):
        """ Initialization function
            y_true: list or numpy array (number of samples)
            y_pred: list or numpy array (number of samples)
            y_score: numpy array contains the actual outputs before decision (number of samples, number of classes)
            average_type: determine how to calculate the overall metrics
            digits_count_fp: number of digits after the floating point
            classes_labels: list of classes labels
        """
        
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_score = np.array(y_score)
        self.number_of_classes = number_of_classes
        self.y_true_one_hot = label_binarize(y_true, classes=list(range(self.number_of_classes)))
        
        self.number_of_samples = len(self.y_true)
        self.number_of_samples_per_class = [(self.y_true==c).sum() 
                                for c in range(self.number_of_classes)]
        
        self.classes_labels = ['Class ' + str(c) for c in range(self.number_of_classes)] \
                                if classes_labels is None \
                                else classes_labels
        
        self.digits_count_fp = digits_count_fp
        self.average_type = average_type
        
        self.TP_TN_FP_FN()
        self.calculate_confusion_matrix()
        self.calculate_confusion_tables()
        self.accuracy()
        self.recall()
        self.precision()
        self.f1_score()
        self.specificity()
        self.cohen_kappa()
        self.calculate_roc_auc()
        
        
    def TP_TN_FP_FN(self):
        self.TP = np.zeros(self.number_of_classes)
        self.FP = np.zeros(self.number_of_classes)
        self.TN = np.zeros(self.number_of_classes)
        self.FN = np.zeros(self.number_of_classes)
        
        for cls in range(self.number_of_classes):
            # Calculate
            self.TP[cls] = (self.y_pred[self.y_true == cls] == cls).sum()
            self.FN[cls] = (self.y_pred[self.y_true == cls] != cls).sum()
            
            self.TN[cls] = (self.y_pred[self.y_true != cls] != cls).sum()
            self.FP[cls] = (self.y_pred[self.y_true != cls] == cls).sum()            
    
    def calculate_confusion_matrix(self):
        """ Function to calculate confusion matrix and weighted confusion matrix """
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        
        classes_weights = np.array(self.number_of_samples_per_class).reshape(
            self.number_of_classes, 1)
        
        self.normalized_confusion_matrix = (self.confusion_matrix/classes_weights).round(self.digits_count_fp)
    
    def calculate_confusion_tables(self):
        """ Function to calculate confusion table and weighted confusion table 
            for each class
        """
        self.confusion_tables = np.zeros((self.number_of_classes, 2, 2))
        self.normalized_confusion_tables = np.zeros((self.number_of_classes, 2, 2))
        
        for cls in range(self.number_of_classes):
            # Normal confusion table
            self.confusion_tables[cls, 0, 0] = self.TP[cls] # TP
            self.confusion_tables[cls, 0, 1] = self.FP[cls] # FP
            self.confusion_tables[cls, 1, 0] = self.FN[cls] # FN
            self.confusion_tables[cls, 1, 1] = self.TN[cls] # TN
            
            # Weighted confusion table
            table_weights = self.confusion_tables[cls].sum(axis=0).reshape(1, 2)
            self.normalized_confusion_tables[cls] = (self.confusion_tables[cls]/table_weights).round(self.digits_count_fp)
        
        # Convert the data type into int
        self.confusion_tables = self.confusion_tables.astype(int)
    
    def accuracy(self, sample_weight = None):
        """ Refer to sklearn for full doc"""
        if sample_weight is None:
            sample_weight  = np.ones(self.number_of_samples)
        self.overall_accuracy = accuracy_score(
            self.y_true, self.y_pred, sample_weight=sample_weight).round(self.digits_count_fp)
        
    def recall(self):
        """Recall is also known as Sensitivity and True Positive Rate"""
        self.overall_recall = recall_score(
            self.y_true, self.y_pred, average = self.average_type).round(self.digits_count_fp)
        self.classes_recall = recall_score(
            self.y_true, self.y_pred, average = None).round(self.digits_count_fp)
        
    def precision(self):
        """ Precision or Positive Predictive Value """
        self.overall_precision = precision_score(
            self.y_true, self.y_pred, average = self.average_type).round(self.digits_count_fp)
        self.classes_precision = precision_score(
            self.y_true, self.y_pred, average = None).round(self.digits_count_fp)
    
    def f1_score(self):
        """ f1_score is harmonic mean of recall and precision"""
        self.overall_f1_score = f1_score(
            self.y_true, self.y_pred, average = self.average_type).round(self.digits_count_fp)
        self.classes_f1_score = f1_score(
            self.y_true, self.y_pred, average = None).round(self.digits_count_fp)
    
    
    def specificity(self):
        """ Specificity is also known as True Negative Rate """
        self.overall_specificity = specificity_score(
            self.y_true, self.y_pred, average = self.average_type).round(self.digits_count_fp)
        self.classes_specificity = specificity_score(
            self.y_true, self.y_pred, average = None).round(self.digits_count_fp)
    
    def cohen_kappa(self):
        self.overall_cohen_kappa = cohen_kappa_score(self.y_true, self.y_pred).round(self.digits_count_fp)    
        
    def calculate_roc_auc(self):
        """ Refer to : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        """
        if self.number_of_classes == 2:
            self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_score[:,1])
            self.roc_auc = auc(self.fpr, self.tpr).round(self.digits_count_fp)
            
            # Rounding
            self.fpr = self.fpr.round(self.digits_count_fp)
            self.tpr = self.tpr.round(self.digits_count_fp)
            self.thresholds = self.thresholds.round(self.digits_count_fp)
            return 
        
        # Compute ROC curve and ROC area for each class
        self.fpr = dict()    # fpr: False positive rate
        self.tpr = dict()    # tpr: True positive rate
        self.roc_auc = dict()
        for i in range(self.number_of_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(
                self.y_true_one_hot[:, i], self.y_score[:, i])
            self.roc_auc[i] = auc(self.fpr[i] , self.tpr[i]).round(
                self.digits_count_fp).round(self.digits_count_fp)
            
            # Rounding
            self.fpr[i] = self.fpr[i].round(self.digits_count_fp)
            self.tpr[i] = self.tpr[i].round(self.digits_count_fp)


        # Compute micro-average ROC curve and ROC area
        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(
            self.y_true_one_hot.ravel(), self.y_score.ravel())
        self.roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"]).round(self.digits_count_fp)
        
        # Rounding
        self.fpr["micro"] = self.fpr["micro"].round(self.digits_count_fp)
        self.tpr["micro"] = self.tpr["micro"].round(self.digits_count_fp)
        
        # Compute macro-average ROC curve and ROC area
        
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.number_of_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.number_of_classes):
            mean_tpr += interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.number_of_classes
        
        # Rounding
        self.fpr["macro"] = all_fpr.round(self.digits_count_fp)
        self.tpr["macro"] = mean_tpr.round(self.digits_count_fp)
        self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"]).round(self.digits_count_fp)
        
    def show_confusion_matrix(self):
        
        self.calculate_confusion_matrix()
        # fig = plt.figure(figsize=(8,3))
        fig = plt.figure(figsize=(10,4))
        ax = plt.subplot(1,2,1)
        ClassificationAnalysis.plot_confusion_matrix(self.confusion_matrix, 
                                                   self.classes_labels,
                                                   title = 'Confusion Matrix',
                                                   cmap = plt.cm.Blues,
                                                   figure_axis = ax)
        
        ax = plt.subplot(1,2,2)
        ClassificationAnalysis.plot_confusion_matrix(self.normalized_confusion_matrix, 
                                                   self.classes_labels,
                                                   title = 'Normalized Confusion Matrix',
                                                   cmap = plt.cm.Blues,
                                                   figure_axis = ax)
        fig.tight_layout()
        plt.show()
    
    def show_confusion_tables(self):
        self.calculate_confusion_tables()
        fig = plt.figure(figsize=(8,self.number_of_classes*2))
        table_counter = 0
        for cls in range(self.number_of_classes):
            table_counter += 1
            ax = plt.subplot(self.number_of_classes, 3, table_counter)
            plt.grid(False)
            ClassificationAnalysis.plot_confusion_table_sample(ax)
            
            table_counter += 1
            ax = plt.subplot(self.number_of_classes, 3, table_counter)
            ClassificationAnalysis.plot_confusion_matrix(self.confusion_tables[cls], 
                                                       None,
                                                       title = self.classes_labels[cls],
                                                       cmap = plt.cm.Blues,
                                                       figure_axis=ax)
            
            table_counter += 1
            ax = plt.subplot(self.number_of_classes, 3, table_counter)
            ClassificationAnalysis.plot_confusion_matrix(self.normalized_confusion_tables[cls], 
                                                       None,
                                                       title = self.classes_labels[cls],
                                                       cmap = plt.cm.Blues,
                                                       figure_axis = ax)
        fig.tight_layout()
        plt.show()
    
    def show_all(self):
        new_lines = '\n'
        
        print('Confusion Matrix' + new_lines)
        self.show_confusion_matrix()
        
        print(new_lines + 'Confusion Tables' + new_lines)
        self.show_confusion_tables()
        
        # Overall
        overall_metrics = {
            'Accuracy': self.overall_accuracy,
            'Recall': self.overall_recall,
            'Precision': self.overall_precision,
            'F1_score': self.overall_f1_score,
            'Specificity': self.overall_specificity,
            'Cohen_Kappa': self.overall_cohen_kappa
            }
        print(new_lines + 'Overall Metrics' + new_lines)
        ClassificationAnalysis.show_overall_metrics(overall_metrics)
        
        # Per class
        classes_metrics = {
            'Recall': self.classes_recall,
            'Precision': self.classes_precision,
            'F1_score': self.classes_f1_score,
            'Specificity': self.classes_specificity
            }
        print(new_lines + 'Classes Metrics' + new_lines)
        ClassificationAnalysis.show_classes_metrics(classes_metrics, self.classes_labels)
        
        print(new_lines + 'ROC Curve' + new_lines)
        ClassificationAnalysis.show_roc_curve(self.number_of_classes, self.fpr, self.tpr, self.roc_auc)

# a = ClassifierReport(y_true, y_pred, y_score, 
#                       classes_labels=dataset.target_names.tolist())
# a.show_all()

# a = ClassifierReport(y_true, y_pred, y_score,
#                       classes_labels=classes_labels)
# a.show_all()

# print(np.array(a.TP, dtype=int))
# print(np.array(a.FP, dtype=int))
# print(np.array(a.FN, dtype=int))
# print(np.array(a.TN, dtype=int))

print('Importing Done ...')