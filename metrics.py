import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, balanced_accuracy_score, accuracy_score ## also imports the balanced acc


######################################## METRICS FUNCTIONS #########################################
## Accuracy function
def acc(y_true:torch.tensor, y_pred:torch.tensor):
    '''calculate the accuracy score'''

    return accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

## Balanced accuracy function
def bacc(y_true:torch.tensor, y_pred:torch.tensor):
    '''calculates the balanced accuracy score'''

    return balanced_accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())



def compute_roc_curve(y_true:torch.tensor, y_probs:torch.tensor):
    '''calculares the receiver operating characteristic curve (ROC)'''

    #assumes that y_probs is a probability tensor, not a tensor containing predictions

    fpr, tpr , thrs = roc_curve(y_true.cpu().numpy(), y_probs.cpu().numpy(), pos_label=1)


    return fpr, tpr, thrs



## Calculates precision, recall, f1 and macro AUC
def calculate_other_metrics(labels:torch.tensor, predictions:torch.tensor):
    '''calculates precision, recall, f1 and macro AUC'''


    precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
    recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
    f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
    
    # Compute ROC curve and AUC for each class
    num_classes = len(torch.unique(labels))
    fpr = []
    tpr = []
    roc_auc = []
    for class_idx in range(num_classes):
        class_labels = (labels.cpu().numpy() == class_idx)
        class_predictions = (predictions.cpu().numpy() == class_idx)
        fpr_i, tpr_i, _ = roc_curve(class_labels, class_predictions, pos_label=1)
        roc_auc_i = auc(fpr_i, tpr_i)
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        roc_auc.append(roc_auc_i)
    
    # Calculate macro-average AUC
    macro_auc = sum(roc_auc) / num_classes
    
    return precision, recall, f1, macro_auc




