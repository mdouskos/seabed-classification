import sklearn.metrics as metr
import numpy as np
import pandas as pd

# import openpyxl


def _get_cm(y_gt, y_pred, labels=None, round_prec=2):
    # compute metrics
    cm = metr.confusion_matrix(y_gt, y_pred)
    kappa = metr.cohen_kappa_score(y_gt, y_pred)
    OA = metr.accuracy_score(y_gt, y_pred)
    UA = metr.precision_score(y_gt, y_pred, average=None)
    # UA_avg  = metr.precision_score   (y_gt, y_pred, average='weighted')
    PA = metr.recall_score(y_gt, y_pred, average=None)
    # PA_avg  = metr.recall_score      (y_gt, y_pred, average='weighted')

    # confusion matrix with UA, PA
    sz1, sz2 = cm.shape
    cm_with_stats = np.zeros((sz1 + 2, sz2 + 2))
    cm_with_stats[0:-2, 0:-2] = cm
    cm_with_stats[-1, 0:-2] = np.round(UA, round_prec)
    cm_with_stats[0:-2, -1] = np.round(PA, round_prec)
    cm_with_stats[-2, 0:-2] = np.sum(cm, axis=0)
    cm_with_stats[0:-2, -2] = np.sum(cm, axis=1)

    # convert to list
    cm_list = cm_with_stats.tolist()

    if labels is None:
        labels = []
        for i in range(1, sz1 + 1):
            labels.append("Class " + str(i))
    
    # first row
    first_row = []
    first_row.extend(labels)
    first_row.append("sum")
    first_row.append("PA")

    # first col
    first_col = []
    first_col.extend(labels)
    first_col.append("sum")
    first_col.append("UA")

    # fill rest of the text
    idx = 0
    for sublist in cm_list:
        if idx == sz1:
            cm_list[idx] = sublist
            sublist[-2] = "kappa:"
            sublist[-1] = round(kappa, round_prec)
        elif idx == sz1 + 1:
            sublist[-2] = "OA:"
            sublist[-1] = round(OA, round_prec)
            cm_list[idx] = sublist
        idx += 1

    # Convert to data frame
    df = pd.DataFrame(np.array(cm_list))
    df.columns = first_row
    df.index = first_col

    return df

def confusion_matrix(datapath, pred_nm, gt_nm, xls_nm, labels=None, sheet_nm="CM"):

    # load data
    # (pred, geoTransform, proj, drv_name) = geoimread(datapath + pred_nm)
    pred = np.load(pred_nm)
    gt = np.load(gt_nm)

    # convert to int
    gt = gt.astype("int")
    pred = pred.astype("int")

    # remove background pixels
    gt0 = gt > 0
    y_gt = gt[gt0]
    y_pred = pred[gt0]

    pr0 = y_pred > 0
    y_gt = y_gt[pr0]
    y_pred = y_pred[pr0]

    df = _get_cm(y_gt, y_pred, labels)

    # Write to xls
    writer = pd.ExcelWriter(datapath + "/" + xls_nm + ".xlsx")
    df.to_excel(writer, sheet_nm)
    writer.save()

    return df