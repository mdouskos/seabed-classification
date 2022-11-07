import sklearn.metrics as metr
import numpy as np
import pandas as pd

# import openpyxl


def compute_cm(y_gt, y_pred, labels=None, round_prec=2, cls_names=None, xls_path=None):
    # compute metrics
    cm = metr.confusion_matrix(y_gt, y_pred)
    # kappa = metr.cohen_kappa_score(y_gt, y_pred)
    f1_score = metr.f1_score(y_gt, y_pred, average="macro")
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
            if cls_names is not None:
                assert len(cls_names) == sz1, f"{sz1} class names should be provided"
                labels.append(cls_names[i - 1])
            else:
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
            sublist[-2] = "f1:"
            sublist[-1] = round(f1_score, round_prec)
        elif idx == sz1 + 1:
            sublist[-2] = "OA:"
            sublist[-1] = round(OA, round_prec)
            cm_list[idx] = sublist
        idx += 1

    # Convert to data frame
    df = pd.DataFrame(np.array(cm_list))
    df.columns = first_row
    df.index = first_col

    if xls_path:
        writer = pd.ExcelWriter(xls_path)
        df.to_excel(writer, "CM")
        writer.save()

    return df


def compute_metrics(y, predictions, polygons=None):
    # Compute metrics for each run
    accuracy = (predictions == y).sum().item() / y.size
    kappa = metr.cohen_kappa_score(y, predictions)
    f1 = metr.f1_score(y, predictions, average="macro")

    acc = None
    # Compute polygon accuracy
    if polygons is not None:
        polygon_ids = np.unique(polygons)
        acc = []
        for vp in polygon_ids:
            vp_inds = polygons == vp
            polygon_predictions, prediction_counts = np.unique(
                predictions[vp_inds], return_counts=True
            )
            polygon_predicted_class = polygon_predictions[np.argmax(prediction_counts)]
            if polygon_predicted_class == y[vp_inds][0]:
                acc.append(True)
            else:
                acc.append(False)

    return {"accuracy": accuracy, "kappa": kappa, "f1-score": f1, "polygon": acc}
