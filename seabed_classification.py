import argparse
import pathlib

import logging

import numpy as np

from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow

from libs.seabed_classifier import seabed_classifier_type, seabed_classifier_name

import pandas as pd

from copy import deepcopy

from utils.reporting import _get_cm
from utils.dataset import (
    split_by_polygon_id,
    hist_match,
)


from sklearn.metrics import cohen_kappa_score

from time import monotonic

from libs.survey_data import SurveyData

# np.random.seed(1234)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Seabed classification.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Dataset directory (leave empty for current dir)",
    )
    parser.add_argument(
        "--inputs", type=str, nargs="*", help="Input areas (JSON files)"
    )
    parser.add_argument(
        "--validation",
        type=str,
        nargs="*",
        help="Validation areas (JSON files), leave empty for validation on input areas",
    )
    parser.add_argument("--runs", type=int, help="Number of runs", default=1)
    parser.add_argument(
        "--perc",
        type=float,
        help="Percentage of training in decimal form ex. 0.7",
        default=0.7,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specify model type",
        default="rf",
        choices=("nn", "rf", "svm"),
    )
    parser.add_argument(
        "--normalize",
        type=str,
        help="Normalization type",
        default="std",
        choices=("none", "minmax", "std"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Specify features to use (bs: backscatter only, bathy: bathymetr, bpi: BPI)",
        default="bs",
        choices=("bs", "bathy", "bpi"),
    )
    parser.add_argument("--hist-match", action="store_true", help="Histogram matching")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="Specify positional embedding dimension, 0 for no embedding",
        default=0,
    )
    parser.add_argument(
        "--embedding-sigma",
        type=float,
        help="Specify positional embedding sigma",
        default=1.0,
    )
    parser.add_argument(
        "--epochs", type=int, help="Neural network training epochs", default=10
    )
    parser.add_argument(
        "--batch", type=int, help="Batch size for training MLP", default=2048
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--class-map", action="store_true", help="Compute classification map"
    )
    parser.add_argument(
        "--plot-aligned", action="store_true", help="Plot aligned data layers"
    )
    parser.add_argument(
        "--input-val", type=str, help="Path of validation image", default=""
    )
    parser.add_argument(
        "--sample-radius",
        action="store_true",
        help="Radius around samples (in meters)",
        default=1,
    )
    parser.add_argument("--output-dir", type=str, help="Output directory", default="./")
    parser.add_argument(
        "--xls-file", type=str, help="XLS filename for saving confusion matrix"
    )
    parser.add_argument(
        "--track", action="store_true", help="Track experiments with mlflow"
    )
    return parser.parse_args()


def main():

    args = parse_arguments()
    dataset_path = pathlib.Path(args.dataset_dir) / args.inputs[0]
    region = SurveyData(dataset_path, dataset_dir=args.dataset_dir, mode=args.mode)

    data_train = region.get_data(
        pe_dim=args.embedding_dim, pe_sigma=args.embedding_sigma
    )

    X = data_train["data"]
    y = data_train["gt"]
    polys = data_train["polygons"]

    print("Number of samples: %d" % len(y))
    with (mlflow.start_run() if args.track else nullcontext()):
        if args.track:
            param_dict = {
                "Dataset": args.dataset_dir,
                "Mode": args.mode,
                "Normalization": args.normalize,
                "Runs": args.runs,
                "Split": args.perc,
                "PEdim": args.embedding_dim,
                "Sigma": args.embedding_sigma,
            }
            if args.model == "nn":
                param_dict.update({"Epochs": args.epochs, "Batch": args.batch})
            mlflow.log_params(param_dict)
            mlflow.set_tag("Model", args.model)
            mlflow.set_tag("PE", args.embedding_dim > 0)
        acc_res = []
        kappa_res = []
        best_result = None
        pred_all = None
        if not args.input_val:
            data_valid = data_train
        else:
            ValueError("Not implemented yet")
            # data_valid = process_data(
            #     args.input_val,
            #     args.gt_val,
            #     bathymetry_file=args.bathy_val,
            #     normalize=args.normalize,
            # )

        for cnt, run in enumerate(range(args.runs)):
            print("Run %d/%d" % (run + 1, args.runs))
            if not args.input_val:
                train_set, val_set, train_sum, val_sum = split_by_polygon_id(
                    X, y, polys, args.perc, return_sums=True
                )
                split_ratios = train_sum / (train_sum + val_sum)
                print(
                    "Training split ratio per class: [%s]"
                    % (" ".join(["{0:0.2f}".format(val) for val in split_ratios]))
                )
            else:
                ValueError("Not implemented yet")
                # train_set = tdata

                # data_val = data_valid["data"]

                # if args.hist_match:
                #     for ii in range(data_val.shape[1]):
                #         data_val[:, ii] = hist_match(
                #             data_val[:, ii], data_train["data"][:, ii]
                #         )

                # dataset_tensors_val = [
                #     torch.tensor(in_data_val),
                #     torch.tensor(data_valid["gt"]),
                # ]
                # tdata_val = TensorDataset(*dataset_tensors_val)
                # val_set = tdata_val

            options = dict()
            if args.model == "nn":
                options = {
                    "hidden_sizes": [512, 512],
                    "epochs": args.epochs,
                    "batch_size": args.batch,
                    "gpu": args.gpu,
                }
            elif args.model == "rf":
                options = {"n_estimators": 2}

            model = seabed_classifier_type[args.model](**options)
            logger.info(f"Training {seabed_classifier_name[args.model]} model")
            start = monotonic()
            model.fit(train_set[0], train_set[1])
            tdiff = monotonic() - start
            logger.info("Time elapsed: %f seconds" % tdiff)

            pred_lab = model.predict(val_set[0])               

            # Compute metrics for each run
            accuracy = (pred_lab == val_set[1]).sum().item() / pred_lab.size
            kappa_val = cohen_kappa_score(val_set[1], pred_lab)
            print("Accuracy: %.2f, kappa: %.3f" % (accuracy * 100, kappa_val))

            # Compute polygon accuracy
            val_polygons = np.unique(val_set[2])
            acc = []
            for vp in val_polygons:
                vp_inds = val_set[2] == vp
                polygon_predictions, prediction_counts = np.unique(
                    pred_lab[vp_inds], return_counts=True
                )
                polygon_predicted_class = polygon_predictions[
                    np.argmax(prediction_counts)
                ]
                if polygon_predicted_class == val_set[1][vp_inds][0]:
                    acc.append(True)
                else:
                    acc.append(False)
            print("Polygon accuracy: %d/%d" % (np.sum(acc), len(acc)))

            acc_res.append(accuracy)
            kappa_res.append(kappa_val)
            if best_result is None or best_result[-1] < accuracy:
                best_model = deepcopy(model)
                best_result = (val_set[:][1], pred_lab, kappa_val, accuracy)

            if args.track:
                mlflow.log_metric("Accuracy", accuracy, step=cnt)
                mlflow.log_metric("Kappa", accuracy, step=cnt)

        # Compute metric statistics
        if len(acc_res) > 1:
            acc_av = np.mean(acc_res)
            acc_std = np.std(acc_res)
            kappa_av = np.mean(kappa_res)
            kappa_std = np.std(kappa_res)
            print(
                "Results after %d runs: AP %f %s %f, kappa %f %s %f"
                % (args.runs, acc_av, chr(177), acc_std, kappa_av, chr(177), kappa_std)
            )
            if args.track:
                mlflow.log_metrics({"Acc. Average": acc_av, "Acc. Std": acc_std})

        cm = _get_cm(best_result[0], best_result[1], round_prec=4)
        print("Confusion Matrix:")
        print(cm)
        if args.track:
            mlflow.log_metric("Kappa", float(cm.PA[-2]))

        if args.xls_file:
            xls_path = args.output_dir + "/" + args.xls_file + ".xlsx"
            writer = pd.ExcelWriter(xls_path)
            cm.to_excel(writer, "CM")
            writer.save()
            if args.track:
                mlflow.log_artifact(xls_path)

        if args.class_map:
            pred_all = best_model.predict(data_valid["data_all"])
            data_out = np.zeros_like(data_valid["mask_data"], dtype=np.uint8)
            data_out[data_valid["mask_data"]] = pred_all + 1

            out_raster = data_valid["data_raster"].isel(band=0)
            out_raster.data = data_out.astype(np.float32)
            out_raster.rio.to_raster(
                pathlib.Path(args.output_dir) / "out.tif", dtype=np.uint8
            )
            plt.imshow(data_out)
            plt.show()
    # return


if __name__ == "__main__":
    main()
