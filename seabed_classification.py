import os
import argparse
import pathlib

import numpy as np

from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import torch

from torch.utils.data import TensorDataset

from libs.mlp import create_model, model_eval, train_network

import pandas as pd

from utils.reporting import _get_cm
from utils.dataset import (
    process_data,
    positional_encoding,
    split_by_polygon_id,
    hist_match,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import cohen_kappa_score

from time import monotonic

import json

from libs.survey_data import SurveyData

# np.random.seed(1234)


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
        default="nn",
        choices=("nn", "rf", "svm"),
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize data")
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
    parser.add_argument("--use-polygons", action="store_true", help="Use polygon IDs")
    parser.add_argument(
        "--use-bathymetry", action="store_true", help="Use bathymetry as feature"
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


def get_file_loc(dirn, filen):
    outpath = ""
    if dirn:
        outpath = os.path.join(dirn, filen)
    else:
        outpath = filen
    return outpath


def main():

    args = parse_arguments()

    # polygons_file = None
    # bathy_file = None
    # dataset_path = pathlib.Path(args.dataset_dir)
    # for input in args.inputs:
    #     input_json = dataset_path / input
    #     with open(input_json) as f:
    #         input_data = json.load(f)

    #     data_train = process_data(
    #         dataset_path / input_data["directory"] / input_data["backscatter"],
    #         dataset_path / input_data["directory"] / input_data["groundtruth"],
    #         dataset_path / input_data["directory"] / input_data["polygons"],
    #         bathy_file,
    #         normalize=args.normalize,
    #         plot=args.plot_aligned,
    #     )

    args = parse_arguments()
    dataset_path = pathlib.Path(args.dataset_dir) / args.inputs[0]
    region = SurveyData(dataset_path, dataset_dir=args.dataset_dir, mode='bs')

    data_train = region.get_data()
    d_emb = args.embedding_dim
    sigma = args.embedding_sigma

    if d_emb > 0:
        print("Using positional embedding with dim %d and sigma %.3g" % (d_emb, sigma))
        pos_data = np.where(data_train["mask_gt"])
        # sigma_vec = sigma/np.array(raster_shape)
        # pos_vec = pos_data/np.expand_dims(raster_shape, 1)
        in_data, div_term = positional_encoding(
            data_train["data"], pos_data, d_emb, sigma
        )
    else:
        in_data = data_train["data"]

    dataset_tensors = [torch.tensor(in_data), torch.tensor(data_train["gt"])]
    if args.use_polygons:
        dataset_tensors += [torch.tensor(data_train["polygons"])]
    tdata = TensorDataset(*dataset_tensors)
    len_data = len(tdata)
    print("Number of samples: %d" % len_data)

    train_size = int(len_data * args.perc)
    val_size = len_data - train_size

    with (mlflow.start_run() if args.track else nullcontext()):
        if args.track:
            param_dict = {
                "Dataset": args.dataset_dir,
                "Bathymetry": args.use_bathymetry,
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
            data_valid = process_data(
                args.input_val,
                args.gt_val,
                bathymetry_file=args.bathy_val,
                normalize=args.normalize,
            )

        for cnt, run in enumerate(range(args.runs)):
            print("Run %d/%d" % (run + 1, args.runs))
            if args.use_polygons:
                train_set, val_set, train_sum, val_sum = split_by_polygon_id(
                    tdata, args.perc, return_sums=True
                )
                split_ratios = train_sum / (train_sum + val_sum)
                print(
                    "Training split ratio per class: [%s]"
                    % (" ".join(["{0:0.2f}".format(val) for val in split_ratios]))
                )
            elif not args.input_val:
                train_set, val_set = torch.utils.data.random_split(
                    tdata, (train_size, val_size), torch.Generator().manual_seed(42)
                )
                # val_set = tdata
            else:
                train_set = tdata

                data_val = data_valid["data"]

                if args.hist_match:
                    for ii in range(data_val.shape[1]):
                        data_val[:, ii] = hist_match(
                            data_val[:, ii], data_train["data"][:, ii]
                        )

                if d_emb > 0:
                    print(
                        "Using positional embedding with dim %d and sigma %.3f"
                        % (d_emb, sigma)
                    )
                    pos_data_val = np.where(data_valid["mask_gt"])
                    # sigma_vec = sigma/np.array(raster_shape)
                    # pos_vec = pos_data/np.expand_dims(raster_shape, 1)
                    in_data_val, _ = positional_encoding(
                        data_val,
                        pos_data_val,
                        d_emb,
                        sigma,
                        div_term=div_term,
                    )
                else:
                    in_data_val = data_val

                dataset_tensors_val = [
                    torch.tensor(in_data_val),
                    torch.tensor(data_valid["gt"]),
                ]
                tdata_val = TensorDataset(*dataset_tensors_val)
                val_set = tdata_val

            if args.class_map:

                all_data = data_valid["data_raster"].data.reshape((3, -1))
                all_data = all_data[:, data_valid["mask_data"].flatten()].transpose()
                if args.use_bathymetry:
                    all_data = np.append(
                        all_data,
                        data_valid["bathy_raster"].data.flatten()[
                            data_valid["mask_data"].flatten(), np.newaxis
                        ],
                        axis=1,
                    )
                if args.normalize:
                    all_data_n = (all_data - np.mean(all_data, axis=0)) / np.std(
                        all_data, axis=0
                    )
                else:
                    all_data_n = all_data

                if d_emb > 0:
                    # pos_x, pos_y = np.meshgrid(
                    #     np.arange(raster_shape[1]), np.arange(raster_shape[0])
                    # )
                    pos_all = np.where(data_valid["mask_data"])
                    all_data_n, _ = positional_encoding(
                        all_data_n, pos_all, d_emb, sigma, div_term=div_term
                    )

                all_dataset = TensorDataset(torch.tensor(all_data_n))

            if args.model == "nn":
                D_in = in_data.shape[1]  # + d_emb
                H1 = 512
                H2 = 512

                num_classes = np.unique(data_train["gt"]).size
                model = create_model(D_in, H1, H2, num_classes, args.gpu)

                print("Training Neural Network for %d epochs" % args.epochs)
                start = monotonic()
                model, pred_lab = train_network(
                    model, train_set, val_set, args.epochs, args.batch, args.gpu
                )
                tdiff = monotonic() - start
                print("Time elapsed: %f seconds" % tdiff)

                if args.class_map:
                    print("Running over all data...", end=" ")
                    data_loader = torch.utils.data.DataLoader(
                        all_dataset, batch_size=args.batch
                    )
                    pred_all, _ = model_eval(
                        model, data_loader, gt_available=False, gpu=args.gpu
                    )
                    print("done!")

            elif args.model == "rf":
                print("Training RF model")
                model = RandomForestClassifier(
                    n_estimators=2,
                    # random_state=0,
                    # max_depth = "auto",
                )
                start = monotonic()
                model.fit(train_set[:][0].numpy(), train_set[:][1].numpy())
                pred_lab = model.predict(val_set[:][0].numpy())
                tdiff = monotonic() - start
                print("Time elapsed: %f seconds" % tdiff)
                if args.class_map:
                    pred_all = model.predict(all_data_n)
            elif args.model == "svm":
                print("Training LinearSVM model")
                model = LinearSVC()
                start = monotonic()
                model.fit(train_set[:][0].numpy(), train_set[:][1].numpy())
                pred_lab = model.predict(val_set[:][0].numpy())
                tdiff = monotonic() - start
                print("Time elapsed: %f seconds" % tdiff)
                if args.class_map:
                    pred_all = model.predict(all_data_n)
            else:
                raise ValueError("Unknown model type %s" % args.model)

            accuracy = (pred_lab == val_set[:][1].numpy()).sum().item() / pred_lab.size
            kappa_val = cohen_kappa_score(val_set[:][1], pred_lab)
            print("Accuracy: %.2f, kappa: %.3f" % (accuracy * 100, kappa_val))
            if args.use_polygons:
                val_polygons = np.unique(val_set[:][2])
                acc = []
                for vp in val_polygons:
                    vp_inds = val_set[:][2] == vp
                    polygon_predictions, prediction_counts = np.unique(
                        pred_lab[vp_inds], return_counts=True
                    )
                    polygon_predicted_class = polygon_predictions[
                        np.argmax(prediction_counts)
                    ]
                    if polygon_predicted_class == val_set[vp_inds][1][0]:
                        acc.append(True)
                    else:
                        acc.append(False)
                print("Polygon accuracy: %d/%d" % (np.sum(acc), len(acc)))
            acc_res.append(accuracy)
            kappa_res.append(kappa_val)
            if best_result is None or best_result[-1] < accuracy:
                best_result = (val_set[:][1], pred_lab, pred_all, kappa_val, accuracy)

            if args.track:
                mlflow.log_metric("Accuracy", accuracy, step=cnt)
                mlflow.log_metric("Kappa", accuracy, step=cnt)

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
            data_out = np.zeros(np.prod(data_valid["input_shape"]))
            data_out[data_valid["mask_data"].flatten()] = best_result[2] + 1
            data_out = data_out.reshape(data_valid["input_shape"])

            out_raster = data_valid["gt_raster"].copy()
            out_raster.data = data_out.astype(np.float32)
            out_raster.rio.to_raster(
                os.path.join(args.output_dir, "out.tif"), dtype=np.uint8
            )
            plt.imshow(data_out)
            plt.show()
    # return


if __name__ == "__main__":
    main()
