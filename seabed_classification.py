import os
import argparse

import numpy as np

from contextlib import nullcontext

import rioxarray

import matplotlib
import matplotlib.pyplot as plt
import mlflow

from torch.utils.data import TensorDataset

from libs.mlp import *

import pandas as pd

from utils.reporting import _get_cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from time import monotonic


# matplotlib.use('TkAgg')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Seabed classification.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Dataset directory (leave empty for individual files)",
    )
    parser.add_argument(
        "--input", type=str, help="Path of input image", default="backscatter.tif"
    )
    parser.add_argument(
        "--gt", type=str, help="Path of reference data image", default="GT.tif"
    )
    parser.add_argument(
        "--bathy", type=str, help="Bathymetry layer", default="bathymetry.tif"
    )
    parser.add_argument("--runs", type=int, help="Number of runs", default=1)
    parser.add_argument(
        "--perc",
        type=float,
        help="Percentage of training in decimal form ex. 0.9",
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

    print("Reading ground truth...", end=" ")
    gt_raster = rioxarray.open_rasterio(get_file_loc(args.dataset_dir, args.gt))
    if gt_raster.data.dtype != np.uint8:
        gt_raster.data = gt_raster.data.astype(np.uint8)
    mask_gt = np.all(gt_raster.data != gt_raster.rio.nodata, axis=0)
    mask_cum = mask_gt
    num_classes = np.unique(gt_raster.data).size - 1
    print("done!")

    print("Reading data...", end=" ")
    input_raster = rioxarray.open_rasterio(get_file_loc(args.dataset_dir, args.input))
    if input_raster.data.dtype != np.uint8:
        input_raster.data = input_raster.data.astype(np.uint8)
    mask_input = np.all(input_raster.data != input_raster.rio.nodata, axis=0)
    mask_valid = mask_input
    mask_cum = np.logical_and(mask_cum, mask_input)
    print("done!")

    if args.bathy:
        print("Reading bathy...", end=" ")
        bathy_raster = rioxarray.open_rasterio(
            get_file_loc(args.dataset_dir, args.bathy)
        )
        mask_bathy = np.all(bathy_raster.data != bathy_raster.rio.nodata, axis=0)
        mask_cum = np.logical_and(mask_cum, mask_bathy)
        mask_valid = np.logical_and(mask_valid, mask_bathy)
        print("done!")

    if args.plot_aligned:
        plt.imshow(input_raster.data.transpose(1, 2, 0).astype(np.uint8))
        if args.bathy:
            bathy_show = bathy_raster.data[0, ...]
            bathy_show[~mask_bathy] = np.nan
            plt.imshow(bathy_show, alpha=0.5)
        gt_show = gt_raster.data[0, ...].astype(np.float32)
        gt_show[~mask_gt] = np.nan
        plt.imshow(gt_show, cmap="Set1")
        plt.show()

    pos_y, pos_x = np.where(mask_cum)
    gt_filt = gt_raster.data[0, mask_cum] - 1
    data_pixels = input_raster.data.reshape((3, -1))[:, mask_cum.flatten()].transpose()
    if args.bathy:
        data_pixels = np.append(
            data_pixels,
            bathy_raster.data.flatten()[mask_cum.flatten(), np.newaxis],
            axis=1,
        )
    if args.normalize:
        data_pixels_n = (data_pixels - np.mean(data_pixels, axis=0)) / np.std(
            data_pixels, axis=0
        )
    else:
        data_pixels_n = data_pixels

    d_emb = args.embedding_dim
    sigma = args.embedding_sigma

    if d_emb > 0:
        print("Using positional embedding with dim %d and sigma %.3f" % (d_emb, sigma))
        div_term = np.random.randn(2, d_emb // 2) * sigma ** 2
        p_emb_kern = np.einsum("ij,ik->jk", [pos_x, pos_y], div_term)

        in_data = np.concatenate(
            [np.sin(p_emb_kern), np.cos(p_emb_kern), data_pixels_n], 1
        )
    else:
        in_data = data_pixels_n

    tdata = TensorDataset(torch.tensor(in_data), torch.tensor(gt_filt))
    len_data = len(tdata)
    print("Number of samples: %d" % len_data)

    train_size = int(len_data * args.perc)
    val_size = len_data - train_size

    with (mlflow.start_run() if args.track else nullcontext()) as ctxt:
        if args.track:
            param_dict = {
                "Dataset": args.dataset_dir,
                "Bathymetry": args.bathy,
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
            mlflow.set_tag("PE", args.embedding_dim>0)
        acc_res = []
        for cnt, run in enumerate(range(args.runs)):
            print("Run %d/%d" % (run + 1, args.runs))
            train_set, val_set = torch.utils.data.random_split(
                tdata, (train_size, val_size)  # , torch.Generator().manual_seed(42)
            )
            # val_set = tdata

            all_data = input_raster.data.reshape((3, -1))
            all_data = all_data[:, mask_valid.flatten()].transpose()
            if args.bathy:
                all_data = np.append(
                    all_data,
                    bathy_raster.data.flatten()[mask_valid.flatten(), np.newaxis],
                    axis=1,
                )
            if args.normalize:
                all_data_n = (all_data - np.mean(data_pixels, axis=0)) / np.std(
                    data_pixels, axis=0
                )
            else:
                all_data_n = all_data

            all_dataset = TensorDataset(torch.tensor(all_data_n))

            if args.model == "nn":
                D_in = data_pixels_n.shape[1] + d_emb
                H1 = 512
                H2 = 512

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
                model = RandomForestClassifier(n_estimators=2)  # , max_depth = "auto")
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
            print("Accuracy: %.2f" % (accuracy * 100))
            acc_res.append(accuracy)
            if args.track:
                mlflow.log_metric("Accuracy", accuracy, step=cnt)

        if len(acc_res) > 1:
            acc_av = np.mean(acc_res)
            acc_std = np.std(acc_res)
            print(
                "Results after %d runs: AP %f, sigma %f" % (args.runs, acc_av, acc_std)
            )
            if args.track:
                mlflow.log_metrics({"Acc. Average": acc_av, "Acc. Std": acc_std})

        cm = _get_cm(val_set[:][1].numpy(), pred_lab, round_prec=4)
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
            data_out = np.zeros(np.prod(input_raster.data.shape[1:]))
            data_out[mask_input.flatten()] = pred_all + 1
            data_out = data_out.reshape(input_raster.data.shape[1:])

            out_raster = gt_raster.copy()
            out_raster.data = np.expand_dims(data_out.astype(np.float32), 0)
            out_raster.rio.to_raster(
                os.path.join(args.output_dir, "out.tif"), dtype=np.uint8
            )
    # return


if __name__ == "__main__":
    main()