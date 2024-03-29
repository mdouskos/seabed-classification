import argparse
import pathlib

import logging

import numpy as np

from contextlib import nullcontext

import matplotlib.pyplot as plt

import mlflow

from libs.seabed_classifier import seabed_classifier_type, seabed_classifier_name

from copy import deepcopy

from utils.reporting import compute_cm, compute_metrics
from utils.dataset import (
    split_by_polygon_id,
    normalize_data,
    hist_match,
)

from time import monotonic

from libs.survey_data import SurveyData

np.random.seed(1234)

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
        "--taxonomy",
        type=str,
        help="Specify taxonomy type",
        default="folk 5",
        choices=("folk 5", "folk 7"),
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
        choices=("none", "minmax", "std", "range"),
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
    parser.add_argument(
        "--nn-size", type=int, help="Size of MLP hidden layers", default=512
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--class-map", action="store_true", help="Compute classification map"
    )
    parser.add_argument("--plot", action="store_true", help="Plot data layers")
    parser.add_argument(
        "--bpi-radius",
        type=int,
        help="Radius around samples (in pixels) for computing BPI",
        default=17,
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        help="Radius around samples (in pixels) defining sampling area",
        default=15,
    )
    parser.add_argument("--output-dir", type=str, help="Output directory", default="./")
    parser.add_argument(
        "--xls-file", type=str, help="XLS filename for saving confusion matrix"
    )
    parser.add_argument(
        "--track", action="store_true", help="Track experiments with mlflow"
    )
    return parser.parse_args()


cls_name_maps = {
    "folk 7": np.array(
        ["Mud", "sandy Mud", "Sand", "Mixed sediment", "Coarse sediment"]
    ),
    "folk 5": np.array(["Mud", "Sand", "Mixed sediment", "Coarse sediment"]),
}


def build_dataset_dict(
    dataset_dir,
    regions,
    mode,
    plot=False,
    taxonomy="folk 7",
    samples_key="samples",
    **options,
):
    X = None
    y = None
    polys = None
    data_all = []
    data_raster = []
    data_raster_mask = []
    data_names = []
    polys_offset = 0
    for i, region in enumerate(regions):
        dataset_path = dataset_dir / region
        region = SurveyData(
            dataset_path,
            dataset_dir=dataset_dir,
            mode=mode,
            taxonomy=taxonomy,
            samples_key=samples_key,
        )
        if plot:
            region.plot()

        data = region.get_data(**options)

        if i == 0:
            X = data["data"]
            y = data["gt"]
            polys = data["polygons"]
        else:
            X = np.vstack((X, data["data"]))
            y = np.hstack((y, data["gt"]))
            polys = np.hstack((polys, data["polygons"] + polys_offset))
        data_all.append(data["data_all"])
        data_raster.append(data["backscatter"])
        data_raster_mask.append(data["backscatter_mask"])
        data_names.append(data["name"])
        polys_offset += np.max(polys)

    data_dict = {
        "X": X,
        "y": y,
        "polys": polys,
        "data_all": data_all,
        "raster": data_raster,
        "raster_mask": data_raster_mask,
        "names": data_names,
    }

    return data_dict


def compute_classification_map(
    model, data, raster, raster_mask, out_name="out.tif", id_map=None, plot=False
):
    pred_all = model.predict(data)
    data_out = np.zeros_like(raster_mask, dtype=np.uint8)
    if id_map is not None:
        pred_all = id_map[pred_all]
    data_out[raster_mask] = pred_all + 1

    out_raster = raster.isel(band=0)
    out_raster.data = data_out.astype(np.float32)
    out_raster.rio.to_raster(out_name, dtype=np.uint8)
    if plot:
        plt.imshow(data_out)
        plt.show()


def main():
    args = parse_arguments()

    data_proc_options = {
        "pe_dim": args.embedding_dim,
        "pe_sigma": args.embedding_sigma,
        "dilation_radius": args.dilation_radius,
        "bpi_radius": args.bpi_radius,
        "morph_element": "square",
    }
    dataset_dir = pathlib.Path(args.dataset_dir)
    data_train_dict = build_dataset_dict(
        dataset_dir,
        args.inputs,
        mode=args.mode,
        taxonomy=args.taxonomy,
        plot=args.plot,
        # samples_key="samples_train",
        **data_proc_options,
    )
    if args.validation is None:
        data_val_dict = {}
    else:
        data_val_dict = build_dataset_dict(
            dataset_dir,
            args.validation,
            mode=args.mode,
            taxonomy=args.taxonomy,
            # samples_key="samples_valid",
            **data_proc_options,
        )

    X = data_train_dict["X"]
    y = data_train_dict["y"]
    polys = data_train_dict["polys"]
    logger.info(f"Number of samples: {len(y)}")
    with (mlflow.start_run() if args.track else nullcontext()):
        if args.track:
            param_dict = {
                "Model": args.model,
                "Normalization": args.normalize,
                "Mode": args.mode,
                "Split": args.perc,
                "PEdim": args.embedding_dim,
                "PESigma": args.embedding_sigma,
                "Dilation radius": args.dilation_radius,
                "BPI radius": args.bpi_radius,
            }
            if args.model == "nn":
                param_dict.update({"Epochs": args.epochs, "Batch": args.batch})
            mlflow.log_params(param_dict)
            mlflow.set_tag("Inputs", args.inputs)
            mlflow.set_tag("Validation", args.validation)
            mlflow.set_tag("Taxonomy", args.taxonomy)
            mlflow.set_tag("Runs", args.runs)
            mlflow.set_tag("PE", args.embedding_dim > 0)

        acc_res = []
        kappa_res = []
        f1_res = []
        best_result = None

        orig_ids = np.unique(y)

        n_runs = args.runs
        if args.validation is not None:
            n_runs = 1
            train_set = [data_train_dict["X"], data_train_dict["y"], None]
            val_set = [data_val_dict["X"], data_val_dict["y"], None]
            if args.hist_match:
                val_set[0] = hist_match(val_set[0], X)

        for run in range(1, n_runs + 1):
            logger.info(f"Run {run}/{n_runs}")
            if args.validation is None:
                data_val_dict = data_train_dict
                train_set, val_set, train_sum, val_sum = split_by_polygon_id(
                    X, y, polys, args.perc, return_sums=True, shuffle=True
                )
                split_ratios = train_sum / (train_sum + val_sum)
                logger.info(
                    "Training split ratio per class: [%s]"
                    % (" ".join(["{0:0.2f}".format(val) for val in split_ratios]))
                )

            X_train = train_set[0]
            X_ref = X_train.copy()
            _, y_train = np.unique(train_set[1], return_inverse=True)
            X_val = val_set[0]
            _, y_val = np.unique(val_set[1], return_inverse=True)
            poly_val = val_set[2]
            if args.normalize != "none":
                X_val = normalize_data(
                    X_val,
                    X_ref=X_ref,
                    normalization_type=args.normalize,
                )
                X_train = normalize_data(X_train, normalization_type=args.normalize)

            model_options = dict()
            if args.model == "nn":
                model_options = {
                    "hidden_sizes": [args.nn_size, args.nn_size],
                    "epochs": args.epochs,
                    "batch_size": args.batch,
                    "gpu": args.gpu,
                }
            elif args.model == "rf":
                model_options = {"n_estimators": 2, "random_state": 0}

            model = seabed_classifier_type[args.model](**model_options)
            logger.info(f"Training {seabed_classifier_name[args.model]} model")
            start = monotonic()
            model.fit(X_train, y_train)
            tdiff = monotonic() - start
            logger.info(f"Time elapsed: {tdiff:.5f} seconds")

            pred_labels = model.predict(X_val)

            metrics = compute_metrics(y_val, pred_labels, polygons=poly_val)

            # if args.track:
            #     mlflow.log_metric("Accuracy", metrics["accuracy"], step=run)
            #     mlflow.log_metric("Kappa", metrics["kappa"], step=run)
            #     mlflow.log_metric("F1-score", metrics["f1-score"], step=run)

            logger.info(
                f"Accuracy: {metrics['accuracy']:.3f}"
                + f", kappa: {metrics['kappa']:.3f}"
                + f", f1-score {metrics['f1-score']:.3f}"
            )
            if metrics["polygon"]:
                logger.info(
                    f"Polygon accuracy: {np.sum(metrics['polygon'])}/{len(metrics['polygon'])}"
                )

            acc_res.append(metrics["accuracy"])
            kappa_res.append(metrics["kappa"])
            f1_res.append(metrics["f1-score"])

            if (
                best_result is None
                or best_result["metrics"]["accuracy"] < metrics["accuracy"]
            ):
                best_result = {
                    "model": deepcopy(model),
                    "y": y_val,
                    "predicted": pred_labels,
                    "X_ref": X_ref,
                    "metrics": metrics,
                }

        # Compute metric statistics
        if len(acc_res) > 1:
            acc_av = np.mean(acc_res)
            acc_std = np.std(acc_res)
            kappa_av = np.mean(kappa_res)
            kappa_std = np.std(kappa_res)
            f1_av = np.mean(f1_res)
            f1_std = np.std(f1_res)
            plusminus_symbol = chr(177)
            logger.info(
                f"Results after {n_runs} runs: AP {acc_av:.3f} {plusminus_symbol} {acc_std:.3f}"
                + f", kappa {kappa_av:.3f} {plusminus_symbol} {kappa_std:.3f}"
                + f", f1-score {f1_av:.3f} {plusminus_symbol} {f1_std:.3f}"
            )
        else:
            acc_av = best_result["metrics"]["accuracy"]
            kappa_av = best_result["metrics"]["kappa"]
            f1_av = best_result["metrics"]["f1-score"]
            acc_std, kappa_std, f1_std = (0, 0, 0)
        if args.track:
            mlflow.log_metrics(
                {"Acc. Average": acc_av, "F1 Average": f1_av, "K Average": kappa_av}
            )
            mlflow.log_metrics(
                {"Acc. Std": acc_std, "F1 Std": f1_std, "K Std": kappa_std}
            )

        # Compute confusion matrix
        xls_path = None
        if args.xls_file:
            xls_path = pathlib.Path(args.output_dir) / (args.xls_file + ".xlsx")
        cm = compute_cm(
            best_result["y"],
            best_result["predicted"],
            round_prec=4,
            cls_names=cls_name_maps[args.taxonomy][orig_ids],
            xls_path=xls_path,
        )
        logger.info(f"Confusion Matrix:\n{cm}")
        if args.track and args.xls_file:
            mlflow.log_artifact(xls_path)

    if args.class_map:
        data = data_val_dict["data_all"]
        raster = data_val_dict["raster"]
        raster_mask = data_val_dict["raster_mask"]
        names = data_val_dict["names"]
        output_dir = pathlib.Path(args.output_dir)
        for i in range(len(data)):
            out_name = output_dir / f"cmap_{names[i]}.tif"
            if args.normalize != "none":
                data_all = normalize_data(
                    data[i],
                    X_ref=best_result["X_ref"],
                    normalization_type=args.normalize,
                )
            else:
                data_all = data[i]
            compute_classification_map(
                best_result["model"],
                data_all,
                raster[i],
                raster_mask[i],
                out_name,
                id_map=orig_ids,
                plot=args.plot,
            )


if __name__ == "__main__":
    main()
