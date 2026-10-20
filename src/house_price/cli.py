"""Command-line interface: ``house-price {generate-data,train,predict}``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__, config
from .data import generate_synthetic_data, save_data
from .predict import HouseFeatures, predict_price
from .train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="house-price", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="create the synthetic dataset")
    gen.add_argument("--n-samples", type=int, default=1200)
    gen.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    gen.add_argument("--output", default=str(config.DATA_PATH))

    tr = sub.add_parser("train", help="train, evaluate and save the best model")
    tr.add_argument("--data", default=str(config.DATA_PATH))
    tr.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    tr.add_argument("--tune", action="store_true", help="randomized hyper-parameter search")
    tr.add_argument("--no-plots", action="store_true")

    pr = sub.add_parser("predict", help="predict the price of one house")
    pr.add_argument("--area", type=float, required=True, help="area in sq ft")
    pr.add_argument("--bedrooms", type=int, required=True)
    pr.add_argument("--bathrooms", type=int, required=True)
    pr.add_argument("--age", type=float, required=True, help="age in years")
    pr.add_argument("--distance", type=float, required=True, help="km to city centre")
    pr.add_argument("--location", choices=config.LOCATIONS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.command == "generate-data":
            save_data(generate_synthetic_data(args.n_samples, args.seed), args.output)
        elif args.command == "train":
            train(data_path=args.data, seed=args.seed, tune=args.tune, make_plots=not args.no_plots)
        elif args.command == "predict":
            features = HouseFeatures(
                area_sqft=args.area, bedrooms=args.bedrooms, bathrooms=args.bathrooms,
                age_years=args.age, distance_to_city_km=args.distance, location=args.location,
            )
            print(f"Predicted price: {config.CURRENCY} {predict_price(features):,.0f}")
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
