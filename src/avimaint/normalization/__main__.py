"""Command-line entry point for the normalization research workflow."""

from __future__ import annotations

import argparse

from avimaint.configuration import load_yaml
def main() -> None:
    parser = argparse.ArgumentParser(description="AviMaint normalization experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare", "split", "train", "make-silver"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    for command in ("predict", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument(
            "--split", choices=["train", "validation", "test", "sensitivity"], required=True
        )
        child.add_argument(
            "--system",
            choices=[
                "raw",
                "most_frequent_replacement",
                "rules",
                "byt5",
                "selective_byt5",
                "rules_then_byt5",
            ],
            default="selective_byt5",
        )
    corpus = subparsers.add_parser("predict-corpus")
    corpus.add_argument("--config", required=True)
    corpus.add_argument(
        "--system",
        choices=["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"],
        required=True,
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.command == "audit":
        from avimaint.normalization.audit import run_audit
        output = run_audit(config)
    elif args.command == "prepare":
        from avimaint.normalization.audit import prepare_approved_pairs
        output = prepare_approved_pairs(config)
    elif args.command == "split":
        from avimaint.normalization.splitting import run_split
        output = run_split(config)
    elif args.command == "train":
        from avimaint.normalization.training import train_from_config
        output = train_from_config(config)
    elif args.command == "make-silver":
        from avimaint.normalization.silver import make_silver
        output = make_silver(config)
    elif args.command == "predict":
        from avimaint.normalization.prediction import predict
        output = predict(config, args.split, args.system)
    elif args.command == "predict-corpus":
        from avimaint.normalization.prediction import predict_corpus
        output = predict_corpus(config, args.system)
    else:
        from avimaint.normalization.evaluation import evaluate
        output = evaluate(config, args.split, args.system)
    print(output)


if __name__ == "__main__":
    main()
