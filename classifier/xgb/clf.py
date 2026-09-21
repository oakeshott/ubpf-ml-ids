#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


def load_train_test(train_csv: str, test_csv: str, target_col: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_train_full = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    y_train_series = train_df[target_col]
    y_test_series = test_df[target_col]

    y_train_full = y_train_series.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="XGBClassifier (train/test from different CSVs)"
    )
    parser.add_argument(
        "--train-csv", type=str, required=True, help="学習用CSV（train+val）"
    )
    parser.add_argument("--test-csv", type=str, required=True, help="テスト用CSV")
    parser.add_argument(
        "--target", type=str, required=True, help="目的変数（ラベル）の列名"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="学習CSVのうちValidationに回す割合"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="乱数シード (default: 42)"
    )
    parser.add_argument(
        "--resdir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="xgb.joblib",
        help="モデルファイル (default: xgb.joblib)",
    )
    parser.add_argument(
        "--scaler_path", type=str, default="scaler.joblib", help="学習率"
    )
    args = parser.parse_args()

    print(f"[INFO] Loading train from {args.train_csv}")
    print(f"[INFO] Loading test  from {args.test_csv}")
    X_train_full, y_train_full, X_test, y_test = load_train_test(
        args.train_csv, args.test_csv, args.target
    )
    print(
        f"[INFO] X_train_full shape = {X_train_full.shape}, y_train_full shape = {y_train_full.shape}"
    )
    print(
        f"[INFO] X_test shape       = {X_test.shape}, y_test shape       = {y_test.shape}"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.val_ratio,
        random_state=args.random_state,
        stratify=y_train_full,
    )
    print(
        f"[INFO] Train size = {X_train.shape[0]}, Val size = {X_val.shape[0]}, Test size = {X_test.shape[0]}"
    )
    # standard_scaler = joblib.load(args.scaler_path)
    # X_test_norm = standard_scaler.transform(X_test)

    clf = joblib.load(args.model_path)

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    print("[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    ret = classification_report(y_test, y_test_pred, digits=6, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    ret["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    os.makedirs(args.resdir, exist_ok=True)
    with open(f"{args.resdir}/classification_report.json", "w") as f:
        json.dump(
            ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": ")
        )


if __name__ == "__main__":
    main()
