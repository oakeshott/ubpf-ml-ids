#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import torch


def load_train_test(train_csv: str, test_csv: str, target_col: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_train_full = train_df.drop(columns=[target_col]).values.astype(np.float32)
    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)

    y_train_series = train_df[target_col]
    y_test_series = test_df[target_col]

    y_train_full = y_train_series.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="LinearSVC (train/test from different CSVs)"
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
        "--c", type=float, default=1.0, help="正則化パラメータ C (default: 1.0)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=10000, help="最大反復回数 (default: 10000)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="乱数シード (default: 42)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="svm.joblib",
        help="モデル保存先 (default: svm.joblib)",
    )
    parser.add_argument(
        "--scaler-path", type=str, default="scaler.joblib", help="学習率"
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

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(X_train)
    print(standard_scaler.mean_, standard_scaler.scale_)
    X_train_full_norm = standard_scaler.transform(X_train_full)
    X_train_norm = standard_scaler.transform(X_train)
    X_val_norm = standard_scaler.transform(X_val)
    X_test_norm = standard_scaler.transform(X_test)

    joblib.dump(standard_scaler, args.scaler_path)

    # X_train_full_norm = (X_train_full_norm * (2 ** 16)).round()
    # X_train_norm = (X_train_norm * (2 ** 16)).round()
    # X_val_norm   = (X_val_norm * (2 ** 16)).round()
    # X_test_norm  = (X_test_norm * (2 ** 16)).round()

    # numpy -> tensor
    # X_train_t = torch.from_numpy(X_train_norm).float()
    # y_train_t = torch.from_numpy(y_train).long()
    # X_val_t = torch.from_numpy(X_val_norm).float()
    # y_val_t = torch.from_numpy(y_val).long()
    # X_test_t = torch.from_numpy(X_test_norm).float()
    # y_test_t = torch.from_numpy(y_test).long()

    clf = LinearSVC(
        C=args.c,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    # #
    # params1 = {
    #     "Epsilon1": 0.1,
    #     "Epsilon2": 0.1,
    #     "C1": 1,
    #     "C2": 1,
    #     "kernel_type": 0,
    #     "kernel_param": 1,
    #     "fuzzy": 0,
    # }
    # clf = TwinSVMClassifier(**params1)

    print("[INFO] Training LinearSVC...")
    clf.fit(X_train_norm, y_train)
    print("[INFO] Training done.")

    y_val_pred = clf.predict(X_val_norm)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n[RESULT] Validation Accuracy: {val_acc:.4f}\n")

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(y_test, y_test_pred))
    print("[RESULT] Classification Report (Test):")

    joblib.dump(clf, args.model_path)


if __name__ == "__main__":
    main()
