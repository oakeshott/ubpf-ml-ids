#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
        "--n-estimators", type=int, default=10, help="木の本数 (default: 100)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=10, help="木の最大深さ (default: 6)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="学習率 (default: 0.1)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="乱数シード (default: 42)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="xgb.joblib",
        help="モデル保存先 (default: xgb.joblib)",
    )
    parser.add_argument(
        "--scaler_path", type=str, default="scaler.joblib", help="学習率"
    )
    args = parser.parse_args()

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "xgboost がインストールされていません。pip install xgboost を実行してください。"
        ) from exc

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
    # standard_scaler = preprocessing.StandardScaler()
    # standard_scaler.fit(X_train)
    # print(standard_scaler.mean_, standard_scaler.scale_)
    # X_train_full_norm = standard_scaler.transform(X_train_full)
    # X_train_norm = standard_scaler.transform(X_train)
    # X_val_norm = standard_scaler.transform(X_val)
    # X_test_norm = standard_scaler.transform(X_test)

    # joblib.dump(standard_scaler, args.scaler_path)

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=-1,
    )

    print("[INFO] Training XGBClassifier...")
    clf.fit(X_train, y_train)
    print("[INFO] Training done.")

    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n[RESULT] Validation Accuracy: {val_acc:.4f}\n")

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    print("[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    joblib.dump(clf, args.model_path)


if __name__ == "__main__":
    main()
