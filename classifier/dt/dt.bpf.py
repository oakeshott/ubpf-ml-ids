#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_train_test(train_csv: str, test_csv: str, target_col: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_train_full = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    X_train_full = (X_train_full * 2**16).round()
    X_test = (X_test * 2**16).round()

    y_train_series = train_df[target_col]
    y_test_series = test_df[target_col]

    y_train_full = y_train_series.values.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="DecisionTreeClassifier (train/test from different CSVs)"
    )
    parser.add_argument("--train-csv", type=str, required=True, help="学習用CSV（train+val）")
    parser.add_argument("--test-csv", type=str, required=True, help="テスト用CSV")
    parser.add_argument("--target", type=str, required=True, help="目的変数（ラベル）の列名")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="学習CSVのうちValidationに回す割合")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="木の最大深さ (default: None)")
    parser.add_argument("--criterion", type=str, default="gini",
                        choices=["gini", "entropy", "log_loss"],
                        help="分割規準 (default: gini)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="乱数シード (default: 42)")
    args = parser.parse_args()

    print(f"[INFO] Loading train from {args.train_csv}")
    print(f"[INFO] Loading test  from {args.test_csv}")
    X_train_full, y_train_full, X_test, y_test = load_train_test(
        args.train_csv, args.test_csv, args.target
    )
    print(f"[INFO] X_train_full shape = {X_train_full.shape}, y_train_full shape = {y_train_full.shape}")
    print(f"[INFO] X_test shape       = {X_test.shape}, y_test shape       = {y_test.shape}")

    # train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_ratio,
        random_state=args.random_state,
        stratify=y_train_full
    )
    print(f"[INFO] Train size = {X_train.shape[0]}, Val size = {X_val.shape[0]}, Test size = {X_test.shape[0]}")

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        criterion=args.criterion,
        random_state=args.random_state,
    )

    # === Train ===
    X_train = X_train.values.astype(np.int64)
    print("[INFO] Training DecisionTree...")
    clf.fit(X_train, y_train)
    print("[INFO] Training done.")

    # === Validation ===
    X_val = X_val.values.astype(np.int64)
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n[RESULT] Validation Accuracy: {val_acc:.4f}\n")


    # === Test ===
    X_test = X_test.values.astype(np.int64)
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    print("[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    value = clf.tree_.value.squeeze().argmax(axis=1)
    features = clf.tree_.feature
    threshold = clf.tree_.threshold.round().astype(np.int64)

    with open("dt_params.h", "w") as f:
        f.write(f"#define CHILDLEN_LEFT_SIZE {len(children_left)}\n");
        f.write(f"#define CHILDLEN_RIGHT_SIZE {len(children_right)}\n");
        f.write(f"#define FEATURES_SIZE {len(features)}\n");
        f.write(f"#define THRESHOLD_SIZE {len(threshold)}\n");
        f.write(f"#define VALUE_SIZE {len(value)}\n");
        f.write('const int64_t children_left[CHILDLEN_LEFT_SIZE] = {')
        for k, val in enumerate(children_left):
            if k == len(children_left) - 1:
                f.write(f'{val}')
            else:
                f.write(f'{val}, ')
        f.write('};\n')
        f.write('const int64_t children_right[CHILDLEN_RIGHT_SIZE] = {')
        for k, val in enumerate(children_right):
            if k == len(children_right) - 1:
                f.write(f'{val}')
            else:
                f.write(f'{val}, ')
        f.write('};\n')
        f.write('const int64_t features[FEATURES_SIZE] = {')
        for k, val in enumerate(features):
            if k == len(features) - 1:
                f.write(f'{val}')
            else:
                f.write(f'{val}, ')
        f.write('};\n')
        f.write('const int64_t threshold[THRESHOLD_SIZE] = {')
        for k, val in enumerate(threshold):
            if k == len(threshold) - 1:
                f.write(f'{val}')
            else:
                f.write(f'{val}, ')
        f.write('};\n')
        f.write('const int64_t value[VALUE_SIZE] = {')
        for k, val in enumerate(value):
            if k == len(value) - 1:
                f.write(f'{val}')
            else:
                f.write(f'{val}, ')
        f.write('};\n')


if __name__ == "__main__":
    main()
