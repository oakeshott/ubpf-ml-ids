#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ctypes as ct
import json
import os
from ctypes import CDLL, POINTER

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


FIXED_SCALE = 2**16


def load_train_test(train_csv: str, test_csv: str, target_col: str, fixed_scale: int):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_train_full = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    X_train_full = np.rint(
        np.asarray(X_train_full, dtype=np.float64) * fixed_scale
    ).astype(np.int64)
    X_test = np.rint(np.asarray(X_test, dtype=np.float64) * fixed_scale).astype(
        np.int64
    )

    y_train_series = train_df[target_col]
    y_test_series = test_df[target_col]

    y_train_full = y_train_series.values.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-point XGB C/BPF shared library on a test CSV"
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
        "--lib-path",
        type=str,
        default="./xgb.so",
        help="xgb.cから作成した共有ライブラリ (default: ./xgb.so)",
    )
    parser.add_argument(
        "--resdir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--fixed-scale",
        type=int,
        default=FIXED_SCALE,
        help="特徴量の固定小数点スケール。xgb.bpf.pyと一致させること (default: 2^16)",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading train from {args.train_csv}")
    print(f"[INFO] Loading test  from {args.test_csv}")
    X_train_full, y_train_full, X_test, y_test = load_train_test(
        args.train_csv, args.test_csv, args.target, args.fixed_scale
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

    lib = CDLL(args.lib_path)
    xgb = lib.xgb

    c_uint_p = ct.POINTER(ct.c_uint)
    xgb.argtypes = [POINTER(ct.c_longlong), c_uint_p]
    xgb.restype = None

    y_test_pred = []
    for sample in tqdm(X_test):
        sample = np.ascontiguousarray(sample, dtype=np.int64)
        class_indices_np = np.zeros(1, dtype=np.uintc)
        class_indices_np = np.ascontiguousarray(class_indices_np)
        shape = class_indices_np.shape

        class_indices = class_indices_np.ctypes.data_as(c_uint_p)
        sample_ptr = sample.ctypes.data_as(POINTER(ct.c_longlong))
        xgb(sample_ptr, class_indices)
        pred = np.ctypeslib.as_array(class_indices, shape)
        y_test_pred.append(pred[0])

    y_test_pred = np.asarray(y_test_pred, dtype=np.int64)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    print("[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    ret = classification_report(y_test, y_test_pred, digits=6, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    ret["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    os.makedirs(args.resdir, exist_ok=True)
    out_path = os.path.join(args.resdir, "classification_report.bpf.json")
    with open(out_path, "w") as f:
        json.dump(
            ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": ")
        )
    print(f"[INFO] Saved classification report to {out_path}")


if __name__ == "__main__":
    main()
