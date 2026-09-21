#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import ctypes as ct
from ctypes import CDLL, POINTER
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


FIXED_SCALE = 2**16


def load_test(test_csv: str, target_col: str, fixed_scale: int):
    test_df = pd.read_csv(test_csv)

    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_test = test_df.drop(columns=[target_col])
    X_test = (X_test * fixed_scale).round()
    y_test = test_df[target_col].values.astype(np.int64)

    return X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-point SVM shared library generated from svm.c"
    )
    parser.add_argument("--test-csv", type=str, required=True, help="テスト用CSV")
    parser.add_argument(
        "--target", type=str, required=True, help="目的変数（ラベル）の列名"
    )
    parser.add_argument(
        "--lib-path",
        type=str,
        default="./svm.so",
        help="svm.c から生成した共有ライブラリ (default: ./svm.so)",
    )
    parser.add_argument(
        "--resdir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--fixed-scale",
        type=int,
        default=FIXED_SCALE,
        help="固定小数点スケール (default: 2^16)",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading test from {args.test_csv}")
    X_test, y_test = load_test(args.test_csv, args.target, args.fixed_scale)
    print(f"[INFO] X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    lib = CDLL(args.lib_path)
    svm = lib.svm

    c_uint_p = ct.POINTER(ct.c_uint)
    svm.argtypes = [POINTER(ct.c_longlong), c_uint_p]
    svm.restype = None

    X_test_np = X_test.values.astype(np.int64)
    y_test_pred = []
    out_len = 1

    for sample in tqdm(X_test_np):
        sample = np.ascontiguousarray(sample, dtype=np.int64)
        class_indices_np = np.zeros(out_len, dtype=np.uintc)
        class_indices_np = np.ascontiguousarray(class_indices_np)

        sample_p = sample.ctypes.data_as(POINTER(ct.c_longlong))
        class_indices_p = class_indices_np.ctypes.data_as(c_uint_p)

        svm(sample_p, class_indices_p)
        y_test_pred.append(int(class_indices_np[0]))

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
