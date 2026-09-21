#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


FIXED_SCALE = 2**16


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

    y_train_full = y_train_series.values.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def write_int64_array(f, name: str, values):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    f.write(f"const int64_t {name}[{name.upper()}_SIZE] = {{")
    for k, val in enumerate(values):
        sep = "" if k == len(values) - 1 else ", "
        f.write(f"{int(val)}{sep}")
    f.write("};\n")


def write_uint_array(f, name: str, values):
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    f.write(f"const unsigned int {name}[{name.upper()}_SIZE] = {{")
    for k, val in enumerate(values):
        sep = "" if k == len(values) - 1 else ", "
        f.write(f"{int(val)}{sep}")
    f.write("};\n")


def export_svm_params(
    clf: LinearSVC,
    scaler: preprocessing.StandardScaler,
    num_features: int,
    model_path: str,
    fixed_scale: int,
):
    """
    Export LinearSVC parameters for svm.c.

    svm.c receives fixed-point input:
        x_int[j] = round(x_float[j] * fixed_scale)

    LinearSVC was trained on standardized features:
        z[j] = (x_float[j] - mean[j]) / scale[j]

    The scaler is folded into the linear model:
        score = sum_j coef[j] * z[j] + intercept
              = sum_j (coef[j] / scale[j]) * x_float[j]
                + intercept - sum_j coef[j] * mean[j] / scale[j]

    For integer inference in svm.c:
        score_int = intercept_int + sum_j coef_int[j] * x_int[j] / fixed_scale
    where:
        coef_int[j] = round((coef[j] / scale[j]) * fixed_scale)
        intercept_int = round(intercept_eff * fixed_scale)
    """
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)
    classes = np.asarray(clf.classes_, dtype=np.uint32)

    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)

    coef_eff = coef / scale.reshape(1, -1)
    intercept_eff = intercept - np.sum(
        coef * mean.reshape(1, -1) / scale.reshape(1, -1), axis=1
    )

    coef_int = np.rint(coef_eff * fixed_scale).astype(np.int64)
    intercept_int = np.rint(intercept_eff * fixed_scale).astype(np.int64)

    output_dim = len(classes)
    num_classifiers = coef_int.shape[0]

    with open(model_path, "w") as f:
        f.write("#ifndef SVM_PARAMS_H\n")
        f.write("#define SVM_PARAMS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_FEATURES {num_features}\n")
        f.write(f"#define OUTPUT_DIM {output_dim}\n")
        f.write(f"#define SVM_NUM_CLASSIFIERS {num_classifiers}\n")
        f.write(f"#define SVM_SCALE {fixed_scale}\n")
        f.write(f"#define SVM_COEF_SIZE {coef_int.size}\n")
        f.write(f"#define SVM_INTERCEPT_SIZE {intercept_int.size}\n")
        f.write(f"#define SVM_CLASSES_SIZE {classes.size}\n\n")
        write_int64_array(f, "svm_coef", coef_int)
        write_int64_array(f, "svm_intercept", intercept_int)
        write_uint_array(f, "svm_classes", classes)
        f.write("\n#endif /* SVM_PARAMS_H */\n")


def main():
    parser = argparse.ArgumentParser(
        description="LinearSVC parameter exporter for fixed-point BPF/C inference"
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
        default="svm_params.h",
        help="SVMパラメータヘッダの保存先 (default: svm_params.h)",
    )
    parser.add_argument(
        "--fixed-scale",
        type=int,
        default=FIXED_SCALE,
        help="固定小数点スケール (default: 2^16)",
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

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    clf = LinearSVC(
        C=args.c,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    print("[INFO] Training LinearSVC...")
    clf.fit(X_train_norm, y_train)
    print("[INFO] Training done.")

    y_val_pred = clf.predict(X_val_norm)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n[RESULT] Validation Accuracy: {val_acc:.4f}\n")
    print(classification_report(y_val, y_val_pred))

    # y_test_pred = clf.predict(X_test_norm)
    # test_acc = accuracy_score(y_test, y_test_pred)
    # print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    # print("[RESULT] Classification Report (Test):")
    # print(classification_report(y_test, y_test_pred))
    #
    export_svm_params(
        clf=clf,
        scaler=scaler,
        num_features=X_test.shape[1],
        model_path=args.model_path,
        fixed_scale=args.fixed_scale,
    )
    print(f"[INFO] Exported SVM parameters to {args.model_path}")


if __name__ == "__main__":
    main()
