#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


FIXED_SCALE = 2**16
LOGIT_SCALE = 2**16
TREE_LEAF = -1
TREE_UNDEFINED = -2

# def _clean_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
#     """
#     Convert all feature columns to numeric values and remove NaN/inf before
#     fixed-point conversion.  The same train-derived fill values are applied to
#     both train and test so that xgb.bpf.py and clf.xgb.bpf.py are consistent.
#     """
#     train_num = train_x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
#     test_num = test_x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
#
#     # Median is robust for IDS-style flow features.  Columns that are entirely
#     # missing/non-numeric are filled with 0.
#     fill_values = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
#     train_num = train_num.fillna(fill_values).fillna(0.0)
#     test_num = test_num.fillna(fill_values).fillna(0.0)
#
#     return train_num, test_num, {str(k): float(v) for k, v in fill_values.items()}


def load_train_test(train_csv: str, test_csv: str, target_col: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_train_full = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    # X_train_full, X_test, fill_values = _clean_features(X_train_full, X_test)

    y_train_series = train_df[target_col]
    y_test_series = test_df[target_col]

    y_train_full = y_train_series.values.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def fixed_point_features(X, fixed_scale: int) -> np.ndarray:
    return np.rint(np.asarray(X, dtype=np.float64) * fixed_scale).astype(np.int64)


def sigmoid_margin_to_class(raw_margin: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.where(raw_margin >= 0.0, classes[1], classes[0]).astype(np.int64)


def get_base_score_as_margin(clf) -> float:
    """
    Return the base score as a raw margin for binary:logistic.

    XGBoost stores base_score in the model configuration as a probability-like
    value for logistic objectives in many versions.  The generated C code
    compares the accumulated raw margin with zero, so we convert probability p
    to logit(p).  If the value cannot be parsed, use 0.0, which corresponds to
    p = 0.5.
    """
    try:
        config = json.loads(clf.get_booster().save_config())
        raw = config["learner"]["learner_model_param"].get("base_score", "5E-1")
        if isinstance(raw, list):
            raw = raw[0]
        raw_s = str(raw).strip()
        if raw_s.startswith("[") and raw_s.endswith("]"):
            arr = json.loads(raw_s)
            raw_s = str(arr[0])
        p = float(raw_s)
        if 0.0 < p < 1.0:
            return math.log(p / (1.0 - p))
        return p
    except Exception:
        return 0.0


def feature_index(split_name) -> int:
    if isinstance(split_name, int):
        return split_name
    s = str(split_name)
    if s.startswith("f") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    raise ValueError(f"Unsupported XGBoost feature name: {split_name!r}")


def threshold_to_int(split_condition: float) -> int:
    """
    XGBoost branches left when x < split_condition.
    Since C/BPF receives integer fixed-point features, x < t is equivalent to
    x < ceil(t) for integer x.
    """
    return int(math.ceil(float(split_condition)))


def traverse_xgb_json_tree(
    node: Dict, arrays: Dict[str, Dict[int, int]], logit_scale: int
):
    node_id = int(node["nodeid"])

    if "leaf" in node:
        arrays["children_left"][node_id] = TREE_LEAF
        arrays["children_right"][node_id] = TREE_LEAF
        arrays["features"][node_id] = TREE_UNDEFINED
        arrays["threshold"][node_id] = TREE_UNDEFINED
        arrays["value"][node_id] = int(round(float(node["leaf"]) * logit_scale))
        return

    arrays["features"][node_id] = feature_index(node["split"])
    arrays["threshold"][node_id] = threshold_to_int(node["split_condition"])
    arrays["children_left"][node_id] = int(node["yes"])
    arrays["children_right"][node_id] = int(node["no"])
    arrays["value"][node_id] = 0

    for child in node.get("children", []):
        traverse_xgb_json_tree(child, arrays, logit_scale)


def export_xgb_params(
    clf, model_path: str, num_features: int, fixed_scale: int, logit_scale: int
):
    booster = clf.get_booster()
    dumps = booster.get_dump(dump_format="json")
    parsed_trees = [json.loads(s) for s in dumps]

    per_tree = []
    max_node_id = 0
    for tree in parsed_trees:
        arrays = {
            "children_left": {},
            "children_right": {},
            "features": {},
            "threshold": {},
            "value": {},
        }
        traverse_xgb_json_tree(tree, arrays, logit_scale)
        tree_max_id = max(max(d.keys()) if d else 0 for d in arrays.values())
        max_node_id = max(max_node_id, tree_max_id)
        per_tree.append((arrays, tree_max_id))

    n_nodes = max_node_id + 1

    children_left: List[int] = []
    children_right: List[int] = []
    features: List[int] = []
    threshold: List[int] = []
    value: List[int] = []

    for arrays, _ in per_tree:
        for node_id in range(n_nodes):
            children_left.append(arrays["children_left"].get(node_id, TREE_LEAF))
            children_right.append(arrays["children_right"].get(node_id, TREE_LEAF))
            features.append(arrays["features"].get(node_id, TREE_UNDEFINED))
            threshold.append(arrays["threshold"].get(node_id, TREE_UNDEFINED))
            value.append(arrays["value"].get(node_id, 0))

    classes = np.asarray(clf.classes_, dtype=np.uint32)
    if len(classes) != 2:
        raise ValueError(
            "Current xgb.c supports binary classification only (OUTPUT_DIM must be 2)."
        )

    base_margin = get_base_score_as_margin(clf)
    xgb_base_score = int(round(base_margin * logit_scale))

    with open(model_path, "w") as f:
        f.write("#ifndef XGB_PARAMS_H\n")
        f.write("#define XGB_PARAMS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define N {n_nodes}\n")
        f.write(f"#define NUM_FEATURES {num_features}\n")
        f.write(f"#define MAX_TREE_DEPTH {getattr(clf, 'max_depth', 0)}\n")
        f.write(f"#define N_ESTIMATORS {len(parsed_trees)}\n")
        f.write(f"#define OUTPUT_DIM {len(classes)}\n")
        f.write(f"#define XGB_FEATURE_SCALE {fixed_scale}\n")
        f.write(f"#define XGB_LOGIT_SCALE {logit_scale}\n")
        f.write(f"#define XGB_BASE_SCORE {xgb_base_score}\n")
        f.write(f"#define CHILDREN_LEFT_SIZE {len(children_left)}\n")
        f.write(f"#define CHILDREN_RIGHT_SIZE {len(children_right)}\n")
        f.write(f"#define CHILDLEN_LEFT_SIZE {len(children_left)}\n")
        f.write(f"#define CHILDLEN_RIGHT_SIZE {len(children_right)}\n")
        f.write(f"#define FEATURES_SIZE {len(features)}\n")
        f.write(f"#define THRESHOLD_SIZE {len(threshold)}\n")
        f.write(f"#define VALUE_SIZE {len(value)}\n")
        f.write(f"#define XGB_CLASSES_SIZE {len(classes)}\n\n")

        write_int64_array(f, "children_left", children_left)
        write_int64_array(f, "children_right", children_right)
        write_int64_array(f, "features", features)
        write_int64_array(f, "threshold", threshold)
        write_int64_array(f, "value", value)
        write_uint_array(f, "xgb_classes", classes)
        f.write("\n#endif /* XGB_PARAMS_H */\n")


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


def main():
    parser = argparse.ArgumentParser(
        description="XGBClassifier parameter exporter for fixed-point BPF/C inference"
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
        "--n-estimators", type=int, default=10, help="木の本数 (default: 10)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=10, help="木の最大深さ (default: 10)"
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
        default="xgb_params.h",
        help="XGBパラメータヘッダの保存先 (default: xgb_params.h)",
    )
    parser.add_argument(
        "--fixed-scale",
        type=int,
        default=FIXED_SCALE,
        help="特徴量の固定小数点スケール (default: 2^16)",
    )
    parser.add_argument(
        "--logit-scale",
        type=int,
        default=LOGIT_SCALE,
        help="leaf/base_scoreの固定小数点スケール (default: 2^20)",
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

    classes = np.unique(y_train_full)
    if len(classes) != 2:
        raise ValueError(
            "Current xgb.c supports binary classification only. y_train must have exactly 2 classes."
        )
    if not np.array_equal(classes, np.array([0, 1])):
        print(
            f"[WARN] XGBoost requires labels encoded as 0/1. Current classes: {classes}"
        )
        print("[WARN] Please encode labels as 0 and 1 before running this exporter.")

    X_train_full_int = fixed_point_features(X_train_full, args.fixed_scale)
    X_test_int = fixed_point_features(X_test, args.fixed_scale)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_int,
        y_train_full,
        test_size=args.val_ratio,
        random_state=args.random_state,
        stratify=y_train_full,
    )
    print(
        f"[INFO] Train size = {X_train.shape[0]}, Val size = {X_val.shape[0]}, Test size = {X_test_int.shape[0]}"
    )
    # standard_scaler = preprocessing.StandardScaler()
    # standard_scaler.fit(X_train)
    # print(standard_scaler.mean_, standard_scaler.scale_)
    # X_train_full_norm = standard_scaler.transform(X_train_full)
    # X_train_norm = standard_scaler.transform(X_train)
    # X_val_norm = standard_scaler.transform(X_val)
    # X_test_norm = standard_scaler.transform(X_test)

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=-1,
    )

    print("[INFO] Training XGBClassifier on fixed-point features...")
    clf.fit(X_train, y_train)
    print("[INFO] Training done.")

    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n[RESULT] Validation Accuracy: {val_acc:.4f}\n")
    print(classification_report(y_val, y_val_pred))
    #
    # y_test_pred = clf.predict(X_test_int)
    # test_acc = accuracy_score(y_test, y_test_pred)
    # print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    # print("[RESULT] Classification Report (Test):")
    # print(classification_report(y_test, y_test_pred))

    # Optional consistency check against raw margin sign, which is what xgb.c uses.
    # raw_margin = clf.predict(X_test_int, output_margin=True)
    # sign_pred = sigmoid_margin_to_class(raw_margin, clf.classes_)
    # if not np.array_equal(sign_pred, y_test_pred.astype(np.int64)):
    #     mismatch = int(np.sum(sign_pred != y_test_pred.astype(np.int64)))
    #     print(
    #         f"[WARN] Raw-margin sign differs from clf.predict for {mismatch} test samples."
    #     )

    export_xgb_params(
        clf=clf,
        model_path=args.model_path,
        num_features=X_train_full_int.shape[1],
        fixed_scale=args.fixed_scale,
        logit_scale=args.logit_scale,
    )
    print(f"[INFO] Exported XGB parameters to {args.model_path}")


if __name__ == "__main__":
    main()
