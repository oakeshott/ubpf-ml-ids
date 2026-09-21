#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from model import MLP
import warnings

warnings.filterwarnings("ignore")


def load_train_test(train_csv: str, test_csv: str, target_col: str):
    """train/test を別CSVから読み込み、学習ラベルでfactorizeしたコードに揃える。"""
    test_df = pd.read_csv(test_csv)

    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    # 特徴量
    X_test = test_df.drop(columns=[target_col]).values.astype(np.float32)

    # ラベル（学習側でfactorize → 同じマッピングでtestもエンコード）
    y_test_series = test_df[target_col]

    # y_train_codes, uniques = pd.factorize(y_train_series)
    # label_mapping = {cls: idx for idx, cls in enumerate(uniques)}
    #
    # print("[INFO] Label mapping (train):")
    # for cls, idx in label_mapping.items():
    #     print(f"  {idx} -> {cls}")
    #
    # y_test_codes = y_test_series.map(label_mapping)
    # if y_test_codes.isnull().any():
    #     unknown = y_test_series[y_test_codes.isnull()].unique()
    #     raise ValueError(f"テストデータに学習で見ていないラベルがあります: {unknown}")

    y_test = y_test_series.values.astype(np.int64)

    return X_test, y_test


def standardize_with_train(X_train_full, *others):
    """学習データ全体の統計量で正規化し、他の配列にも同じ変換を適用。"""
    mean = X_train_full.mean(axis=0, keepdims=True)
    std = X_train_full.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    def norm(x):
        return (x - mean) / std

    return (norm(X_train_full),) + tuple(norm(o) for o in others)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch MLP: train/val/test を別CSVから"
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
        "--hidden", type=str, default="100", help="隠れ層ユニット（例: 128,64）"
    )
    parser.add_argument("--epochs", type=int, default=50, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--model-path", type=str, default="mlp.pth", help="学習率")
    parser.add_argument(
        "--scaler-path", type=str, default="scaler.joblib", help="学習率"
    )
    parser.add_argument(
        "--resdir", type=str, default="results", help="Output directory"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    hidden_dims = tuple(int(x) for x in args.hidden.split(","))

    # === データ読み込み ===
    print(f"[INFO] Loading train from {args.train_csv}")
    print(f"[INFO] Loading test  from {args.test_csv}")
    X_test, y_test = load_train_test(args.train_csv, args.test_csv, args.target)
    print(
        f"[INFO] X_test shape       = {X_test.shape}, y_test shape       = {y_test.shape}"
    )

    # === 正規化（学習データ全体の統計量を使う） ===
    # X_train_full_norm, X_train_norm, X_val_norm, X_test_norm = standardize_with_train(
    #     X_train_full, X_train, X_val, X_test
    # )
    standard_scaler = joblib.load(args.scaler_path)
    # X_train_full_norm = standard_scaler.transform(X_train_full)
    # X_train_norm = standard_scaler.transform(X_train)
    # X_val_norm = standard_scaler.transform(X_val)
    X_test_norm = standard_scaler.transform(X_test)

    # X_train_full_norm = (X_train_full_norm * (2 ** 16)).round()
    # X_train_norm = (X_train_norm * (2 ** 16)).round()
    # X_val_norm   = (X_val_norm * (2 ** 16)).round()
    # X_test_norm  = (X_test_norm * (2 ** 16)).round()

    # numpy -> tensor
    X_test_t = torch.from_numpy(X_test_norm).float()
    y_test_t = torch.from_numpy(y_test).long()

    # test_loader = DataLoader(
    #     TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size, shuffle=False
    # )

    in_dim = X_test.shape[1]
    # num_classes = len(label_mapping)
    num_classes = 2
    model = MLP(in_dim, hidden_dims, num_classes).to(device)
    print(model)
    saved_dict = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(saved_dict["model_state_dict"])

    # === Test で classification_report ===
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    print("\n[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_pred))
    ret = classification_report(y_test, y_pred, digits=6, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    ret["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    os.makedirs(args.resdir, exist_ok=True)
    with open(f"{args.resdir}/classification_report.json", "w") as f:
        json.dump(
            ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": ")
        )


if __name__ == "__main__":
    main()
