#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from model import MLP


def load_train_test(
    train_csv: str, test_csv: str, target_col: str, feature_importance_csv: str
):
    """train/test を別CSVから読み込み、学習ラベルでfactorizeしたコードに揃える。"""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    feature_importance = pd.read_csv(feature_importance_csv, index_col=0)
    feature_importance = feature_importance.sort_values("mean_rank", ascending=True)[
        :16
    ].index.values

    # 特徴量
    X_train_full = train_df.drop(columns=[target_col])[
        feature_importance
    ].values.astype(np.float32)
    X_test = test_df.drop(columns=[target_col])[feature_importance].values.astype(
        np.float32
    )

    # ラベル（学習側でfactorize → 同じマッピングでtestもエンコード）
    y_train_series = train_df[target_col]
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

    y_train_full = y_train_series.values.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


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
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--model-path", type=str, default="mlp.pth", help="学習率")
    parser.add_argument(
        "--scaler-path", type=str, default="scaler.joblib", help="学習率"
    )
    parser.add_argument(
        "--feature_importance",
        type=str,
        default="../dataset/csv/flow/feature_importance.csv",
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
    X_train_full, y_train_full, X_test, y_test = load_train_test(
        args.train_csv, args.test_csv, args.target, args.feature_importance
    )
    print(
        f"[INFO] X_train_full shape = {X_train_full.shape}, y_train_full shape = {y_train_full.shape}"
    )
    print(
        f"[INFO] X_test shape       = {X_test.shape}, y_test shape       = {y_test.shape}"
    )

    # === train/val 分割（学習CSVの一部をValidationに） ===
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.val_ratio,
        random_state=42,
        stratify=y_train_full,
    )

    # === 正規化（学習データ全体の統計量を使う） ===
    # X_train_full_norm, X_train_norm, X_val_norm, X_test_norm = standardize_with_train(
    #     X_train_full, X_train, X_val, X_test
    # )
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
    X_train_t = torch.from_numpy(X_train_norm).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val_norm).float()
    y_val_t = torch.from_numpy(y_val).long()
    X_test_t = torch.from_numpy(X_test_norm).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size, shuffle=False
    )

    in_dim = X_train.shape[1]
    # num_classes = len(label_mapping)
    num_classes = 2
    model = MLP(in_dim, hidden_dims, num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # === 学習ループ ===
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()

        train_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        val_acc = evaluate(model, val_loader, device)

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"TrainLoss={train_loss:.4f} "
                f"TrainAcc={train_acc:.4f} "
                f"ValAcc={val_acc:.4f}"
            )

    # === Test で classification_report ===
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    print("\n[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_pred))

    # サンプル表示
    print("\n[RESULT] Sample predictions on Test (first 5):")
    for i in range(min(5, len(y_test))):
        print(f"  true={y_test[i]}, pred={y_pred[i]}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        args.model_path,
    )


if __name__ == "__main__":
    main()
