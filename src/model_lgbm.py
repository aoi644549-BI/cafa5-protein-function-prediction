"""
model_lgbm.py
保存済み埋め込みを読み込んでLightGBMでGOターム予測モデルを学習・評価する
使い方: python src/model_lgbm.py --emb_path data/embeddings/embeddings_esm2_t30_150M_UR50D_5000.npy
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
import argparse
import os


def main(args):
    # ===== 埋め込み読み込み =====
    print("埋め込み読み込み中...")
    X = np.load(args.emb_path)
    ids = np.load(args.ids_path)
    print(f"埋め込み形状: {X.shape}")

    # ===== ラベル作成 =====
    print("ラベル作成中...")
    terms_df = pd.read_csv(args.terms_path, sep="\t")
    terms_sub = terms_df[terms_df['EntryID'].isin(ids)]
    top_terms = terms_sub['term'].value_counts().head(args.n_terms).index.tolist()
    print(f"使用するGOターム数: {len(top_terms)}")

    label_matrix = np.zeros((len(ids), len(top_terms)), dtype=int)
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    for _, row in terms_sub.iterrows():
        if row['EntryID'] in id_to_idx and row['term'] in term_to_idx:
            label_matrix[id_to_idx[row['EntryID']], term_to_idx[row['term']]] = 1

    # ===== 訓練/テスト分割 =====
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = label_matrix[:split], label_matrix[split:]

    # ===== LightGBM学習 =====
    print(f"\nモデル学習中（LightGBM × {len(top_terms)}個）...")
    preds = np.zeros_like(y_test, dtype=float)
    preds_train = np.zeros_like(y_train, dtype=float)

    for i, term in enumerate(top_terms):
        clf = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            verbose=-1
        )
        clf.fit(
            X_train, y_train[:, i],
            eval_set=[(X_test, y_test[:, i])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds[:, i] = clf.predict_proba(X_test)[:, 1]
        preds_train[:, i] = clf.predict_proba(X_train)[:, 1]

        if (i + 1) % 20 == 0:
            print(f"  進捗: {i+1}/{len(top_terms)}")

    # ===== 評価 =====
    y_pred = (preds > 0.5).astype(int)
    y_pred_train = (preds_train > 0.5).astype(int)

    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_train = f1_score(y_train, y_pred_train, average='micro', zero_division=0)

    print(f"\n===== 評価結果（LightGBM）=====")
    print(f"訓練 F1 micro: {f1_train:.4f}")
    print(f"テスト F1 micro: {f1_micro:.4f}")
    print(f"テスト F1 macro: {f1_macro:.4f}")
    print(f"差分（過学習指標）: {f1_train - f1_micro:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBMによるGOターム予測")
    parser.add_argument("--emb_path",      default="data/embeddings/embeddings_esm2_t30_150M_UR50D_5000.npy")
    parser.add_argument("--ids_path",      default="data/embeddings/ids_5000.npy")
    parser.add_argument("--terms_path",    default="data/Train/train_terms.tsv")
    parser.add_argument("--n_terms",       type=int,   default=100)
    parser.add_argument("--n_estimators",  type=int,   default=200)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves",    type=int,   default=31)
    args = parser.parse_args()
    main(args)
