"""
model_nn_tax.py
ESM2埋め込み + タクソノミー情報でGOターム予測
使い方: python src/model_nn_tax.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import argparse

class ProteinFunctionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ===== 埋め込み読み込み =====
    print("埋め込み読み込み中...")
    X = np.load(args.emb_path)
    ids = np.load(args.ids_path)
    print(f"ESM2埋め込み形状: {X.shape}")

    # ===== タクソノミー情報の追加 =====
    print("タクソノミー情報読み込み中...")
    tax_df = pd.read_csv(args.tax_path, sep="\t")
    tax_df = tax_df.set_index('EntryID')

    # 出現頻度上位N個のtaxonomyIDをone-hot encoding
    top_tax = tax_df['taxonomyID'].value_counts().head(args.n_tax).index.tolist()
    print(f"使用するtaxonomyID数: {len(top_tax)}")

    tax_matrix = np.zeros((len(ids), len(top_tax)), dtype=np.float32)
    tax_to_idx = {t: i for i, t in enumerate(top_tax)}

    for i, id_ in enumerate(ids):
        if id_ in tax_df.index:
            tax_id = tax_df.loc[id_, 'taxonomyID']
            if tax_id in tax_to_idx:
                tax_matrix[i, tax_to_idx[tax_id]] = 1.0

    # ESM2埋め込みとタクソノミーを結合
    X_combined = np.concatenate([X, tax_matrix], axis=1)
    print(f"結合後の特徴量形状: {X_combined.shape}")

    # ===== ラベル作成 =====
    print("ラベル作成中...")
    terms_df = pd.read_csv(args.terms_path, sep="\t")
    terms_sub = terms_df[terms_df['EntryID'].isin(ids)]
    top_terms = terms_sub['term'].value_counts().head(args.n_terms).index.tolist()
    print(f"使用するGOターム数: {len(top_terms)}")

    label_matrix = np.zeros((len(ids), len(top_terms)), dtype=np.float32)
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    for _, row in terms_sub.iterrows():
        if row['EntryID'] in id_to_idx and row['term'] in term_to_idx:
            label_matrix[id_to_idx[row['EntryID']], term_to_idx[row['term']]] = 1.0

    # ===== 訓練/テスト分割 =====
    split = int(len(X_combined) * 0.8)
    X_train = torch.tensor(X_combined[:split], dtype=torch.float32)
    X_test  = torch.tensor(X_combined[split:], dtype=torch.float32)
    y_train = torch.tensor(label_matrix[:split], dtype=torch.float32)
    y_test  = torch.tensor(label_matrix[split:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )

    # ===== モデル構築 =====
    input_dim = X_combined.shape[1]
    model = ProteinFunctionNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(top_terms),
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # ===== 学習 =====
    print(f"\n学習開始（{args.epochs}エポック）...")
    print(f"入力次元: {input_dim} = ESM2({X.shape[1]}) + taxonomy({len(top_tax)})")
    best_f1 = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            pred_test  = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
            pred_train = torch.sigmoid(model(X_train.to(device))).cpu().numpy()

        y_pred       = (pred_test  > 0.5).astype(int)
        y_pred_train = (pred_train > 0.5).astype(int)

        f1_test  = f1_score(y_test.numpy(),  y_pred,       average='micro', zero_division=0)
        f1_train = f1_score(y_train.numpy(), y_pred_train, average='micro', zero_division=0)
        f1_macro = f1_score(y_test.numpy(),  y_pred,       average='macro', zero_division=0)

        scheduler.step(1 - f1_test)

        if f1_test > best_f1:
            best_f1 = f1_test

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train F1: {f1_train:.4f} | "
              f"Test F1: {f1_test:.4f} | "
              f"Macro F1: {f1_macro:.4f} | "
              f"差分: {f1_train-f1_test:.4f}")

    print(f"\n===== 最終結果 =====")
    print(f"ベストテスト F1 micro: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path",    default="data/embeddings/embeddings_esm2_t30_150M_UR50D_142246.npy")
    parser.add_argument("--ids_path",    default="data/embeddings/ids_142246.npy")
    parser.add_argument("--terms_path",  default="data/Train/train_terms.tsv")
    parser.add_argument("--tax_path",    default="data/Train/train_taxonomy.tsv")
    parser.add_argument("--n_terms",     type=int,   default=1000)
    parser.add_argument("--n_tax",       type=int,   default=200)
    parser.add_argument("--hidden_dim",  type=int,   default=1024)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--weight_decay",type=float, default=0.0)
    args = parser.parse_args()
    main(args)
