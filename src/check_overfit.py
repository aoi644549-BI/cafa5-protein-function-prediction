import torch
import esm
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

records = list(SeqIO.parse("data/Train/train_sequences.fasta", "fasta"))[:1000]
terms_df = pd.read_csv("data/Train/train_terms.tsv", sep="\t")
ids = [r.id for r in records]
terms_sub = terms_df[terms_df['EntryID'].isin(ids)]
top_terms = terms_sub['term'].value_counts().head(50).index.tolist()

label_matrix = np.zeros((len(ids), len(top_terms)), dtype=int)
term_to_idx = {t: i for i, t in enumerate(top_terms)}
id_to_idx = {id_: i for i, id_ in enumerate(ids)}
for _, row in terms_sub.iterrows():
    if row['EntryID'] in id_to_idx and row['term'] in term_to_idx:
        label_matrix[id_to_idx[row['EntryID']], term_to_idx[row['term']]] = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model = model.to(device).eval()
batch_converter = alphabet.get_batch_converter()

embeddings = []
for i, r in enumerate(records):
    seq = str(r.seq)[:512]
    data = [(r.id, seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    emb = results["representations"][12].mean(dim=1).cpu().numpy()
    embeddings.append(emb[0])
    del tokens, results
    torch.cuda.empty_cache()

X = np.array(embeddings)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = label_matrix[:split], label_matrix[split:]

# C値を変えて比較
print("===== 正則化の強さによる比較 =====")
for C in [0.01, 0.1, 0.5, 1.0]:
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=C), n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    f1_train = f1_score(y_train, y_pred_train, average='micro', zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, average='micro', zero_division=0)
    
    print(f"\nC={C}")
    print(f"  訓練 F1 micro: {f1_train:.4f}")
    print(f"  テスト F1 micro: {f1_test:.4f}")
    print(f"  差分: {f1_train - f1_test:.4f}")
