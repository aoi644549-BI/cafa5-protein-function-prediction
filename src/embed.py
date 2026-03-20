"""
embed.py
ESM2でタンパク質配列の埋め込みベクトルを生成し、npyファイルに保存する
使い方: python src/embed.py --n_sequences 5000 --model_size 150M
"""

import torch
import esm
import numpy as np
from Bio import SeqIO
import argparse
import os

def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ===== モデルサイズの選択 =====
    model_map = {
        "35M":  ("esm2_t12_35M_UR50D",  12),
        "150M": ("esm2_t30_150M_UR50D", 30),
        "650M": ("esm2_t33_650M_UR50D", 33),
    }
    if args.model_size not in model_map:
        raise ValueError(f"model_sizeは35M / 150M / 650M のいずれかを指定してください")

    model_name, repr_layer = model_map[args.model_size]

    # ===== データ読み込み =====
    print(f"データ読み込み中... ({args.n_sequences}件)")
    records = list(SeqIO.parse(args.fasta_path, "fasta"))
    if args.n_sequences > 0:
        records = records[:args.n_sequences]
    ids = [r.id for r in records]
    print(f"対象タンパク質数: {len(records)}")

    # ===== ESM2モデル読み込み =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"ESM2モデル読み込み中... ({model_name})")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("読み込み完了！")

    # ===== 埋め込み生成 =====
    print("埋め込み生成中...")
    embeddings = []
    for i, r in enumerate(records):
        seq = str(r.seq)[:args.max_seq_len]
        data = [(r.id, seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer])
        emb = results["representations"][repr_layer].mean(dim=1).cpu().numpy()
        embeddings.append(emb[0])
        if (i + 1) % 500 == 0:
            print(f"  進捗: {i+1}/{len(records)}")
        del tokens, results
        torch.cuda.empty_cache()

    embeddings = np.array(embeddings)
    print(f"埋め込み形状: {embeddings.shape}")

    # ===== 保存 =====
    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, f"embeddings_{model_name}_{len(records)}.npy")
    ids_path = os.path.join(args.output_dir, f"ids_{len(records)}.npy")

    np.save(emb_path, embeddings)
    np.save(ids_path, np.array(ids))

    print(f"\n保存完了！")
    print(f"  埋め込み: {emb_path}")
    print(f"  ID一覧:   {ids_path}")
    print(f"  VRAM使用量: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM2埋め込み生成・保存")
    parser.add_argument("--fasta_path",   default="data/Train/train_sequences.fasta")
    parser.add_argument("--output_dir",   default="data/embeddings")
    parser.add_argument("--model_size",   default="150M", choices=["35M", "150M", "650M"])
    parser.add_argument("--n_sequences",  type=int, default=5000)
    parser.add_argument("--max_seq_len",  type=int, default=512)
    args = parser.parse_args()
    main(args)
