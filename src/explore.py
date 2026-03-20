from Bio import SeqIO
import pandas as pd

# タンパク質配列の読み込み
records = list(SeqIO.parse("data/Train/train_sequences.fasta", "fasta"))
print(f"タンパク質数: {len(records)}")
print(f"\n最初の3件:")
for r in records[:3]:
    print(f"  ID: {r.id}, 配列長: {len(r.seq)}")

# GOタームの読み込み
terms = pd.read_csv("data/Train/train_terms.tsv", sep="\t")
print(f"\nGOターム数: {len(terms)}")
print(f"カラム: {terms.columns.tolist()}")
print(f"\n最初の5行:")
print(terms.head())

# aspectの分布確認
print("\n=== aspect別の分布 ===")
print(terms['aspect'].value_counts())

# 1タンパク質あたりのGOターム数
go_per_protein = terms.groupby('EntryID')['term'].count()
print(f"\n=== 1タンパク質あたりのGOターム数 ===")
print(f"平均: {go_per_protein.mean():.1f}")
print(f"最小: {go_per_protein.min()}")
print(f"最大: {go_per_protein.max()}")
print(f"中央値: {go_per_protein.median():.1f}")

# 配列長の統計
seq_lengths = [len(r.seq) for r in records]
print(f"\n=== 配列長の統計 ===")
print(f"平均: {sum(seq_lengths)/len(seq_lengths):.1f}")
print(f"最小: {min(seq_lengths)}")
print(f"最大: {max(seq_lengths)}")
