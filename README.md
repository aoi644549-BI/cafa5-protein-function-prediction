# CAFA 5 Protein Function Prediction

## 概要

タンパク質のアミノ酸配列からGene Ontology（GO）タームを予測する多ラベル分類モデルの実装です。
タンパク質言語モデル ESM2 で配列を埋め込みベクトルに変換し、タクソノミー情報と組み合わせて機能を予測します。

## 取り組む背景・モチベーション

TRT創薬を専攻する立場から、タンパク質機能予測は創薬の最上流に位置する重要な問題です。
ヒトゲノムの約2万タンパク質のうち、現在の医薬品が標的にしているのは約700個のみです。
未知タンパク質の機能を精度よく予測することで、新たな創薬ターゲットの発見につながります。
```
アミノ酸配列
  ↓
タンパク質機能予測（このプロジェクト）
  ↓
創薬ターゲットの同定
  ↓
化合物設計（TRT創薬の専門領域）
  ↓
医薬品開発
```

## ベストスコア

| モデル | F1 micro | F1 macro |
|---|---|---|
| **ESM2-150M + NN + Taxonomy** | **0.6763** | **0.5987** |

詳細は [results/scores.md](results/scores.md) を参照。

## 技術スタック

- **Python** 3.10
- **ESM2**（Meta AI）: タンパク質言語モデル
- **PyTorch**: GPU推論・ニューラルネットワーク学習
- **scikit-learn**: ロジスティック回帰・評価
- **LightGBM**: 勾配ブースティング
- **BioPython**: FASTAファイル処理

## パイプライン
```
アミノ酸配列（FASTA） + タクソノミー情報（TSV）
  ↓ src/embed.py（ESM2-150M・640次元・GPU推論）
埋め込みベクトル（.npyファイルに保存）
  ↓ src/model_nn_tax.py
ESM2埋め込み（640次元）+ taxonomy one-hot（200次元）
  ↓ ニューラルネットワーク（3層）
GOターム予測スコア → F1スコアで評価
```

## ファイル構成
```
├── src/
│   ├── explore.py          # データ探索・EDA
│   ├── embed.py            # ESM2埋め込み生成・保存
│   ├── model.py            # ロジスティック回帰モデル
│   ├── model_lgbm.py       # LightGBMモデル
│   ├── model_nn.py         # ニューラルネットワーク（ESM2のみ）
│   ├── model_nn_tax.py     # ニューラルネットワーク（ESM2+taxonomy）
│   └── check_overfit.py    # 過学習検証
├── results/
│   └── scores.md           # スコアの推移・考察
├── requirements.txt
└── README.md
```

## セットアップ
```bash
# 環境構築
conda create -n cafa5 python=3.10 -y
conda activate cafa5
pip install -r requirements.txt

# データはKaggleからダウンロード
# https://www.kaggle.com/competitions/cafa-5-protein-function-prediction
kaggle competitions download -c cafa-5-protein-function-prediction
unzip cafa-5-protein-function-prediction.zip -d data/
```

## 実行方法
```bash
# STEP 1: データ探索
python src/explore.py

# STEP 2: 埋め込み生成・保存（一度だけ実行・約4時間）
python src/embed.py --n_sequences -1 --model_size 150M

# STEP 3: モデル学習・評価
# ロジスティック回帰
python src/model.py

# LightGBM
python src/model_lgbm.py

# ニューラルネットワーク（ESM2のみ）
python src/model_nn.py --epochs 50 --hidden_dim 1024

# ニューラルネットワーク（ESM2+taxonomy）← ベストモデル
python src/model_nn_tax.py --epochs 50 --hidden_dim 1024 --dropout 0.2
```

## 実験で得られた知見

### 1. タクソノミー情報の追加が最も効果的
ESM2埋め込みに生物種情報を追加するだけでF1 microが0.59→0.68に大幅改善。
「どの生物種のタンパク質か」という情報がGOターム予測に非常に有効。

### 2. 埋め込み生成と学習の分離
埋め込み生成を一度だけ行いnpyファイルに保存することで実験の再現性が保証され、
ハイパーパラメータ探索を高速に行える。

### 3. データ量とモデルの表現力
全データ142,246件では線形モデルより非線形モデル（NN）が有効。
データが増えるほどNNの恩恵が大きくなる。

## 今後の改善方針

1. GOターム数の拡張（50→5,000個）
2. GO階層構造を活用したプロパゲーション後処理
3. ProtT5など複数タンパク質言語モデルのアンサンブル

## 参考

- [CAFA 5 Kaggleコンペ](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
- [ESM2論文 (Lin et al., Science 2023)](https://www.science.org/doi/10.1126/science.ade2574)
- [Gene Ontology](http://geneontology.org/)
- [2位チームの解法](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/433207)
