# CAFA 5 Protein Function Prediction

## 概要

タンパク質のアミノ酸配列からGene Ontology（GO）タームを予測する多ラベル分類モデルの実装です。
タンパク質言語モデル ESM2 で配列を埋め込みベクトルに変換し、機械学習モデルで機能を予測します。

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

## 技術スタック

- **Python** 3.10
- **ESM2**（Meta AI）: タンパク質言語モデル
- **PyTorch**: GPU推論（RTX 5050）
- **scikit-learn**: ロジスティック回帰・評価
- **LightGBM**: 勾配ブースティング
- **BioPython**: FASTAファイル処理

## パイプライン
```
アミノ酸配列（FASTA）
  ↓ src/embed.py（ESM2-150M・640次元・GPU推論）
埋め込みベクトル（.npyファイルに保存）
  ↓ src/model.py または src/model_lgbm.py
GOターム予測スコア → F1スコアで評価
```

## ベストスコア

| モデル | テストF1 micro | テストF1 macro | 過学習差分 |
|---|---|---|---|
| **ESM2-150M + Logistic Regression** | **0.5725** | **0.3749** | **0.0386** ✅ |
| ESM2-150M + LightGBM | 0.4514 | 0.1650 | 0.0594 ✅ |

詳細なスコアの推移は [results/scores.md](results/scores.md) を参照。

## ファイル構成
```
├── src/
│   ├── explore.py        # データ探索・EDA
│   ├── embed.py          # ESM2埋め込み生成・保存
│   ├── model.py          # ロジスティック回帰モデル
│   ├── model_lgbm.py     # LightGBMモデル
│   └── check_overfit.py  # 過学習検証
├── results/
│   └── scores.md         # スコアの推移・考察
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

# STEP 2: 埋め込み生成・保存（一度だけ実行・約10分）
python src/embed.py --n_sequences 5000 --model_size 150M

# STEP 3: モデル学習・評価（数秒で実行可能）
python src/model.py
python src/model_lgbm.py --n_estimators 50 --num_leaves 7
```

## 実験で得られた知見

### 1. 埋め込み生成と学習の分離が重要
埋め込み生成を一度だけ行いnpyファイルに保存することで、実験の再現性が保証され、
ハイパーパラメータ探索を高速に行える。

### 2. データ量とGOターム数のトレードオフ
GOターム数を増やすと各タームの学習データが減り精度が低下する。
5,000件のデータでは50〜100タームが適切な上限。

### 3. モデル選択
5,000件ではロジスティック回帰がLightGBMより優秀。
LightGBMは過学習しやすく、データ量が少ない場合は強い正則化が必要。

## 今後の改善方針

1. 全データ（142,246件）での学習
2. GO階層構造を活用したプロパゲーション後処理
3. ProtT5など複数タンパク質言語モデルのアンサンブル
4. ニューラルネットワークへの変更

## 参考

- [CAFA 5 Kaggleコンペ](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
- [ESM2論文 (Lin et al., Science 2023)](https://www.science.org/doi/10.1126/science.ade2574)
- [Gene Ontology](http://geneontology.org/)
- [2位チームの解法](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/discussion/433207)
