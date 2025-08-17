# Day 5: Matplotlib/Seaborn基礎（データ可視化）

## 概要
データ可視化の基礎を学習します。MatplotlibとSeabornを使って、様々な種類のグラフを作成し、データの傾向や関係性を視覚的に表現する方法を習得します。

## 学習内容

### 初心者向け（matplotlib_seaborn_basic_simple.ipynb）
- **基本的なグラフの作成**
  - 折れ線グラフ（時系列データの推移）
  - 棒グラフ（カテゴリ間の比較）
  - 円グラフ（割合の表示）
  - 散布図（2変数の関係）
- **グラフの装飾**
  - タイトル、ラベル、凡例の追加
  - 色とスタイルの設定
  - 日本語表示の設定
- **Seabornの基礎**
  - 箱ひげ図（データの分布）
  - ヒートマップ（相関関係の可視化）
- **実践的な例**
  - カフェの売上データを使った分析
  - 曜日別、月別の傾向分析

### ベーシック版（matplotlib_seaborn_basic.ipynb）
- **高度な可視化技術**
  - オブジェクト指向インターフェース（Figure, Axes）
  - GridSpecを使った複雑なレイアウト
  - 3Dプロットと極座標プロット
- **統計的可視化**
  - 回帰プロット
  - バイオリンプロット
  - KDE（カーネル密度推定）プロット
  - FacetGridによる多次元可視化
- **カスタマイズテクニック**
  - カスタムカラーマップ
  - 二軸グラフ
  - アニメーション基礎
- **インタラクティブ可視化**
  - Plotlyの基本的な使い方
  - ダッシュボードの作成

## 主な学習ポイント

### グラフ選択の基準
- **推移を見る** → 折れ線グラフ
- **比較する** → 棒グラフ
- **割合を見る** → 円グラフ
- **相関を見る** → 散布図
- **分布を見る** → ヒストグラム、箱ひげ図

### 良いグラフを作るコツ
1. 適切なグラフタイプの選択
2. 明確なタイトルとラベル
3. 見やすい色使い（多すぎない）
4. 適切なスケール設定
5. 必要に応じた注釈の追加

## 実行環境
- Python 3.x
- 必要なライブラリ：
  - matplotlib
  - seaborn
  - pandas
  - numpy
  - plotly（オプション）
  - japanize-matplotlib（日本語表示用、推奨）

### 日本語表示について
Matplotlibで日本語を表示するには、以下のいずれかの方法を使用してください：

**方法1: japanize-matplotlibを使用（推奨）**
```bash
pip install japanize-matplotlib
```

使用方法：
```python
import matplotlib.pyplot as plt
import japanize_matplotlib  # これだけで日本語が使えるようになります
```

**方法2: 手動でフォントを設定**
- macOS: Hiragino Sans, Yu Gothic など
- Windows: Yu Gothic, Meiryo など
- Linux: Noto Sans CJK JP など

詳細はノートブック内のフォント設定セクションを参照してください。

## 使い方
1. Jupyter Notebookを起動
2. 初心者の方は `matplotlib_seaborn_basic_simple.ipynb` から始める
3. 基礎ができたら `matplotlib_seaborn_basic.ipynb` で応用を学ぶ

## 演習問題
各ノートブックには実践的な演習問題が含まれています：
- 積み上げ棒グラフの作成
- 複数グラフの配置
- カスタムダッシュボードの作成
- 異常値検出の可視化

## 次のステップ
- Day 6: データ型と欠損値処理
- より高度な可視化（地理空間データ、ネットワーク図など）
- インタラクティブダッシュボードの構築（Dash, Streamlit）