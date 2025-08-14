# Day 2: NumPy基礎（配列・行列操作）

## 学習内容

NumPyの基本的な使い方を学び、配列・行列操作の基礎を習得しました。

## ファイル構成

- `numpy_basics.ipynb` - NumPyの基本操作を学ぶJupyterノートブック
- `practice_exercises.py` - 実践的な演習問題集

## 学習のポイント

### 1. 配列の作成方法
- `np.array()` - リストから配列を作成
- `np.zeros()`, `np.ones()` - 特定の値で初期化
- `np.arange()`, `np.linspace()` - 連続値の生成
- `np.random` - ランダムな配列の生成

### 2. インデックスとスライス
- 基本的なインデックス操作
- スライスによる部分配列の取得
- ブールインデックスによる条件抽出

### 3. 配列の形状変更
- `reshape()` - 形状の変更
- `flatten()`, `ravel()` - 1次元化
- `.T`, `transpose()` - 転置

### 4. 配列の演算
- 要素ごとの四則演算
- 行列積（`@` または `np.dot()`）
- ブロードキャスティング

### 5. 統計関数
- `sum()`, `mean()`, `std()`, `var()`
- `min()`, `max()`, `argmin()`, `argmax()`
- 軸指定による集約（axis=0, axis=1）

## 実行方法

### Jupyterノートブックの実行
```bash
conda activate ml-ai-env
jupyter lab numpy_basics.ipynb
```

### 実践演習の実行
```bash
conda activate ml-ai-env
python practice_exercises.py
```

## 演習問題のポイント

1. **行列操作** - 魔方陣を使った行列の基本操作
2. **ブロードキャスティング** - 学生の成績データを使った統計処理
3. **画像処理シミュレーション** - 畳み込み演算の基礎
4. **統計分析** - 2群間の比較
5. **線形代数** - 連立方程式の解法

## 📖 学んだこと（学習ログ）

- NumPyは数値計算の基盤となるライブラリで、効率的な配列操作が可能
- ブロードキャスティングにより、異なる形状の配列間でも演算が可能
- 軸（axis）を指定することで、多次元配列の特定の方向に沿った集約ができる
- NumPyの配列操作は、今後学ぶPandasやTensorFlow/PyTorchの基礎となる
- ベクトル化された演算により、Pythonのループよりも高速な処理が可能

## 次のステップ

明日（Day 3）は、Pandas基礎①（DataFrame操作）を学習予定です。NumPyの知識を基に、より高レベルなデータ操作を習得します。
