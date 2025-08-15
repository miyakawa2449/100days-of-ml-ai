# Day 3: Pandas基礎① - DataFrame操作

## 学習テーマ
Pandasの中心的なデータ構造であるDataFrameの基本操作を学習

## 学習概要
- DataFrameの作成方法（辞書、リスト、NumPy配列から）
- データの選択、フィルタリング、ソート
- 統計情報の取得とグループ集計
- CSVファイルの読み書き
- 欠損値の処理と列の操作

## ファイル構成
- `main.py`: DataFrameの各種操作を実装したメインプログラム
- `sample_data.csv`: プログラム実行時に生成されるサンプルCSVファイル

## 実行方法
```bash
conda activate ml-ai-env
cd day3
python main.py
```

## 使い方
プログラムを実行すると、以下の順番でDataFrame操作のデモが実行されます：
1. DataFrameの作成方法（3種類）
2. 基本操作（選択、フィルタリング、ソート）
3. 統計情報の取得
4. CSVファイルの読み書き
5. その他の便利な操作（列の追加・削除、欠損値処理）

## 📖 学んだこと（学習ログ）

### DataFrameの作成
- **辞書から**: `pd.DataFrame({'col1': [値1, 値2], 'col2': [値3, 値4]})`
- **リストから**: `pd.DataFrame([[値1, 値2], [値3, 値4]], columns=['col1', 'col2'])`
- **NumPy配列から**: `pd.DataFrame(np_array, columns=['col1', 'col2'])`

### データアクセス
- **列選択**: `df['column']`（単一）、`df[['col1', 'col2']]`（複数）
- **行選択**: `df.iloc[0]`（インデックス）、`df.iloc[1:3]`（スライス）
- **条件フィルタ**: `df[df['age'] >= 30]`、`df[(条件1) & (条件2)]`

### 統計と集計
- **基本情報**: `df.info()` - データ型、欠損値の確認
- **統計サマリー**: `df.describe()` - count, mean, std, min, max等
- **グループ集計**: `df.groupby('column')['target'].mean()`

### CSV操作
- **書き込み**: `df.to_csv('file.csv', index=False)`
- **読み込み**: `pd.read_csv('file.csv', usecols=['col1', 'col2'])`

### その他の重要な操作
- **新規列追加**: `df['new'] = df['A'] + df['B']`
- **列削除**: `df.drop('column', axis=1)`
- **欠損値確認**: `df.isnull().sum()`
- **欠損値補完**: `df.fillna(0)`
