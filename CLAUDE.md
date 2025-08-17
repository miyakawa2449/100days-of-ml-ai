# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

これは100日間のAI・機械学習・データ分析チャレンジを管理するプロジェクトです。日々の学習進捗を記録し、プロジェクトを作成・管理するためのツールを提供します。

## 環境設定

このプロジェクトは conda 環境 `ml-ai-env` を使用します。

```bash
# conda環境のアクティベート
conda activate ml-ai-env
```

### 重要: Matplotlib日本語表示設定

**matplotlibでグラフを作成する際は、必ずjapanize-matplotlibを使用して日本語表示を有効化してください。**

```bash
# 初回のみインストール
pip install japanize-matplotlib
```

```python
# 必須: matplotlibを使用するJupyterノートブックの冒頭で実行
import matplotlib.pyplot as plt
import japanize_matplotlib  # これで日本語が正常に表示されます
```

この設定を忘れると、グラフのタイトル・軸ラベル・凡例の日本語が文字化け（□□□）になります。

## 主要なコマンド

### 新しい学習プロジェクトの作成
```bash
python create-project.py <プロジェクト名>
```
指定したフォルダに空の `main.py` と `README.md` を作成します。

### 今日の学習を完了としてマーク
```bash
python scripts/mark_today_done.py --day <日数> [--notes "メモ"] [--date "YYYY-MM-DD"]
```
- `--day`: 完了した日数（必須）
- `--notes`: 学習メモ（任意）
- `--date`: 日付を指定（省略時はJSTの今日）

### READMEの進捗表を更新
```bash
python scripts/update_readme.py
```
`progress/progress.csv` の内容を基に、README.md の進捗セクションを自動更新します。

## プロジェクト構造

- `create-project.py`: 新しい学習プロジェクトのひな形を作成
- `scripts/mark_today_done.py`: progress.csv の特定の日を完了状態に更新
- `scripts/update_readme.py`: README.md の進捗表示を自動生成・更新
- `progress/progress.csv`: 100日間の学習計画と進捗を管理するCSVファイル
  - カラム: Day, Phase, Theme, Date, Done, Notes

## 学習フェーズ

1. **基礎強化** (Day 1-15): Python基礎、Numpy、Pandas、統計、線形代数
2. **ML基礎** (Day 16-35): 機械学習の基本アルゴリズム、Scikit-learn、Kaggleタイタニック
3. **AI基礎** (Day 36-60): 深層学習、TensorFlow/Keras、CNN、RNN、NLP
4. **応用** (Day 61-85): Kaggle実践、各種アプリケーション開発
5. **仕上げ** (Day 86-100): オリジナルプロジェクトの企画・実装・公開

## 開発ワークフロー

1. 新しい学習トピック用のプロジェクトを作成
2. 学習を完了したら `mark_today_done.py` で記録
3. `update_readme.py` でREADMEの進捗を更新
4. 必要に応じてGitにコミット

## SNS投稿フォーマット

学習完了後のSNS（Twitter/X）投稿用テンプレート：

```
#100DaysOfMLChallenge Day X 完了！ [絵文字]

📅 Day X: [テーマ]（MM/DD）

今日の学習内容：
- [主要な学習項目1]
- [主要な学習項目2]
- [主要な学習項目3]

💡 学習のポイント：
[今日の気づきや工夫した点を1-2文で]

📊 進捗: X/100日 (X%)
🔗 https://github.com/miyakawa2449/100days-of-ml-ai

#機械学習 #Python #[関連タグ] #DataScience #100日チャレンジ #プログラミング学習
```

## Gitコミットルール

### コミット時の手順
1. **git status** で変更内容を確認
2. **git add** で必要なファイルをステージング
   - 学習用ファイル: `git add day*/`
   - 進捗管理: `git add progress/` `git add README.md`
   - スクリプト: `git add scripts/`
3. **git commit** でコミット（日本語メッセージ）

### コミットメッセージ形式
```
[Day X] タイトル（概要）

- 具体的な変更内容1
- 具体的な変更内容2
```

### コミットメッセージ例
```
[Day 4] Pandas基礎② - データクリーニングを追加

- pandas_data_cleaning_basic_simple.ipynb（Excelユーザー向け）を作成
- pandas_data_cleaning_basic.ipynb（通常版）を作成  
- README.mdに学習内容の詳細を記載
- 進捗状況を更新
```

### 注意事項
- 各Dayの学習が完了したらコミット
- 大きな変更は細かくコミットを分ける
- `.pyc`、`__pycache__`、`.ipynb_checkpoints`はコミットしない

### 投稿例

```
#100DaysOfMLChallenge Day 3 完了！ 🐼

📅 Day 3: Pandas基礎① - DataFrame操作（8/15）

今日の学習内容：
- DataFrameの作成・選択・フィルタリング
- 統計情報の取得とグループ集計
- CSV操作と欠損値処理

💡 学習のポイント：
最初の練習問題が思いの外難しかったので、Excelユーザー向けのシンプル版を作成！
ExcelでできることはPandasでも同じようにできる！

📊 進捗: 3/100日 (3%)
🔗 https://github.com/miyakawa2449/100days-of-ml-ai

#機械学習 #Python #Pandas #データ分析 #Excelユーザー向け #100日チャレンジ #プログラミング学習
```