#!/usr/bin/env python3
import pandas as pd
import numpy as np


def dataframe_creation_examples():
    print("=== 1. DataFrameの作成方法 ===\n")
    
    # 辞書から作成
    data_dict = {
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 28],
        'city': ['Tokyo', 'Osaka', 'Kyoto', 'Nagoya'],
        'salary': [50000, 60000, 55000, 52000]
    }
    df_from_dict = pd.DataFrame(data_dict)
    print("辞書から作成したDataFrame:")
    print(df_from_dict)
    print()
    
    # リストから作成
    data_list = [
        ['Alice', 25, 'Tokyo', 50000],
        ['Bob', 30, 'Osaka', 60000],
        ['Charlie', 35, 'Kyoto', 55000],
        ['David', 28, 'Nagoya', 52000]
    ]
    columns = ['name', 'age', 'city', 'salary']
    df_from_list = pd.DataFrame(data_list, columns=columns)
    print("リストから作成したDataFrame:")
    print(df_from_list)
    print()
    
    # NumPy配列から作成
    np_array = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    df_from_numpy = pd.DataFrame(np_array, columns=['A', 'B', 'C'])
    print("NumPy配列から作成したDataFrame:")
    print(df_from_numpy)
    print()
    
    return df_from_dict


def dataframe_basic_operations(df):
    print("=== 2. DataFrameの基本操作 ===\n")
    
    # 列の選択
    print("単一列の選択 (df['name']):")
    print(df['name'])
    print()
    
    print("複数列の選択 (df[['name', 'age']]):")
    print(df[['name', 'age']])
    print()
    
    # 行の選択
    print("インデックスによる行選択 (df.iloc[0]):")
    print(df.iloc[0])
    print()
    
    print("スライスによる行選択 (df.iloc[1:3]):")
    print(df.iloc[1:3])
    print()
    
    # 条件によるフィルタリング
    print("年齢が30以上の行をフィルタリング:")
    filtered_df = df[df['age'] >= 30]
    print(filtered_df)
    print()
    
    print("複数条件でのフィルタリング (年齢30以上かつ給与55000以上):")
    multi_filtered = df[(df['age'] >= 30) & (df['salary'] >= 55000)]
    print(multi_filtered)
    print()
    
    # ソート
    print("年齢で昇順ソート:")
    sorted_by_age = df.sort_values('age')
    print(sorted_by_age)
    print()
    
    print("給与で降順ソート:")
    sorted_by_salary = df.sort_values('salary', ascending=False)
    print(sorted_by_salary)
    print()


def dataframe_statistics(df):
    print("=== 3. DataFrameの統計情報 ===\n")
    
    # 基本情報
    print("DataFrameの基本情報 (df.info()):")
    df.info()
    print()
    
    # 統計量のサマリー
    print("数値列の統計サマリー (df.describe()):")
    print(df.describe())
    print()
    
    # 個別の統計量
    print("平均値:")
    print(df[['age', 'salary']].mean())
    print()
    
    print("中央値:")
    print(df[['age', 'salary']].median())
    print()
    
    print("標準偏差:")
    print(df[['age', 'salary']].std())
    print()
    
    # グループ集計
    print("都市ごとの平均給与:")
    city_salary = df.groupby('city')['salary'].mean()
    print(city_salary)
    print()


def csv_operations():
    print("=== 4. CSVファイルの読み書き ===\n")
    
    # サンプルデータの作成
    sample_data = pd.DataFrame({
        'product': ['Apple', 'Banana', 'Orange', 'Grape', 'Mango'],
        'price': [100, 50, 80, 200, 150],
        'quantity': [10, 20, 15, 5, 8]
    })
    
    # CSVファイルへの書き込み
    csv_filename = 'day3/sample_data.csv'
    sample_data.to_csv(csv_filename, index=False)
    print(f"CSVファイルを作成しました: {csv_filename}")
    print("書き込んだデータ:")
    print(sample_data)
    print()
    
    # CSVファイルの読み込み
    loaded_data = pd.read_csv(csv_filename)
    print("CSVファイルから読み込んだデータ:")
    print(loaded_data)
    print()
    
    # 読み込みオプションの例
    print("特定の列のみ読み込み (usecols=['product', 'price']):")
    partial_data = pd.read_csv(csv_filename, usecols=['product', 'price'])
    print(partial_data)
    print()


def advanced_operations():
    print("=== 5. その他の便利な操作 ===\n")
    
    # サンプルデータ
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'x']
    })
    
    # 新しい列の追加
    df['D'] = df['A'] + df['B']
    print("新しい列を追加 (D = A + B):")
    print(df)
    print()
    
    # 列の削除
    df_dropped = df.drop('C', axis=1)
    print("列Cを削除:")
    print(df_dropped)
    print()
    
    # 欠損値の処理例
    df_with_nan = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': [5, np.nan, np.nan, 8],
        'col3': [9, 10, 11, 12]
    })
    print("欠損値を含むDataFrame:")
    print(df_with_nan)
    print()
    
    print("欠損値の確認 (isnull().sum()):")
    print(df_with_nan.isnull().sum())
    print()
    
    print("欠損値を0で埋める (fillna(0)):")
    print(df_with_nan.fillna(0))
    print()


if __name__ == "__main__":
    print("Day 3: Pandas基礎① - DataFrame操作\n")
    print("=" * 50)
    
    # 1. DataFrameの作成
    df = dataframe_creation_examples()
    
    print("\n" + "=" * 50 + "\n")
    
    # 2. 基本操作
    dataframe_basic_operations(df)
    
    print("\n" + "=" * 50 + "\n")
    
    # 3. 統計情報
    dataframe_statistics(df)
    
    print("\n" + "=" * 50 + "\n")
    
    # 4. CSV操作
    csv_operations()
    
    print("\n" + "=" * 50 + "\n")
    
    # 5. その他の操作
    advanced_operations()
    
    print("\n学習完了！")