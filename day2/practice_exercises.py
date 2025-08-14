#!/usr/bin/env python3
"""
Day 2: NumPy実践問題集
配列操作と行列演算の練習問題
"""

import numpy as np

def exercise_1_matrix_operations():
    """
    演習1: 行列操作の練習
    """
    print("=== 演習1: 行列操作 ===")
    
    # 問題1: 3x3の魔方陣を作成（各行・各列・対角線の和が15）
    magic_square = np.array([[2, 7, 6],
                             [9, 5, 1],
                             [4, 3, 8]])
    
    print("魔方陣:")
    print(magic_square)
    print(f"各行の和: {np.sum(magic_square, axis=1)}")
    print(f"各列の和: {np.sum(magic_square, axis=0)}")
    print(f"対角線の和: {np.trace(magic_square)}")
    print(f"逆対角線の和: {np.sum(np.fliplr(magic_square).diagonal())}")
    print()

def exercise_2_broadcasting():
    """
    演習2: ブロードキャスティングを使った計算
    """
    print("=== 演習2: ブロードキャスティング ===")
    
    # 学生の点数データ（5人×3科目）
    scores = np.array([[85, 90, 78],
                       [92, 88, 95],
                       [78, 85, 80],
                       [88, 92, 90],
                       [95, 85, 88]])
    
    subjects = ['数学', '英語', '理科']
    students = ['学生A', '学生B', '学生C', '学生D', '学生E']
    
    print("点数データ:")
    print(scores)
    print()
    
    # 各科目の平均点
    subject_mean = np.mean(scores, axis=0)
    print("各科目の平均点:")
    for i, subj in enumerate(subjects):
        print(f"  {subj}: {subject_mean[i]:.1f}")
    
    # 各学生の平均点
    student_mean = np.mean(scores, axis=1)
    print("\n各学生の平均点:")
    for i, student in enumerate(students):
        print(f"  {student}: {student_mean[i]:.1f}")
    
    # 偏差値の計算（平均50、標準偏差10に正規化）
    subject_std = np.std(scores, axis=0)
    deviation_scores = 50 + 10 * (scores - subject_mean) / subject_std
    
    print("\n偏差値:")
    print(deviation_scores.astype(int))
    print()

def exercise_3_image_processing():
    """
    演習3: 画像処理のシミュレーション
    """
    print("=== 演習3: 画像処理シミュレーション ===")
    
    # 8x8のグレースケール画像をシミュレート
    np.random.seed(42)
    image = np.random.randint(0, 256, size=(8, 8))
    
    print("元の画像（8x8）:")
    print(image)
    
    # カーネルフィルタ（エッジ検出）
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    
    # 簡単な畳み込み演算（パディングなし）
    output_size = image.shape[0] - kernel.shape[0] + 1
    filtered = np.zeros((output_size, output_size))
    
    for i in range(output_size):
        for j in range(output_size):
            region = image[i:i+3, j:j+3]
            filtered[i, j] = np.sum(region * kernel)
    
    print("\nエッジ検出後（6x6）:")
    print(filtered.astype(int))
    print()

def exercise_4_statistical_analysis():
    """
    演習4: 統計分析
    """
    print("=== 演習4: 統計分析 ===")
    
    # 2つのグループのデータ
    np.random.seed(42)
    group_a = np.random.normal(100, 15, 50)  # 平均100、標準偏差15
    group_b = np.random.normal(110, 12, 50)  # 平均110、標準偏差12
    
    print("グループA:")
    print(f"  平均: {np.mean(group_a):.2f}")
    print(f"  標準偏差: {np.std(group_a):.2f}")
    print(f"  中央値: {np.median(group_a):.2f}")
    
    print("\nグループB:")
    print(f"  平均: {np.mean(group_b):.2f}")
    print(f"  標準偏差: {np.std(group_b):.2f}")
    print(f"  中央値: {np.median(group_b):.2f}")
    
    # t検定の統計量（簡易版）
    pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
    t_stat = (np.mean(group_b) - np.mean(group_a)) / (pooled_std * np.sqrt(2/50))
    print(f"\nt統計量: {t_stat:.3f}")
    print()

def exercise_5_linear_algebra():
    """
    演習5: 線形代数の応用
    """
    print("=== 演習5: 線形代数の応用 ===")
    
    # 連立方程式を解く
    # 2x + 3y = 7
    # 4x - y = 1
    A = np.array([[2, 3],
                  [4, -1]])
    b = np.array([7, 1])
    
    print("連立方程式:")
    print("2x + 3y = 7")
    print("4x - y = 1")
    
    # 逆行列を使った解法
    x = np.linalg.inv(A) @ b
    print(f"\n解: x = {x[0]:.2f}, y = {x[1]:.2f}")
    
    # 検証
    print(f"検証: 2×{x[0]:.2f} + 3×{x[1]:.2f} = {2*x[0] + 3*x[1]:.2f}")
    print(f"      4×{x[0]:.2f} - {x[1]:.2f} = {4*x[0] - x[1]:.2f}")
    
    # 固有値と固有ベクトル
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\n固有値: {eigenvalues}")
    print(f"固有ベクトル:\n{eigenvectors}")
    print()

def main():
    """メイン実行関数"""
    exercises = [
        exercise_1_matrix_operations,
        exercise_2_broadcasting,
        exercise_3_image_processing,
        exercise_4_statistical_analysis,
        exercise_5_linear_algebra
    ]
    
    for exercise in exercises:
        exercise()
        print("-" * 50)
        print()

if __name__ == "__main__":
    main()