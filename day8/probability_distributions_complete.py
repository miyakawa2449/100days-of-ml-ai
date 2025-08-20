#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計基礎② - 確率・分布（完全版）

確率論と確率分布の理論的背景から実践的応用まで詳しく学習します。
日本語フォント問題を解決した完全版です。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import factorial
import warnings
import itertools
from collections import Counter
import matplotlib.font_manager as fm
import os

# 警告を抑制
warnings.filterwarnings('ignore')

def setup_japanese_font():
    """日本語フォントの設定"""
    # matplotlibのキャッシュをクリア
    try:
        import matplotlib
        matplotlib.font_manager._rebuild()
    except:
        pass
    
    # macOSで確実に動作する日本語フォントパスを直接指定
    japanese_font_paths = [
        '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc', 
        '/Library/Fonts/ヒラギノ角ゴ ProN W3.otf',
        '/System/Library/Fonts/Arial Unicode.ttf',
        '/Library/Fonts/Arial Unicode.ttf'
    ]
    
    font_found = False
    for font_path in japanese_font_paths:
        if os.path.exists(font_path):
            try:
                # フォントを直接登録
                fm.fontManager.addfont(font_path)
                # フォントプロパティを設定
                prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = prop.get_name()
                plt.rcParams['font.sans-serif'] = [prop.get_name()]
                print(f"日本語フォント設定成功: {prop.get_name()}")
                font_found = True
                break
            except Exception as e:
                continue
    
    if not font_found:
        print("日本語フォントが利用できません。英語表記を使用します。")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    return font_found

def main():
    print("=" * 60)
    print("統計基礎② - 確率・分布")
    print("=" * 60)
    
    # 日本語フォント設定
    use_japanese = setup_japanese_font()
    
    # スタイル設定
    plt.style.use('default')
    np.random.seed(42)
    
    # ===============================
    # 1. 確率の基本
    # ===============================
    print("\n1. 確率の基本")
    print("-" * 20)
    
    # サイコロシミュレーション
    dice_rolls = np.random.randint(1, 7, size=1000)
    unique, counts = np.unique(dice_rolls, return_counts=True)
    
    print("🎲 サイコロ1000回の結果")
    for face, count in zip(unique, counts):
        prob = count / 1000
        print(f"{face}: {count}回 ({prob:.3f})")
    print(f"理論値: 各面 {1/6:.3f}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, counts, color='skyblue', alpha=0.7, edgecolor='navy')
    plt.axhline(y=1000/6, color='red', linestyle='--', linewidth=2, 
                label=f'理論値 ({1000/6:.0f}回)' if use_japanese else f'Theoretical ({1000/6:.0f})')
    
    # 各棒に数値を表示
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 str(count), ha='center', va='bottom')
    
    if use_japanese:
        plt.xlabel('サイコロの目')
        plt.ylabel('出現回数')
        plt.title('サイコロ1000回の結果')
    else:
        plt.xlabel('Dice Face')
        plt.ylabel('Frequency')
        plt.title('Results of 1000 Dice Rolls')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # ===============================
    # 2. ベイズの定理
    # ===============================
    print("\n2. ベイズの定理：医療診断の例")
    print("-" * 35)
    
    # パラメータ
    prior_disease = 0.001  # 有病率 0.1%
    sensitivity = 0.99     # 感度 99%
    specificity = 0.95     # 特異度 95%
    
    false_positive_rate = 1 - specificity
    prob_positive = (sensitivity * prior_disease + 
                    false_positive_rate * (1 - prior_disease))
    posterior_disease = (sensitivity * prior_disease) / prob_positive
    
    print(f"事前確率（有病率）: {prior_disease*100:.3f}%")
    print(f"検査の感度: {sensitivity*100:.1f}%")
    print(f"検査の特異度: {specificity*100:.1f}%")
    print(f"検査陽性の確率: {prob_positive*100:.3f}%")
    print(f"陽性時の病気の確率: {posterior_disease*100:.2f}%")
    print(f"→ 検査が陽性でも、実際に病気の確率は{posterior_disease*100:.1f}%程度")
    
    # ===============================
    # 3. 離散確率分布
    # ===============================
    print("\n3. 離散確率分布")
    print("-" * 20)
    
    # 二項分布の例
    n_shots = 10
    p_success = 0.7
    successes = np.random.binomial(n_shots, p_success, size=1000)
    
    print(f"🏀 バスケの自由投げ (n={n_shots}, p={p_success})")
    print(f"平均成功回数: {np.mean(successes):.1f}")
    print(f"理論値: {n_shots * p_success}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 二項分布
    unique_succ, counts_succ = np.unique(successes, return_counts=True)
    probabilities = counts_succ / len(successes)
    x_theory = np.arange(0, n_shots + 1)
    y_theory = stats.binom.pmf(x_theory, n_shots, p_success)
    
    axes[0, 0].bar(unique_succ, probabilities, alpha=0.7, color='lightgreen', 
                   label='シミュレーション' if use_japanese else 'Simulation')
    axes[0, 0].plot(x_theory, y_theory, 'ro-', label='理論値' if use_japanese else 'Theoretical', linewidth=2)
    
    if use_japanese:
        axes[0, 0].set_xlabel('成功回数')
        axes[0, 0].set_ylabel('確率')
        axes[0, 0].set_title(f'二項分布 B({n_shots}, {p_success})')
    else:
        axes[0, 0].set_xlabel('Number of Successes')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].set_title(f'Binomial Distribution B({n_shots}, {p_success})')
    
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # ポアソン分布
    lambda_customers = 3
    customers_per_hour = np.random.poisson(lambda_customers, size=100)
    
    axes[0, 1].hist(customers_per_hour, bins=range(0, 12), density=True, alpha=0.7, 
                    color='orange', edgecolor='black', label='シミュレーション' if use_japanese else 'Simulation')
    
    x_poisson = np.arange(0, 12)
    y_poisson = stats.poisson.pmf(x_poisson, lambda_customers)
    axes[0, 1].plot(x_poisson, y_poisson, 'ro-', label='理論値' if use_japanese else 'Theoretical', linewidth=2)
    
    if use_japanese:
        axes[0, 1].set_xlabel('1時間あたりの来客数')
        axes[0, 1].set_ylabel('確率密度')
        axes[0, 1].set_title(f'ポアソン分布 (λ={lambda_customers})')
    else:
        axes[0, 1].set_xlabel('Customers per Hour')
        axes[0, 1].set_ylabel('Probability Density')
        axes[0, 1].set_title(f'Poisson Distribution (λ={lambda_customers})')
    
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # ===============================
    # 4. 連続確率分布
    # ===============================
    print("\n4. 連続確率分布")
    print("-" * 20)
    
    # 正規分布（身長の例）
    height_mean = 172
    height_std = 6
    heights = np.random.normal(height_mean, height_std, size=1000)
    
    print(f"📏 身長分布 N({height_mean}, {height_std}²)")
    print(f"平均身長: {np.mean(heights):.1f}cm")
    print(f"標準偏差: {np.std(heights):.1f}cm")
    
    # 正規分布の可視化
    axes[1, 0].hist(heights, bins=30, density=True, alpha=0.7, color='lightblue', 
                    edgecolor='black', label='シミュレーションデータ' if use_japanese else 'Simulated Data')
    
    x_norm = np.linspace(heights.min(), heights.max(), 100)
    y_norm = stats.norm.pdf(x_norm, height_mean, height_std)
    axes[1, 0].plot(x_norm, y_norm, 'r-', linewidth=2, label='理論値（正規分布）' if use_japanese else 'Normal Distribution')
    
    axes[1, 0].axvline(height_mean, color='red', linestyle='--', alpha=0.8, 
                       label=f'平均 ({height_mean}cm)' if use_japanese else f'Mean ({height_mean}cm)')
    axes[1, 0].axvspan(height_mean - height_std, height_mean + height_std, 
                       alpha=0.2, color='yellow', label='±1標準偏差' if use_japanese else '±1 Std Dev')
    
    if use_japanese:
        axes[1, 0].set_xlabel('身長 (cm)')
        axes[1, 0].set_ylabel('確率密度')
        axes[1, 0].set_title('身長分布（正規分布）')
    else:
        axes[1, 0].set_xlabel('Height (cm)')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].set_title('Height Distribution (Normal)')
    
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 標準正規分布
    x_std_norm = np.linspace(-4, 4, 100)
    y_std_norm = stats.norm.pdf(x_std_norm, 0, 1)
    
    axes[1, 1].plot(x_std_norm, y_std_norm, 'b-', linewidth=2, label='標準正規分布 N(0,1)' if use_japanese else 'Standard Normal N(0,1)')
    axes[1, 1].fill_between(x_std_norm, y_std_norm, alpha=0.3, color='blue')
    
    # 重要なZ値をマーク
    important_z = [-2, -1, 0, 1, 2]
    for z in important_z:
        axes[1, 1].axvline(z, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].text(z, 0.05, f'Z={z}', ha='center', va='bottom')
    
    if use_japanese:
        axes[1, 1].set_xlabel('Z値（標準化された値）')
        axes[1, 1].set_ylabel('確率密度')
        axes[1, 1].set_title('標準正規分布')
    else:
        axes[1, 1].set_xlabel('Z-value (Standardized)')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].set_title('Standard Normal Distribution')
    
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 確率計算
    prob_180_or_more = 1 - stats.norm.cdf(180, height_mean, height_std)
    prob_under_160 = stats.norm.cdf(160, height_mean, height_std)
    height_90th = stats.norm.ppf(0.9, height_mean, height_std)
    
    print(f"\n身長に関する確率:")
    print(f"180cm以上: {prob_180_or_more:.3f} ({prob_180_or_more*100:.1f}%)")
    print(f"160cm未満: {prob_under_160:.3f} ({prob_under_160*100:.1f}%)")
    print(f"上位10%の身長: {height_90th:.1f}cm以上")
    
    # ===============================
    # 5. 中心極限定理の実演
    # ===============================
    print("\n5. 中心極限定理の実演")
    print("-" * 25)
    
    # 指数分布（非正規）からのサンプル平均の分布
    original_dist = stats.expon(scale=2)
    sample_sizes = [1, 5, 10, 30]
    n_experiments = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, n in enumerate(sample_sizes):
        sample_means = []
        for _ in range(n_experiments):
            samples = original_dist.rvs(n)
            sample_means.append(np.mean(samples))
        
        sample_means = np.array(sample_means)
        
        # ヒストグラム
        axes[i].hist(sample_means, bins=30, density=True, alpha=0.7, 
                     edgecolor='black', label=f'サンプル平均 (n={n})' if use_japanese else f'Sample Mean (n={n})')
        
        # 理論的正規分布
        theoretical_mean = original_dist.mean()
        theoretical_std = original_dist.std() / np.sqrt(n)
        
        x_range = np.linspace(sample_means.min(), sample_means.max(), 100)
        y_range = stats.norm.pdf(x_range, theoretical_mean, theoretical_std)
        axes[i].plot(x_range, y_range, 'r-', linewidth=2, label='理論的正規分布' if use_japanese else 'Theoretical Normal')
        
        if use_japanese:
            axes[i].set_title(f'サンプルサイズ n = {n}')
            axes[i].set_xlabel('サンプル平均')
            axes[i].set_ylabel('確率密度')
        else:
            axes[i].set_title(f'Sample Size n = {n}')
            axes[i].set_xlabel('Sample Mean')
            axes[i].set_ylabel('Probability Density')
        
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        
        print(f"n={n}: 平均={np.mean(sample_means):.3f}, 標準偏差={np.std(sample_means):.3f}")
        print(f"    理論値: 平均={theoretical_mean:.3f}, 標準偏差={theoretical_std:.3f}")
    
    if use_japanese:
        plt.suptitle('中心極限定理：指数分布からのサンプル平均の分布', fontsize=16)
    else:
        plt.suptitle('Central Limit Theorem: Sample Means from Exponential Distribution', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # ===============================
    # 6. A/Bテストの例
    # ===============================
    print("\n6. A/Bテストの統計的設計")
    print("-" * 30)
    
    # パラメータ
    p1 = 0.05      # コントロール群のコンバージョン率
    p2 = 0.06      # 治療群のコンバージョン率
    alpha = 0.05   # 有意水準
    beta = 0.20    # タイプIIエラー率
    
    print(f"コントロール群のコンバージョン率: {p1*100}%")
    print(f"治療群のコンバージョン率: {p2*100}%")
    print(f"相対的改善: {(p2-p1)/p1*100:.1f}%")
    
    # 必要サンプルサイズの概算
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(1 - beta)
    
    pooled_p = (p1 + p2) / 2
    n_per_group = ((z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
                    z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1))**2
    
    n_per_group = int(np.ceil(n_per_group))
    
    print(f"\n必要サンプルサイズ:")
    print(f"各群: {n_per_group}人")
    print(f"総計: {n_per_group * 2}人")
    print(f"検出力: {(1-beta)*100}%")
    
    # ===============================
    # 7. VaRの計算例
    # ===============================
    print("\n7. VaR（Value at Risk）の計算")
    print("-" * 35)
    
    # 株価リターンのシミュレーション
    daily_return_mean = 0.0005
    daily_return_std = 0.02
    n_days = 1000
    
    returns_normal = np.random.normal(daily_return_mean, daily_return_std, n_days)
    
    # VaR計算
    confidence_levels = [0.95, 0.99]
    investment_amount = 100_000_000  # 1億円
    
    print(f"投資額: {investment_amount/1_000_000:.0f}百万円")
    print(f"日次リターン: 平均{daily_return_mean*100:.2f}%, 標準偏差{daily_return_std*100:.1f}%")
    
    for confidence in confidence_levels:
        alpha = 1 - confidence
        
        # パラメトリック法
        var_parametric = -stats.norm.ppf(alpha, daily_return_mean, daily_return_std) * investment_amount
        
        # ヒストリカル法
        var_historical = -np.percentile(returns_normal, alpha * 100) * investment_amount
        
        print(f"\n{confidence*100:.0f}% VaR (1日):")
        print(f"  パラメトリック法: {var_parametric/1_000_000:.1f}百万円")
        print(f"  ヒストリカル法: {var_historical/1_000_000:.1f}百万円")
    
    # VaRの可視化
    plt.figure(figsize=(12, 6))
    
    plt.hist(returns_normal * investment_amount / 1_000_000, bins=50, density=True, 
             alpha=0.7, color='lightblue', edgecolor='black', label='日次損益分布' if use_japanese else 'Daily P&L Distribution')
    
    var_95 = -np.percentile(returns_normal, 5) * investment_amount / 1_000_000
    plt.axvline(-var_95, color='red', linestyle='--', linewidth=2, 
                label=f'95% VaR: {var_95:.1f}百万円' if use_japanese else f'95% VaR: {var_95:.1f}M JPY')
    
    if use_japanese:
        plt.xlabel('日次損益 (百万円)')
        plt.ylabel('確率密度')
        plt.title('投資ポートフォリオの損益分布とVaR')
    else:
        plt.xlabel('Daily P&L (Million JPY)')
        plt.ylabel('Probability Density')
        plt.title('Portfolio P&L Distribution and VaR')
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # ===============================
    # まとめ
    # ===============================
    print("\n" + "=" * 60)
    print("まとめ")
    print("=" * 60)
    print("""
今日学んだ確率・分布の概念：

1. 確率論の基礎
   - コルモゴロフの公理と確率の性質
   - ベイズの定理とその応用
   - 条件付き確率

2. 離散確率分布
   - ベルヌーイ分布・二項分布
   - ポアソン分布とポアソン過程
   - 幾何分布

3. 連続確率分布
   - 正規分布と標準正規分布
   - 中心極限定理の重要性
   - その他の分布（χ²、t、F分布）

4. 実践応用
   - A/Bテストの統計的設計
   - VaRによるリスク管理
   - 医療診断でのベイズ統計

これらの確率分布は、機械学習のアルゴリズム、
統計的推論、リスク管理など、
データサイエンスのあらゆる分野で活用されます。
    """)

if __name__ == "__main__":
    main()