#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計基礎① - 平均・分散・標準偏差（最終修正版）

データ分析の基礎となる記述統計について学習します。
これらの統計量は、データの特徴を数値で要約する重要な指標です。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import matplotlib.font_manager as fm
import matplotlib
import os

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

def setup_japanese_font():
    """日本語フォントの強制設定"""
    # matplotlibのキャッシュをクリア
    try:
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
                
                # テスト描画で確認
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '日本語テスト', fontproperties=prop)
                plt.close(fig)
                
                print(f"日本語フォント設定成功: {prop.get_name()}")
                font_found = True
                break
            except Exception as e:
                continue
    
    if not font_found:
        print("日本語フォントが利用できません。英語表記を使用します。")
        # より確実な英語フォント設定
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    return font_found

def create_japanese_text_plot(text, filename):
    """日本語テキストを画像として保存"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, text, fontsize=14, ha='center', va='center', 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("統計基礎① - 平均・分散・標準偏差")
    print("=" * 60)
    
    # 日本語フォントの設定
    use_japanese = setup_japanese_font()
    
    # 見やすい表示設定
    plt.style.use('default')  # seabornではなくdefaultを使用
    
    # カラーパレットを手動設定
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # ===============================
    # 1. 中心傾向の測度
    # ===============================
    print("\n1. 中心傾向の測度（Measures of Central Tendency）")
    print("-" * 50)
    
    # サンプルデータの生成
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)  # 平均100、標準偏差15の正規分布
    
    # 平均の計算
    mean_value = np.mean(data)
    print(f"算術平均: {mean_value:.2f}")
    
    # 手動計算での確認
    manual_mean = sum(data) / len(data)
    print(f"手動計算: {manual_mean:.2f}")
    
    # 平均の性質：外れ値の影響
    data_with_outlier = np.append(data, [500, 600])  # 外れ値を追加
    
    mean_original = np.mean(data)
    mean_with_outlier = np.mean(data_with_outlier)
    
    print(f"\n外れ値の影響:")
    print(f"元のデータの平均: {mean_original:.2f}")
    print(f"外れ値ありの平均: {mean_with_outlier:.2f}")
    print(f"差: {mean_with_outlier - mean_original:.2f}")
    
    # 中央値の計算
    median_original = np.median(data)
    median_with_outlier = np.median(data_with_outlier)
    
    print(f"\n中央値の比較:")
    print(f"元のデータの中央値: {median_original:.2f}")
    print(f"外れ値ありの中央値: {median_with_outlier:.2f}")
    print(f"差: {median_with_outlier - median_original:.2f}")
    print("→ 中央値は外れ値の影響を受けにくい")
    
    # 離散データでの最頻値
    discrete_data = np.random.randint(1, 10, 100)
    mode_result = stats.mode(discrete_data, keepdims=True)
    print(f"\n最頻値の例:")
    print(f"データ: {discrete_data[:20]}...")
    print(f"最頻値: {mode_result.mode[0]}")
    print(f"出現回数: {mode_result.count[0]}")
    
    # 3つの代表値の比較をグラフ化（英語ラベルで確実に表示）
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 正規分布の場合
    axes[0].hist(data, bins=50, density=True, alpha=0.7, color=colors[0], edgecolor='black')
    axes[0].axvline(mean_original, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_original:.1f}')
    axes[0].axvline(median_original, color='green', linestyle='--', linewidth=2, label=f'Median: {median_original:.1f}')
    axes[0].set_title('Normal Distribution - Central Tendency')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 歪んだ分布の場合
    skewed_data = np.random.exponential(50, 1000)
    axes[1].hist(skewed_data, bins=50, density=True, alpha=0.7, color=colors[2], edgecolor='black')
    axes[1].axvline(np.mean(skewed_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(skewed_data):.1f}')
    axes[1].axvline(np.median(skewed_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(skewed_data):.1f}')
    axes[1].set_title('Skewed Distribution - Central Tendency')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n図の説明：")
    print("左: 正規分布では平均と中央値がほぼ同じ")
    print("右: 歪んだ分布では平均が中央値より大きくなる")
    
    # ===============================
    # 2. 散布度の測度
    # ===============================
    print("\n\n2. 散布度の測度（Measures of Dispersion）")
    print("-" * 50)
    
    # 基本的な散布度の指標
    data_range = np.max(data) - np.min(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    print(f"最小値: {np.min(data):.2f}")
    print(f"最大値: {np.max(data):.2f}")
    print(f"範囲: {data_range:.2f}")
    print(f"\n第1四分位数 (Q1): {q1:.2f}")
    print(f"第3四分位数 (Q3): {q3:.2f}")
    print(f"四分位範囲 (IQR): {iqr:.2f}")
    
    # 箱ひげ図で四分位数を視覚化
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # データを準備
    box_data = [data, data_with_outlier]
    tick_labels = ['Original Data', 'With Outliers']
    
    # 箱ひげ図（修正されたパラメータ名を使用）
    bp = ax.boxplot(box_data, tick_labels=tick_labels, patch_artist=True)
    
    # 色を設定
    box_colors = [colors[0], colors[3]]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Box Plot Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    plt.show()
    
    print("図の説明：元のデータ vs 外れ値ありデータの分布比較")
    
    # 分散の計算過程を詳しく見る
    print(f"\n分散の計算過程:")
    sample_data = np.array([10, 20, 30, 40, 50])
    mean_sample = np.mean(sample_data)
    
    print(f"データ: {sample_data}")
    print(f"平均: {mean_sample}")
    print(f"\n各ステップ:")
    
    # 偏差
    deviations = sample_data - mean_sample
    print(f"1. 偏差 (データ - 平均): {deviations}")
    
    # 偏差の2乗
    squared_deviations = deviations ** 2
    print(f"2. 偏差の2乗: {squared_deviations}")
    
    # 分散
    variance = np.mean(squared_deviations)
    print(f"3. 分散 (偏差の2乗の平均): {variance}")
    
    # NumPyの関数で確認
    print(f"\nnp.var()での計算: {np.var(sample_data)}")
    
    # 標本分散と不偏分散
    n = len(sample_data)
    
    # 標本分散（母集団分散）
    population_var = np.var(sample_data, ddof=0)
    print(f"\n標本分散 (n で割る): {population_var}")
    
    # 不偏分散
    sample_var = np.var(sample_data, ddof=1)
    print(f"不偏分散 (n-1 で割る): {sample_var}")
    
    # 手動計算での確認
    manual_unbiased = np.sum(squared_deviations) / (n - 1)
    print(f"\n手動計算の不偏分散: {manual_unbiased}")
    
    # 標準偏差の計算
    std_population = np.std(data, ddof=0)
    std_sample = np.std(data, ddof=1)
    
    print(f"\n標準偏差:")
    print(f"標本標準偏差: {std_population:.2f}")
    print(f"不偏標準偏差: {std_sample:.2f}")
    print(f"\n分散との関係:")
    print(f"√(標本分散) = √{np.var(data, ddof=0):.2f} = {np.sqrt(np.var(data, ddof=0)):.2f}")
    
    # 標準偏差の意味：68-95-99.7ルール
    mean_data = np.mean(data)
    std_data = np.std(data)
    
    # 各範囲に含まれるデータの割合
    within_1std = np.sum((data >= mean_data - std_data) & (data <= mean_data + std_data)) / len(data)
    within_2std = np.sum((data >= mean_data - 2*std_data) & (data <= mean_data + 2*std_data)) / len(data)
    within_3std = np.sum((data >= mean_data - 3*std_data) & (data <= mean_data + 3*std_data)) / len(data)
    
    print(f"\n正規分布における標準偏差の範囲:")
    print(f"平均 ± 1σ: {within_1std*100:.1f}% (理論値: 68.3%)")
    print(f"平均 ± 2σ: {within_2std*100:.1f}% (理論値: 95.4%)")
    print(f"平均 ± 3σ: {within_3std*100:.1f}% (理論値: 99.7%)")
    
    # 標準偏差の視覚化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # ヒストグラムと正規分布曲線
    n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.7, color=colors[0], edgecolor='black')
    
    # 理論的な正規分布
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean_data, std_data), 'r-', linewidth=2, label='Normal Distribution')
    
    # 標準偏差の範囲を表示
    range_colors = ['green', 'orange', 'red']
    alphas = [0.3, 0.2, 0.1]
    std_labels = ['±1σ', '±2σ', '±3σ']
    
    for i, (color, alpha, label) in enumerate(zip(range_colors, alphas, std_labels)):
        ax.axvspan(mean_data - (i+1)*std_data, mean_data + (i+1)*std_data, 
                   alpha=alpha, color=color, label=label)
    
    ax.axvline(mean_data, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_data:.1f}')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Normal Distribution and Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    print("図の説明：正規分布と標準偏差の範囲（68-95-99.7ルール）")
    
    # ===============================
    # 3. 変動係数
    # ===============================
    print("\n\n3. 変動係数（Coefficient of Variation）")
    print("-" * 50)
    
    # 異なる尺度のデータを比較
    # 身長データ（cm）
    heights = np.random.normal(170, 10, 100)
    # 体重データ（kg）
    weights = np.random.normal(65, 8, 100)
    # 年収データ（万円）
    incomes = np.random.normal(500, 100, 100)
    
    # 変動係数の計算
    cv_height = np.std(heights) / np.mean(heights) * 100
    cv_weight = np.std(weights) / np.mean(weights) * 100
    cv_income = np.std(incomes) / np.mean(incomes) * 100
    
    print("各データの統計量:")
    print(f"身長 - 平均: {np.mean(heights):.1f}cm, 標準偏差: {np.std(heights):.1f}cm, CV: {cv_height:.1f}%")
    print(f"体重 - 平均: {np.mean(weights):.1f}kg, 標準偏差: {np.std(weights):.1f}kg, CV: {cv_weight:.1f}%")
    print(f"年収 - 平均: {np.mean(incomes):.1f}万円, 標準偏差: {np.std(incomes):.1f}万円, CV: {cv_income:.1f}%")
    print("\n→ 変動係数で比較すると、年収が最もばらつきが大きい")
    
    # ===============================
    # 4. 品質管理への応用
    # ===============================
    print("\n\n4. 実践的な応用例")
    print("-" * 50)
    print("4.1 品質管理への応用")
    
    # 製造工程のデータシミュレーション
    np.random.seed(42)
    
    # 目標値: 100mm、許容誤差: ±3mm
    target = 100
    tolerance = 3
    
    # 2つの製造ラインのデータ
    line_A = np.random.normal(100, 0.8, 500)  # より精密
    line_B = np.random.normal(100, 1.5, 500)  # ややばらつきが大きい
    
    # 統計量の計算
    stats_A = {
        'Mean': np.mean(line_A),
        'Std Dev': np.std(line_A),
        'Min': np.min(line_A),
        'Max': np.max(line_A),
        'In Spec %': np.sum((line_A >= target - tolerance) & (line_A <= target + tolerance)) / len(line_A) * 100
    }
    
    stats_B = {
        'Mean': np.mean(line_B),
        'Std Dev': np.std(line_B),
        'Min': np.min(line_B),
        'Max': np.max(line_B),
        'In Spec %': np.sum((line_B >= target - tolerance) & (line_B <= target + tolerance)) / len(line_B) * 100
    }
    
    # 結果の表示
    df_stats = pd.DataFrame([stats_A, stats_B], index=['Line A', 'Line B'])
    print("\n製造ライン比較:")
    print(df_stats.round(2))
    
    # 品質管理チャート
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    line_labels = ['Line A', 'Line B']
    
    # ヒストグラム比較
    axes[0, 0].hist(line_A, bins=30, alpha=0.7, color=colors[1], label=line_labels[0], density=True)
    axes[0, 0].hist(line_B, bins=30, alpha=0.7, color=colors[3], label=line_labels[1], density=True)
    axes[0, 0].axvline(target, color='black', linestyle='--', label='Target')
    axes[0, 0].axvspan(target - tolerance, target + tolerance, alpha=0.2, color='green', label='Tolerance Range')
    axes[0, 0].set_xlabel('Measurement (mm)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Measurements')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 管理図（時系列）
    sample_size = 100
    axes[0, 1].plot(line_A[:sample_size], color=colors[1], alpha=0.7, label=line_labels[0])
    axes[0, 1].plot(line_B[:sample_size], color=colors[3], alpha=0.7, label=line_labels[1])
    axes[0, 1].axhline(target, color='black', linestyle='--', label='Target')
    axes[0, 1].axhline(target + tolerance, color='green', linestyle=':', label='Upper Limit')
    axes[0, 1].axhline(target - tolerance, color='green', linestyle=':', label='Lower Limit')
    axes[0, 1].set_xlabel('Sample Number')
    axes[0, 1].set_ylabel('Measurement (mm)')
    axes[0, 1].set_title('Control Chart')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 箱ひげ図
    bp = axes[1, 0].boxplot([line_A, line_B], tick_labels=line_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor(colors[1])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(colors[3])
    bp['boxes'][1].set_alpha(0.7)
    axes[1, 0].axhline(target, color='black', linestyle='--', label='Target')
    axes[1, 0].set_ylabel('Measurement (mm)')
    axes[1, 0].set_title('Box Plot Comparison')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 工程能力指数
    # Cp = (USL - LSL) / (6 * σ)
    cp_A = (2 * tolerance) / (6 * np.std(line_A))
    cp_B = (2 * tolerance) / (6 * np.std(line_B))
    
    axes[1, 1].bar(line_labels, [cp_A, cp_B], color=[colors[1], colors[3]], alpha=0.7)
    axes[1, 1].axhline(1.33, color='green', linestyle='--', label='Target Cp (1.33)')
    axes[1, 1].set_ylabel('Cp Value')
    axes[1, 1].set_title('Process Capability Index (Cp)')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, (label, value) in enumerate(zip(line_labels, [cp_A, cp_B])):
        axes[1, 1].text(i, value + 0.05, f'{value:.2f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print("図の説明：製造ライン A vs B の品質管理分析")
    print(f"- ライン A: より精密（Cp = {cp_A:.2f}）")
    print(f"- ライン B: ばらつき大（Cp = {cp_B:.2f}）")
    
    # ===============================
    # 5. 金融リスク管理への応用
    # ===============================
    print("\n4.2 金融リスク管理への応用")
    
    # 株式リターンのシミュレーション
    np.random.seed(42)
    
    # 3つの異なるリスク特性を持つ資産
    days = 252  # 1年間の営業日
    
    # 低リスク資産（債券的）
    low_risk = np.random.normal(0.0002, 0.005, days)
    # 中リスク資産（バランス型）
    medium_risk = np.random.normal(0.0003, 0.01, days)
    # 高リスク資産（成長株）
    high_risk = np.random.normal(0.0005, 0.02, days)
    
    # 累積リターン
    cum_low = (1 + low_risk).cumprod()
    cum_medium = (1 + medium_risk).cumprod()
    cum_high = (1 + high_risk).cumprod()
    
    # 統計量の計算
    results = []
    asset_names = ['Low Risk', 'Medium Risk', 'High Risk']
    returns_list = [low_risk, medium_risk, high_risk]
    cumulative_list = [cum_low, cum_medium, cum_high]
    
    for name, returns, cumulative in zip(asset_names, returns_list, cumulative_list):
        annual_return = (cumulative[-1] - 1) * 100
        annual_std = np.std(returns) * np.sqrt(252) * 100
        sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        
        results.append({
            'Asset': name,
            'Annual Return(%)': annual_return,
            'Annual Std(%)': annual_std,
            'Sharpe Ratio': sharpe_ratio
        })
    
    df_results = pd.DataFrame(results)
    print("\nリスク・リターン分析:")
    print(df_results.round(2))
    
    # リスク・リターンの視覚化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    risk_colors = [colors[2], colors[1], colors[3]]
    
    # 累積リターン
    axes[0, 0].plot(cum_low, label=asset_names[0], linewidth=2, color=risk_colors[0])
    axes[0, 0].plot(cum_medium, label=asset_names[1], linewidth=2, color=risk_colors[1])
    axes[0, 0].plot(cum_high, label=asset_names[2], linewidth=2, color=risk_colors[2])
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Cumulative Returns')
    axes[0, 0].set_title('1-Year Cumulative Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # リターンの分布
    axes[0, 1].hist(low_risk * 100, bins=30, alpha=0.5, label=asset_names[0], density=True, color=risk_colors[0])
    axes[0, 1].hist(medium_risk * 100, bins=30, alpha=0.5, label=asset_names[1], density=True, color=risk_colors[1])
    axes[0, 1].hist(high_risk * 100, bins=30, alpha=0.5, label=asset_names[2], density=True, color=risk_colors[2])
    axes[0, 1].set_xlabel('Daily Returns (%)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Daily Returns Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # リスク・リターン散布図
    axes[1, 0].scatter(df_results['Annual Std(%)'], df_results['Annual Return(%)'], s=200, c=risk_colors)
    for idx, row in df_results.iterrows():
        axes[1, 0].annotate(row['Asset'], 
                            (row['Annual Std(%)'], row['Annual Return(%)']),
                            xytext=(5, 5), textcoords='offset points')
    axes[1, 0].set_xlabel('Risk (Annual Std %)')
    axes[1, 0].set_ylabel('Return (Annual %)')
    axes[1, 0].set_title('Risk-Return Analysis')
    axes[1, 0].grid(True, alpha=0.3)
    
    # VaR（Value at Risk）分析
    confidence_level = 0.95
    var_low = np.percentile(low_risk, (1 - confidence_level) * 100) * 100
    var_medium = np.percentile(medium_risk, (1 - confidence_level) * 100) * 100
    var_high = np.percentile(high_risk, (1 - confidence_level) * 100) * 100
    
    axes[1, 1].bar(asset_names, 
                   [-var_low, -var_medium, -var_high],
                   color=risk_colors, alpha=0.7)
    axes[1, 1].set_ylabel('VaR (%)')
    axes[1, 1].set_title(f'{confidence_level*100}% VaR (1-Day)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, (label, value) in enumerate(zip(asset_names, [-var_low, -var_medium, -var_high])):
        axes[1, 1].text(i, value + 0.1, f'{value:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print("図の説明：3つの資産のリスク・リターン特性比較")
    print(f"- 低リスク: 安定した収益、低ボラティリティ")
    print(f"- 中リスク: バランス型、中程度のリスク・リターン")
    print(f"- 高リスク: 高収益期待、高ボラティリティ")
    
    # ===============================
    # まとめ
    # ===============================
    print("\n\n" + "=" * 60)
    print("まとめ")
    print("=" * 60)
    print("""
今日学んだ統計の基礎概念：

1. 中心傾向の測度
   - 平均（Mean）: データの重心
   - 中央値（Median）: 順序の中央
   - 最頻値（Mode）: 最も頻繁な値

2. 散布度の測度
   - 範囲（Range）: 最大値 - 最小値
   - 四分位範囲（IQR）: Q3 - Q1
   - 分散（Variance）: 偏差の2乗の平均
   - 標準偏差（Standard Deviation）: 分散の平方根

3. 応用
   - 品質管理: 工程能力指数、管理図
   - 金融分析: リスク評価、VaR
   - 異常検知: 3σルール、IQRルール

これらの統計量は、データの特徴を定量的に理解し、
意思決定を行うための重要な基礎となります。
    """)

if __name__ == "__main__":
    main()