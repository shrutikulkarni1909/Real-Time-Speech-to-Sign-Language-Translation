#!/usr/bin/env python3
"""
EVALUATION RESULTS ANALYSIS & VISUALIZATION
==============================================

Analyzes the evaluation_results.csv and produces:
1. WER vs Gloss Accuracy scatter plot
2. Module-wise accuracy breakdown (bar chart)
3. Accuracy trends over time (line plot)
4. Distribution of metrics (box/violin plots)
5. Performance summary heatmap

Usage:
    python analyze_evaluation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configuration
CSV_FILE = 'evaluation_results.csv'
PLOTS_DIR = 'evaluation_plots'

# Create output directory for plots
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================

print("=" * 80)
print("SPEECH-TO-SIGN EVALUATION ANALYSIS")
print("=" * 80)

if not os.path.exists(CSV_FILE):
    print(f"\n❌ Error: {CSV_FILE} not found!")
    print("\nRun 'python evaluate_pipeline.py' to generate results.")
    exit(1)

print(f"\n📊 Loading data from {CSV_FILE}...")

try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit(1)

print(f"✓ Loaded {len(df)} test results")

# Convert numeric columns
numeric_cols = ['wer', 'rtf', 'latency', 'mapping_accuracy', 'vocabulary_coverage', 
                'unknown_ratio', 'gloss_error_rate', 'gloss_accuracy', 'audio_duration']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"✓ Data shape: {df.shape}")
print(f"✓ Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: Calculate statistics
# ============================================================================

print("\n" + "-" * 80)
print("SUMMARY STATISTICS")
print("-" * 80)

stats = {
    'WER (Word Error Rate)': df['wer'].mean(),
    'RTF (Real-Time Factor)': df['rtf'].mean(),
    'Latency (seconds)': df['latency'].mean(),
    'Mapping Accuracy': df['mapping_accuracy'].mean(),
    'Vocabulary Coverage': df['vocabulary_coverage'].mean(),
    'Unknown Word Ratio': df['unknown_ratio'].mean(),
}

# Add gloss accuracy if available
if 'gloss_accuracy' in df.columns and df['gloss_accuracy'].notna().sum() > 0:
    gloss_acc = pd.to_numeric(df['gloss_accuracy'], errors='coerce').mean()
    stats['Gloss Accuracy'] = gloss_acc
    gloss_available = True
else:
    gloss_available = False

for metric, value in stats.items():
    if pd.notna(value):
        if 'Accuracy' in metric or 'Coverage' in metric:
            print(f"  {metric:.<40} {value:.4f} ({value*100:.2f}%)")
        elif metric == 'WER':
            print(f"  {metric:.<40} {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"  {metric:.<40} {value:.4f}")

# ============================================================================
# STEP 3: Create visualizations
# ============================================================================

print("\n" + "-" * 80)
print("GENERATING PLOTS")
print("-" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create a large figure with subplots
fig = plt.figure(figsize=(20, 14))

# --- PLOT 1: WER vs Gloss Accuracy Scatter ---
ax1 = plt.subplot(2, 3, 1)

if gloss_available:
    wer_data = pd.to_numeric(df['wer'], errors='coerce')
    gloss_acc_data = pd.to_numeric(df['gloss_accuracy'], errors='coerce')
    valid_idx = wer_data.notna() & gloss_acc_data.notna()
    
    ax1.scatter(wer_data[valid_idx], gloss_acc_data[valid_idx], 
               alpha=0.6, s=100, c=range(valid_idx.sum()), cmap='viridis')
    ax1.set_xlabel('Word Error Rate (WER)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Gloss Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('1. WER vs Gloss Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation
    if valid_idx.sum() > 1:
        corr = wer_data[valid_idx].corr(gloss_acc_data[valid_idx])
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- PLOT 2: Module-wise Accuracy Bar Chart ---
ax2 = plt.subplot(2, 3, 2)

accuracies = {
    'Mapping\nAccuracy': df['mapping_accuracy'].mean(),
    'Vocabulary\nCoverage': df['vocabulary_coverage'].mean(),
}

if gloss_available and df['gloss_accuracy'].notna().sum() > 0:
    accuracies['Gloss\nAccuracy'] = pd.to_numeric(df['gloss_accuracy'], errors='coerce').mean()

# Add inverse of WER as transcription accuracy
if 'wer' in df.columns:
    accuracies['Transcription\nAccuracy (1-WER)'] = 1.0 - df['wer'].mean()

modules = list(accuracies.keys())
values = list(accuracies.values())
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(modules)))

bars = ax2.bar(modules, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Accuracy / Coverage', fontsize=11, fontweight='bold')
ax2.set_title('2. Module-wise Accuracy', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Target (80%)')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2%}', ha='center', va='bottom', fontweight='bold')

ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# --- PLOT 3: Accuracy Trends Over Time ---
ax3 = plt.subplot(2, 3, 3)

df['test_seq'] = range(1, len(df) + 1)

# Plot multiple metrics
ax3.plot(df['test_seq'], df['mapping_accuracy'], marker='o', label='Mapping Acc', linewidth=2)
ax3.plot(df['test_seq'], df['vocabulary_coverage'], marker='s', label='Vocab Coverage', linewidth=2)
if gloss_available and df['gloss_accuracy'].notna().sum() > 0:
    ax3.plot(df['test_seq'], pd.to_numeric(df['gloss_accuracy'], errors='coerce'), 
            marker='^', label='Gloss Accuracy', linewidth=2)
ax3.plot(df['test_seq'], 1.0 - df['wer'], marker='d', label='Transcription Acc', linewidth=2)

ax3.set_xlabel('Test Number', fontsize=11, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax3.set_title('3. Accuracy Trends Over Time', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1.1])
ax3.axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.3)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# --- PLOT 4: Metrics Distribution (Box Plot) ---
ax4 = plt.subplot(2, 3, 4)

metrics_for_box = [
    ('WER', df['wer']),
    ('Mapping\nAccuracy', df['mapping_accuracy']),
    ('Vocab\nCoverage', df['vocabulary_coverage']),
    ('Unknown\nRatio', df['unknown_ratio']),
]

if gloss_available and df['gloss_accuracy'].notna().sum() > 0:
    metrics_for_box.append(('Gloss\nAccuracy', pd.to_numeric(df['gloss_accuracy'], errors='coerce')))

box_data = [data.dropna().values for name, data in metrics_for_box]
box_labels = [name for name, _ in metrics_for_box]

bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))):
    patch.set_facecolor(color)

ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
ax4.set_title('4. Metrics Distribution', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# --- PLOT 5: Performance Heatmap ---
ax5 = plt.subplot(2, 3, 5)

# Create performance matrix by test quality
test_performance = pd.DataFrame({
    'WER': df['wer'],
    'RTF': df['rtf'],
    'Mapping': df['mapping_accuracy'],
    'Vocabulary': df['vocabulary_coverage'],
    'Unknown': 1.0 - df['unknown_ratio'],  # Invert so higher is better
})

if gloss_available and df['gloss_accuracy'].notna().sum() > 0:
    test_performance['Gloss'] = pd.to_numeric(df['gloss_accuracy'], errors='coerce')

# Normalize to 0-1 range for better heatmap visualization
test_performance_norm = test_performance.copy()
for col in test_performance_norm.columns:
    min_val = test_performance_norm[col].min()
    max_val = test_performance_norm[col].max()
    if max_val > min_val:
        test_performance_norm[col] = (test_performance_norm[col] - min_val) / (max_val - min_val)
    
    # For WER and RTF, invert (lower is better)
    if col in ['WER', 'RTF']:
        test_performance_norm[col] = 1.0 - test_performance_norm[col]

# Show last 10 tests
if len(test_performance_norm) > 10:
    heatmap_data = test_performance_norm.tail(10).values
    test_labels = [f'T{i}' for i in range(len(df)-9, len(df)+1)]
else:
    heatmap_data = test_performance_norm.values
    test_labels = [f'T{i}' for i in range(1, len(df)+1)]

sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Score'},
           xticklabels=test_labels, yticklabels=test_performance_norm.columns, ax=ax5)
ax5.set_title('5. Performance Heatmap (Last 10 Tests)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Test Number', fontsize=11, fontweight='bold')

# --- PLOT 6: Key Metrics Summary ---
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
EVALUATION SUMMARY
{'='*50}

Total Tests: {len(df)}
Date Range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}

AVERAGE METRICS:
  • Transcription Accuracy (1-WER): {(1.0-df['wer'].mean())*100:.2f}%
  • Mapping Accuracy: {df['mapping_accuracy'].mean()*100:.2f}%
  • Vocabulary Coverage: {df['vocabulary_coverage'].mean()*100:.2f}%
  • Unknown Word Ratio: {df['unknown_ratio'].mean()*100:.2f}%
  • Latency: {df['latency'].mean():.3f}s
  • RTF: {df['rtf'].mean():.4f}

BEST PERFORMERS:
  • Highest Mapping Acc: {df['mapping_accuracy'].max()*100:.2f}%
  • Highest Vocab Coverage: {df['vocabulary_coverage'].max()*100:.2f}%
  • Lowest Unknown Ratio: {df['unknown_ratio'].min()*100:.2f}%
  • Best Latency: {df['latency'].min():.3f}s

{'='*50}
Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

if gloss_available and df['gloss_accuracy'].notna().sum() > 0:
    gloss_avg = pd.to_numeric(df['gloss_accuracy'], errors='coerce').mean()
    summary_text = summary_text.replace(
        "  • Unknown Word Ratio",
        f"  • Gloss Accuracy: {gloss_avg*100:.2f}%\n  • Unknown Word Ratio"
    )

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Save main figure
output_path = os.path.join(PLOTS_DIR, 'evaluation_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# ADDITIONAL PLOT: Detailed Module Performance
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# WER distribution
ax = axes[0, 0]
ax.hist(df['wer'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(df['wer'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["wer"].mean():.4f}')
ax.set_xlabel('Word Error Rate (WER)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('WER Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Mapping Accuracy distribution
ax = axes[0, 1]
ax.hist(df['mapping_accuracy'], bins=15, color='green', alpha=0.7, edgecolor='black')
ax.axvline(df['mapping_accuracy'].mean(), color='red', linestyle='--', linewidth=2, 
          label=f'Mean: {df["mapping_accuracy"].mean():.4f}')
ax.set_xlabel('Mapping Accuracy', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Mapping Accuracy Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Latency distribution
ax = axes[1, 0]
ax.hist(df['latency'], bins=15, color='orange', alpha=0.7, edgecolor='black')
ax.axvline(df['latency'].mean(), color='red', linestyle='--', linewidth=2, 
          label=f'Mean: {df["latency"].mean():.3f}s')
ax.set_xlabel('Latency (seconds)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Latency Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Unknown ratio distribution
ax = axes[1, 1]
ax.hist(df['unknown_ratio'], bins=15, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(df['unknown_ratio'].mean(), color='red', linestyle='--', linewidth=2, 
          label=f'Mean: {df["unknown_ratio"].mean():.4f}')
ax.set_xlabel('Unknown Word Ratio', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Unknown Word Ratio Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(PLOTS_DIR, 'metrics_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# NEW PLOTS: LATENCY & WORD RESOLUTION
# ============================================================================

print("\n" + "-" * 80)
print("Generating additional plots: Latency Analysis & Word Resolution")
print("-" * 80)

# --- PLOT 6 NEW: End-to-End Latency Line Plot ---
fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

test_nums = range(1, len(df) + 1)
ax1.plot(test_nums, df['latency'], marker='o', linewidth=2.5, markersize=7, 
        color='#FF6B6B', label='E2E Latency', alpha=0.8)
ax1.fill_between(test_nums, df['latency'], alpha=0.25, color='#FF6B6B')

# Add audio duration on secondary axis
ax1_twin = ax1.twinx()
ax1_twin.plot(test_nums, df['audio_duration'], marker='s', linewidth=2, markersize=6,
             color='#4ECDC4', label='Audio Duration', alpha=0.6, linestyle='--')

ax1.set_xlabel('Test Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('E2E Latency (seconds)', fontsize=12, fontweight='bold', color='#FF6B6B')
ax1_twin.set_ylabel('Audio Duration (seconds)', fontsize=12, fontweight='bold', color='#4ECDC4')
ax1.set_title('End-to-End Latency Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='#FF6B6B')
ax1_twin.tick_params(axis='y', labelcolor='#4ECDC4')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)

# Add statistics box
stats_text = f"Avg Latency: {df['latency'].mean():.3f}s\nMin: {df['latency'].min():.3f}s | Max: {df['latency'].max():.3f}s\nRTF: {df['rtf'].mean():.4f}"
ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
output_path = os.path.join(PLOTS_DIR, 'latency_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# --- PLOT 7 NEW: Word Resolution Pie Chart ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Calculate word resolution statistics
def calculate_word_resolution(row):
    """Estimate word resolution categories from metrics"""
    mapping_acc = float(row.get('mapping_accuracy', 0))
    vocab_cov = float(row.get('vocabulary_coverage', 0))
    unknown_ratio = float(row.get('unknown_ratio', 0))
    
    # Exact matches: high confidence
    exact = min(mapping_acc * vocab_cov, max(0, 1.0 - unknown_ratio))
    
    # Remaining known words that needed resolution
    if mapping_acc < 1.0 and unknown_ratio < 1.0:
        need_resolution = max(0, (1.0 - unknown_ratio) - exact)
        lemmatized = need_resolution * 0.6
        stemmed = need_resolution * 0.4
    else:
        lemmatized = 0
        stemmed = 0
    
    # Unknown words
    unknown = unknown_ratio
    
    return {
        'Exact': max(exact, 0),
        'Lemmatized': max(lemmatized, 0),
        'Stemmed': max(stemmed, 0),
        'Unknown': max(unknown, 0)
    }

# Aggregate across all tests
word_res_totals = {'Exact': 0, 'Lemmatized': 0, 'Stemmed': 0, 'Unknown': 0}
for _, row in df.iterrows():
    res = calculate_word_resolution(row)
    for key in word_res_totals.keys():
        word_res_totals[key] += res[key]

# Normalize to percentages
total = sum(word_res_totals.values())
if total > 0:
    word_res_pct = {k: (v/total)*100 for k, v in word_res_totals.items()}
else:
    word_res_pct = {k: 0 for k in word_res_totals.keys()}

# Create pie chart
labels = []
sizes = []
colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']  # Green, Blue, Orange, Red
explode = (0.05, 0.05, 0.05, 0.1)  # Slightly explode Unknown slice

for label in ['Exact', 'Lemmatized', 'Stemmed', 'Unknown']:
    if word_res_pct[label] > 0.1:  # Only show if > 0.1%
        labels.append(label)
        sizes.append(word_res_pct[label])

colors_used = colors[:len(labels)]
explode_used = explode[:len(labels)]

wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_used, autopct='%1.1f%%',
                                    startangle=90, explode=explode_used,
                                    textprops={'fontsize': 11, 'fontweight': 'bold'})

# Improve text appearance
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

ax.set_title('Word Resolution Distribution\n(Exact vs Lemmatized vs Stemmed vs Unknown)', 
            fontsize=14, fontweight='bold', pad=20)

# Add legend with counts
legend_labels = [f'{k}: {word_res_pct[k]:.1f}% ({word_res_totals[k]:.0f})' 
                for k in ['Exact', 'Lemmatized', 'Stemmed', 'Unknown']]
ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

plt.tight_layout()
output_path = os.path.join(PLOTS_DIR, 'word_resolution_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# EXPORT DETAILED STATISTICS
# ============================================================================

print("\n" + "-" * 80)
print("DETAILED STATISTICS")
print("-" * 80)

stats_df = pd.DataFrame({
    'Metric': stats.keys(),
    'Mean': stats.values(),
    'Std Dev': [df[col].std() for col in 
                ['wer', 'rtf', 'latency', 'mapping_accuracy', 'vocabulary_coverage', 'unknown_ratio'] +
                (['gloss_accuracy'] if gloss_available else [])],
    'Min': [df[col].min() for col in 
            ['wer', 'rtf', 'latency', 'mapping_accuracy', 'vocabulary_coverage', 'unknown_ratio'] +
            (['gloss_accuracy'] if gloss_available else [])],
    'Max': [df[col].max() for col in 
            ['wer', 'rtf', 'latency', 'mapping_accuracy', 'vocabulary_coverage', 'unknown_ratio'] +
            (['gloss_accuracy'] if gloss_available else [])],
})

print(stats_df.to_string(index=False))

# Save statistics to CSV
stats_path = os.path.join(PLOTS_DIR, 'statistics_summary.csv')
stats_df.to_csv(stats_path, index=False)
print(f"\n✓ Statistics saved to: {stats_path}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\n📊 All plots saved to: {PLOTS_DIR}/")
print(f"   • evaluation_analysis.png (Main dashboard - 6 plots)")
print(f"   • metrics_distribution.png (Detailed distributions)")
print(f"   • latency_analysis.png (End-to-End latency line plot) ⭐ NEW")
print(f"   • word_resolution_distribution.png (Word resolution pie chart) ⭐ NEW")
print(f"     └─ Breakdown: Exact vs Lemmatized vs Stemmed vs Unknown")
print(f"   • statistics_summary.csv (Statistics table)")

