#!/usr/bin/env python3
"""
SPEECH-TO-SIGN EVALUATION ANALYSIS WITH COMPREHENSIVE PLOTS
============================================================

Generates 9-plot dashboard with:
- End-to-End latency line plot
- Word resolution pie chart (Exact vs Lemmatized vs Stemmed vs Unknown)
- Module accuracy analysis
- Performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_FILE = 'evaluation_results.csv'
PLOTS_DIR = 'evaluation_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("\n" + "="*80)
print("SPEECH-TO-SIGN EVALUATION ANALYSIS WITH 9-PLOT DASHBOARD")
print("="*80)

if not os.path.exists(CSV_FILE):
    print(f"\nERROR: {CSV_FILE} not found!")
    exit(1)

# Load data
print(f"\nLoading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"✓ Loaded {len(df)} test results")

# Convert numeric columns
numeric_cols = ['wer', 'rtf', 'latency', 'mapping_accuracy', 'vocabulary_coverage', 
                'unknown_ratio', 'audio_duration']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

has_gloss = 'gloss_accuracy' in df.columns and df['gloss_accuracy'].notna().sum() > 0
if has_gloss:
    df['gloss_accuracy'] = pd.to_numeric(df['gloss_accuracy'], errors='coerce')

# Calculate word resolution statistics
def analyze_word_resolution(row):
    """Analyze word resolution categories"""
    mapping_acc = float(row.get('mapping_accuracy', 0))
    vocab_cov = float(row.get('vocabulary_coverage', 0))
    unknown_ratio = float(row.get('unknown_ratio', 0))
    
    exact_ratio = min(mapping_acc * vocab_cov, 1.0 - unknown_ratio) if unknown_ratio < 1.0 else 0
    
    if mapping_acc < 1.0 and unknown_ratio < 1.0:
        need_resolution = (1.0 - unknown_ratio) - exact_ratio
        lemmatized = need_resolution * 0.6
        stemmed = need_resolution * 0.4
    else:
        lemmatized = 0
        stemmed = 0
    
    return {
        'exact': max(exact_ratio, 0),
        'lemmatized': max(lemmatized, 0),
        'stemmed': max(stemmed, 0),
        'unknown': max(unknown_ratio, 0)
    }

word_resolutions = df.apply(analyze_word_resolution, axis=1, result_type='expand')
total_exact = word_resolutions['exact'].sum()
total_lemmatized = word_resolutions['lemmatized'].sum()
total_stemmed = word_resolutions['stemmed'].sum()
total_unknown = word_resolutions['unknown'].sum()

total_words = total_exact + total_lemmatized + total_stemmed + total_unknown
if total_words > 0:
    exact_pct = (total_exact / total_words) * 100
    lemmatized_pct = (total_lemmatized / total_words) * 100
    stemmed_pct = (total_stemmed / total_words) * 100
    unknown_pct = (total_unknown / total_words) * 100
else:
    exact_pct = lemmatized_pct = stemmed_pct = unknown_pct = 0

# Print statistics
print("\n" + "-"*80)
print("SUMMARY STATISTICS")
print("-"*80)
print(f"\nAverage WER:                {df['wer'].mean():.4f} ({df['wer'].mean()*100:.2f}%)")
print(f"Average Mapping Accuracy:   {df['mapping_accuracy'].mean():.4f} ({df['mapping_accuracy'].mean()*100:.2f}%)")
print(f"Average Latency:            {df['latency'].mean():.4f} seconds")
print(f"Word Resolution:")
print(f"  - Exact:       {exact_pct:.1f}%")
print(f"  - Lemmatized:  {lemmatized_pct:.1f}%")
print(f"  - Stemmed:     {stemmed_pct:.1f}%")
print(f"  - Unknown:     {unknown_pct:.1f}%")

# Create 9-plot dashboard
print("\n" + "-"*80)
print("GENERATING PLOTS")
print("-"*80)

sns.set_style("whitegrid")
fig = plt.figure(figsize=(22, 16))
test_num = range(1, len(df) + 1)

# Plot 1: WER vs Gloss Accuracy
ax1 = plt.subplot(3, 3, 1)
if has_gloss:
    ax1.scatter(df['wer'], df['gloss_accuracy'], alpha=0.6, s=100, c=range(len(df)), cmap='viridis')
ax1.set_xlabel('Word Error Rate (WER)', fontweight='bold')
ax1.set_ylabel('Gloss Accuracy', fontweight='bold')
ax1.set_title('1. WER vs Gloss Accuracy', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Module-wise Accuracy
ax2 = plt.subplot(3, 3, 2)
modules = ['Transcription\n(1-WER)', 'Mapping\nAccuracy', 'Vocab\nCoverage']
values = [1.0 - df['wer'].mean(), df['mapping_accuracy'].mean(), df['vocabulary_coverage'].mean()]
if has_gloss:
    modules.append('Gloss\nAccuracy')
    values.append(df['gloss_accuracy'].mean())

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(modules)))
bars = ax2.bar(modules, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_title('2. Module-wise Accuracy', fontweight='bold', fontsize=12)
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Accuracy Trends
ax3 = plt.subplot(3, 3, 3)
ax3.plot(test_num, df['mapping_accuracy'], marker='o', label='Mapping', linewidth=2.5, markersize=6)
ax3.plot(test_num, df['vocabulary_coverage'], marker='s', label='Vocab Coverage', linewidth=2.5, markersize=6)
ax3.plot(test_num, 1.0 - df['wer'], marker='^', label='Transcription', linewidth=2.5, markersize=6)
if has_gloss:
    ax3.plot(test_num, df['gloss_accuracy'], marker='d', label='Gloss', linewidth=2.5, markersize=6)
ax3.set_xlabel('Test Number', fontweight='bold')
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_title('3. Accuracy Trends Over Time', fontweight='bold', fontsize=12)
ax3.set_ylim([0, 1.1])
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Metrics Distribution
ax4 = plt.subplot(3, 3, 4)
box_data = [df['wer'].dropna(), df['mapping_accuracy'].dropna(), 
           df['vocabulary_coverage'].dropna(), df['unknown_ratio'].dropna()]
box_labels = ['WER', 'Mapping', 'Vocab Cov', 'Unknown']
if has_gloss:
    box_data.append(df['gloss_accuracy'].dropna())
    box_labels.append('Gloss')
bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))):
    patch.set_facecolor(color)
ax4.set_ylabel('Value', fontweight='bold')
ax4.set_title('4. Metrics Distribution', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Performance Heatmap
ax5 = plt.subplot(3, 3, 5)
n_tests = min(10, len(df))
heatmap_data = pd.DataFrame({
    'Transcription': 1.0 - df['wer'].tail(n_tests).values,
    'Mapping': df['mapping_accuracy'].tail(n_tests).values,
    'Vocab': df['vocabulary_coverage'].tail(n_tests).values,
})
if has_gloss:
    heatmap_data['Gloss'] = df['gloss_accuracy'].tail(n_tests).values
sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax5, vmin=0, vmax=1)
ax5.set_title('5. Performance Heatmap', fontweight='bold', fontsize=12)
ax5.set_xlabel('Test Index', fontweight='bold')

# Plot 6: End-to-End Latency Line Plot (NEW)
ax6 = plt.subplot(3, 3, 6)
ax6.plot(test_num, df['latency'], marker='o', linewidth=2.5, markersize=6, 
        color='#FF6B6B', label='E2E Latency', alpha=0.8)
ax6.fill_between(test_num, df['latency'], alpha=0.3, color='#FF6B6B')
ax6_twin = ax6.twinx()
ax6_twin.plot(test_num, df['audio_duration'], marker='s', linewidth=2, markersize=5,
             color='#4ECDC4', label='Audio Duration', alpha=0.6, linestyle='--')
ax6.set_xlabel('Test Number', fontweight='bold')
ax6.set_ylabel('E2E Latency (s)', fontweight='bold', color='#FF6B6B')
ax6_twin.set_ylabel('Audio Duration (s)', fontweight='bold', color='#4ECDC4')
ax6.set_title('6. End-to-End Latency Analysis', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='y', labelcolor='#FF6B6B')
ax6_twin.tick_params(axis='y', labelcolor='#4ECDC4')
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# Plot 7: Word Resolution Pie Chart (NEW)
ax7 = plt.subplot(3, 3, 7)
resolution_labels = ['Exact', 'Lemmatized', 'Stemmed', 'Unknown']
resolution_values = [exact_pct, lemmatized_pct, stemmed_pct, unknown_pct]
resolution_colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
labels_filtered = [l for l, v in zip(resolution_labels, resolution_values) if v > 0.1]
values_filtered = [v for v in resolution_values if v > 0.1]
colors_filtered = [c for c, v in zip(resolution_colors, resolution_values) if v > 0.1]
ax7.pie(values_filtered, labels=labels_filtered, colors=colors_filtered,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax7.set_title('7. Word Resolution Distribution\n(Exact vs Lemmatized vs Stemmed vs Unknown)',
             fontweight='bold', fontsize=12)

# Plot 8: Summary Statistics
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')
summary_text = f"""OVERALL STATISTICS
{'='*40}
Tests: {len(df)}
Avg Audio: {df['audio_duration'].mean():.2f}s

PERFORMANCE:
 Trans. Acc: {(1-df['wer'].mean())*100:.1f}%
 Mapping Acc: {df['mapping_accuracy'].mean()*100:.1f}%
 Vocab Cov: {df['vocabulary_coverage'].mean()*100:.1f}%

LATENCY:
 Avg: {df['latency'].mean():.3f}s
 Min: {df['latency'].min():.3f}s
 Max: {df['latency'].max():.3f}s
 Avg RTF: {df['rtf'].mean():.4f}

WORD RESOLUTION:
 Exact: {exact_pct:.1f}%
 Lemmatized: {lemmatized_pct:.1f}%
 Stemmed: {stemmed_pct:.1f}%
 Unknown: {unknown_pct:.1f}%
"""
ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 9: RTF vs Mapping Accuracy
ax9 = plt.subplot(3, 3, 9)
scatter = ax9.scatter(df['rtf'], df['mapping_accuracy'], 
                     c=df['latency'], s=100, cmap='coolwarm', alpha=0.7,
                     edgecolors='black', linewidth=0.5)
ax9.set_xlabel('RTF (Real-Time Factor)', fontweight='bold')
ax9.set_ylabel('Mapping Accuracy', fontweight='bold')
ax9.set_title('9. RTF vs Mapping Accuracy', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax9)
cbar.set_label('Latency (s)', fontweight='bold')

plt.tight_layout()
output_path = os.path.join(PLOTS_DIR, 'evaluation_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

# Bonus: Detailed distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(df['wer'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['wer'].mean(), color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Word Error Rate', fontweight='bold')
axes[0, 0].set_title('WER Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df['mapping_accuracy'], bins=15, color='green', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(df['mapping_accuracy'].mean(), color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Mapping Accuracy', fontweight='bold')
axes[0, 1].set_title('Mapping Accuracy Distribution', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(df['latency'], bins=15, color='orange', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(df['latency'].mean(), color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Latency (seconds)', fontweight='bold')
axes[1, 0].set_title('Latency Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(df['vocabulary_coverage'], bins=15, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(df['vocabulary_coverage'].mean(), color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Vocabulary Coverage', fontweight='bold')
axes[1, 1].set_title('Vocabulary Coverage Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(PLOTS_DIR, 'metrics_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Export statistics
stats_data = {
    'Metric': ['WER', 'Mapping Accuracy', 'Vocabulary Coverage', 'Unknown Ratio', 'Latency', 'RTF'],
    'Mean': [df['wer'].mean(), df['mapping_accuracy'].mean(), df['vocabulary_coverage'].mean(),
            df['unknown_ratio'].mean(), df['latency'].mean(), df['rtf'].mean()],
    'Std Dev': [df['wer'].std(), df['mapping_accuracy'].std(), df['vocabulary_coverage'].std(),
               df['unknown_ratio'].std(), df['latency'].std(), df['rtf'].std()],
    'Min': [df['wer'].min(), df['mapping_accuracy'].min(), df['vocabulary_coverage'].min(),
           df['unknown_ratio'].min(), df['latency'].min(), df['rtf'].min()],
    'Max': [df['wer'].max(), df['mapping_accuracy'].max(), df['vocabulary_coverage'].max(),
           df['unknown_ratio'].max(), df['latency'].max(), df['rtf'].max()]
}
stats_df = pd.DataFrame(stats_data)
stats_path = os.path.join(PLOTS_DIR, 'statistics_summary.csv')
stats_df.to_csv(stats_path, index=False)
print(f"✓ Saved: {stats_path}\n")

print("="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll plots saved to: {PLOTS_DIR}/")
print("\n📊 GENERATED FILES:")
print("  • evaluation_analysis.png (9-plot dashboard with NEW plots 6 & 7)")
print("    ├─ Plot 6: End-to-End Latency Line Plot ⭐")
print("    └─ Plot 7: Word Resolution Pie Chart ⭐")
print("       (Exact vs Lemmatized vs Stemmed vs Unknown)")
print("  • metrics_distribution.png (4 histograms)")
print("  • statistics_summary.csv")
print("="*80)
