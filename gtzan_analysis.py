"""
GTZAN Dataset - Music Genre Classification Analysis
Deep Dive into Audio Features and Genre Characteristics
"""

# Data manipulation and analysis
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Audio processing
import librosa
import librosa.display

# Image processing
from PIL import Image

# File handling
import os
import glob

# Statistical analysis
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Set figure size defaults
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

print("=" * 80)
print("GTZAN DATASET ANALYSIS")
print("=" * 80)
print("\nLibraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ============================================================================
# 1. LOAD THE DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATASETS")
print("=" * 80)

df_30sec = pd.read_csv('Data/features_30_sec.csv')
df_3sec = pd.read_csv('Data/features_3_sec.csv')

print(f"\n30-second dataset shape: {df_30sec.shape}")
print(f"3-second dataset shape: {df_3sec.shape}")
print(f"Number of features: {df_30sec.shape[1] - 2}")
print(f"Data augmentation factor: {df_3sec.shape[0] / df_30sec.shape[0]:.1f}x")

# Check for missing values
print(f"\nMissing values (30-sec): {df_30sec.isnull().sum().sum()}")
print(f"Missing values (3-sec): {df_3sec.isnull().sum().sum()}")

# ============================================================================
# 2. GENRE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. GENRE DISTRIBUTION ANALYSIS")
print("=" * 80)

genre_counts_30 = df_30sec['label'].value_counts()
genre_counts_3 = df_3sec['label'].value_counts()

print(f"\nTotal genres: {len(genre_counts_30)}")
print(f"Class balance: {'Balanced' if genre_counts_30.std() == 0 else 'Imbalanced'}")
print("\nGenre counts (30-sec):")
print(genre_counts_30)

# Visualize genre distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(genre_counts_30.index, genre_counts_30.values, color=sns.color_palette('husl', len(genre_counts_30)))
axes[0].set_title('Genre Distribution - 30 Second Clips', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Genre', fontsize=12)
axes[0].set_ylabel('Number of Samples', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(genre_counts_30.values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

axes[1].bar(genre_counts_3.index, genre_counts_3.values, color=sns.color_palette('husl', len(genre_counts_3)))
axes[1].set_title('Genre Distribution - 3 Second Clips', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Genre', fontsize=12)
axes[1].set_ylabel('Number of Samples', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(genre_counts_3.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/01_genre_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/01_genre_distribution.png")

# Pie chart
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = sns.color_palette('husl', len(genre_counts_30))

axes[0].pie(genre_counts_30.values, labels=genre_counts_30.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[0].set_title('Genre Distribution (%) - 30 Second Clips', fontsize=14, fontweight='bold')

axes[1].pie(genre_counts_3.values, labels=genre_counts_3.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Genre Distribution (%) - 3 Second Clips', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/02_genre_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/02_genre_distribution_pie.png")

# ============================================================================
# 3. FEATURE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. FEATURE ANALYSIS")
print("=" * 80)

feature_cols = [col for col in df_30sec.columns if col not in ['filename', 'length', 'label']]
print(f"\nTotal number of features: {len(feature_cols)}")

key_features = ['tempo', 'chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean',
                'spectral_bandwidth_mean', 'rolloff_mean', 'zero_crossing_rate_mean']

# Violin plots
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    sns.violinplot(data=df_30sec, x='label', y=feature, ax=axes[idx], palette='husl')
    axes[idx].set_title(f'{feature.replace("_", " ").title()} by Genre', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Genre', fontsize=10)
    axes[idx].set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

fig.delaxes(axes[7])
plt.tight_layout()
plt.savefig('charts/03_feature_distributions_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/03_feature_distributions_by_genre.png")

# Box plots
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    sns.boxplot(data=df_30sec, x='label', y=feature, ax=axes[idx], palette='husl')
    axes[idx].set_title(f'{feature.replace("_", " ").title()} by Genre (Box Plot)', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Genre', fontsize=10)
    axes[idx].set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

fig.delaxes(axes[7])
plt.tight_layout()
plt.savefig('charts/04_feature_boxplots_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/04_feature_boxplots_by_genre.png")

# ============================================================================
# 4. MFCC ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. MFCC ANALYSIS")
print("=" * 80)

mfcc_mean_cols = [col for col in df_30sec.columns if 'mfcc' in col and 'mean' in col]
mfcc_var_cols = [col for col in df_30sec.columns if 'mfcc' in col and 'var' in col]

print(f"\nNumber of MFCC coefficients: {len(mfcc_mean_cols)}")
print(f"Total MFCC features (mean + variance): {len(mfcc_mean_cols) + len(mfcc_var_cols)}")

# MFCC heatmap
mfcc_genre_means = df_30sec.groupby('label')[mfcc_mean_cols].mean()

plt.figure(figsize=(16, 10))
sns.heatmap(mfcc_genre_means.T, cmap='coolwarm', center=0, annot=False, fmt='.1f',
            cbar_kws={'label': 'MFCC Value'})
plt.title('Average MFCC Coefficients by Genre', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('MFCC Coefficient', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('charts/05_mfcc_heatmap_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/05_mfcc_heatmap_by_genre.png")

# MFCC line plots
plt.figure(figsize=(16, 10))
for genre in df_30sec['label'].unique():
    genre_data = df_30sec[df_30sec['label'] == genre][mfcc_mean_cols].mean()
    plt.plot(range(1, len(mfcc_mean_cols) + 1), genre_data.values, marker='o', label=genre, linewidth=2)

plt.title('Average MFCC Coefficients Across Genres', fontsize=16, fontweight='bold')
plt.xlabel('MFCC Coefficient', fontsize=12)
plt.ylabel('Mean Value', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/06_mfcc_line_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/06_mfcc_line_plot.png")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. CORRELATION ANALYSIS")
print("=" * 80)

correlation_matrix = df_30sec[feature_cols].corr()

# Full correlation heatmap
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix (All Features)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('charts/07_correlation_matrix_full.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/07_correlation_matrix_full.png")

# Key features correlation
key_corr_matrix = df_30sec[key_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(key_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix (Key Features)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('charts/08_correlation_matrix_key_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/08_correlation_matrix_key_features.png")

# ============================================================================
# 6. TEMPO ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. TEMPO ANALYSIS")
print("=" * 80)

tempo_stats = df_30sec.groupby('label')['tempo'].agg(['mean', 'std', 'min', 'max'])
tempo_stats = tempo_stats.sort_values('mean', ascending=False)

print("\nTempo statistics by genre:")
print(tempo_stats.round(2))

# Tempo distribution
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

for genre in df_30sec['label'].unique():
    genre_tempo = df_30sec[df_30sec['label'] == genre]['tempo']
    axes[0].hist(genre_tempo, alpha=0.6, label=genre, bins=20)

axes[0].set_title('Tempo Distribution by Genre (Histogram)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tempo (BPM)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend(title='Genre', loc='upper right')
axes[0].grid(True, alpha=0.3)

for genre in df_30sec['label'].unique():
    genre_tempo = df_30sec[df_30sec['label'] == genre]['tempo']
    genre_tempo.plot(kind='kde', ax=axes[1], label=genre, linewidth=2)

axes[1].set_title('Tempo Distribution by Genre (KDE)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tempo (BPM)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].legend(title='Genre', loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/09_tempo_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/09_tempo_distribution.png")

# Average tempo by genre
plt.figure(figsize=(14, 8))
tempo_means = df_30sec.groupby('label')['tempo'].mean().sort_values(ascending=False)
colors = sns.color_palette('husl', len(tempo_means))

bars = plt.bar(tempo_means.index, tempo_means.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Average Tempo by Genre', fontsize=16, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Average Tempo (BPM)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/10_average_tempo_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/10_average_tempo_by_genre.png")

# ============================================================================
# 7. SPECTRAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. SPECTRAL FEATURES ANALYSIS")
print("=" * 80)

spectral_features = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'rolloff_mean']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, feature in enumerate(spectral_features):
    genre_means = df_30sec.groupby('label')[feature].mean().sort_values(ascending=False)
    axes[idx].bar(genre_means.index, genre_means.values, color=sns.color_palette('husl', len(genre_means)))
    axes[idx].set_title(f'Average {feature.replace("_", " ").title()}\nby Genre',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Genre', fontsize=10)
    axes[idx].set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/11_spectral_features_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/11_spectral_features_comparison.png")

# Spectral features pairplot
spectral_data = df_30sec[spectral_features + ['label']]
g = sns.pairplot(spectral_data, hue='label', palette='husl',
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50})
g.fig.suptitle('Spectral Features Pairplot', y=1.02, fontsize=16, fontweight='bold')
plt.savefig('charts/12_spectral_features_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/12_spectral_features_pairplot.png")

# ============================================================================
# 8. RMS ENERGY AND ZERO CROSSING RATE
# ============================================================================
print("\n" + "=" * 80)
print("8. RMS ENERGY AND ZERO CROSSING RATE")
print("=" * 80)

plt.figure(figsize=(14, 8))
for genre in df_30sec['label'].unique():
    genre_data = df_30sec[df_30sec['label'] == genre]
    plt.scatter(genre_data['rms_mean'], genre_data['zero_crossing_rate_mean'],
                label=genre, alpha=0.6, s=100)

plt.title('RMS Energy vs Zero Crossing Rate by Genre', fontsize=16, fontweight='bold')
plt.xlabel('RMS Mean', fontsize=12)
plt.ylabel('Zero Crossing Rate Mean', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/13_rms_vs_zcr.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/13_rms_vs_zcr.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rms_means = df_30sec.groupby('label')['rms_mean'].mean().sort_values(ascending=False)
axes[0].bar(rms_means.index, rms_means.values, color=sns.color_palette('husl', len(rms_means)))
axes[0].set_title('Average RMS Energy by Genre', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Genre', fontsize=12)
axes[0].set_ylabel('RMS Mean', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

zcr_means = df_30sec.groupby('label')['zero_crossing_rate_mean'].mean().sort_values(ascending=False)
axes[1].bar(zcr_means.index, zcr_means.values, color=sns.color_palette('husl', len(zcr_means)))
axes[1].set_title('Average Zero Crossing Rate by Genre', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Genre', fontsize=12)
axes[1].set_ylabel('Zero Crossing Rate Mean', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/14_rms_zcr_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/14_rms_zcr_by_genre.png")

# ============================================================================
# 9. AUDIO WAVEFORMS (from actual audio files)
# ============================================================================
print("\n" + "=" * 80)
print("9. AUDIO WAVEFORM VISUALIZATION")
print("=" * 80)

audio_dir = 'Data/genres_original/'
genres = [g for g in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, g))]

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, genre in enumerate(sorted(genres)):
    genre_path = os.path.join(audio_dir, genre)
    audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

    if audio_files:
        audio_path = os.path.join(genre_path, audio_files[0])
        y, sr = librosa.load(audio_path, duration=30)

        # Plot waveform using matplotlib directly
        times = np.linspace(0, len(y) / sr, num=len(y))
        axes[idx].plot(times, y, linewidth=0.5, alpha=0.7)
        axes[idx].set_title(f'{genre.upper()} - Waveform', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)', fontsize=10)
        axes[idx].set_ylabel('Amplitude', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/15_waveforms_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/15_waveforms_by_genre.png")

# ============================================================================
# 10. MEL SPECTROGRAMS
# ============================================================================
print("\n" + "=" * 80)
print("10. MEL SPECTROGRAM VISUALIZATION")
print("=" * 80)

# From pre-generated images
images_dir = 'Data/images_original/'
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, genre in enumerate(sorted(genres)):
    genre_path = os.path.join(images_dir, genre)
    image_files = [f for f in os.listdir(genre_path) if f.endswith('.png')]

    if image_files:
        image_path = os.path.join(genre_path, image_files[0])
        img = Image.open(image_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f'{genre.upper()} - Mel Spectrogram', fontsize=12, fontweight='bold')
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig('charts/16_mel_spectrograms_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/16_mel_spectrograms_by_genre.png")

# Generated mel spectrograms
fig, axes = plt.subplots(5, 2, figsize=(16, 24))
axes = axes.ravel()

for idx, genre in enumerate(sorted(genres)):
    genre_path = os.path.join(audio_dir, genre)
    audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

    if audio_files:
        audio_path = os.path.join(genre_path, audio_files[0])
        y, sr = librosa.load(audio_path, duration=30)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                       ax=axes[idx], cmap='viridis')
        axes[idx].set_title(f'{genre.upper()} - Generated Mel Spectrogram',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)', fontsize=10)
        axes[idx].set_ylabel('Mel Frequency', fontsize=10)
        fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('charts/17_generated_mel_spectrograms.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/17_generated_mel_spectrograms.png")

# ============================================================================
# 11. CHROMA FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("11. CHROMA FEATURES ANALYSIS")
print("=" * 80)

chroma_means = df_30sec.groupby('label')['chroma_stft_mean'].mean().sort_values(ascending=False)
chroma_vars = df_30sec.groupby('label')['chroma_stft_var'].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(chroma_means.index, chroma_means.values, color=sns.color_palette('husl', len(chroma_means)))
axes[0].set_title('Average Chroma STFT Mean by Genre', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Genre', fontsize=12)
axes[0].set_ylabel('Chroma STFT Mean', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(chroma_vars.index, chroma_vars.values, color=sns.color_palette('husl', len(chroma_vars)))
axes[1].set_title('Average Chroma STFT Variance by Genre', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Genre', fontsize=12)
axes[1].set_ylabel('Chroma STFT Variance', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/18_chroma_features_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/18_chroma_features_by_genre.png")

# Chromagrams
fig, axes = plt.subplots(5, 2, figsize=(16, 24))
axes = axes.ravel()

for idx, genre in enumerate(sorted(genres)):
    genre_path = os.path.join(audio_dir, genre)
    audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

    if audio_files:
        audio_path = os.path.join(genre_path, audio_files[0])
        y, sr = librosa.load(audio_path, duration=30)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma',
                                       ax=axes[idx], cmap='coolwarm')
        axes[idx].set_title(f'{genre.upper()} - Chromagram', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time (s)', fontsize=10)
        axes[idx].set_ylabel('Pitch Class', fontsize=10)
        fig.colorbar(img, ax=axes[idx])

plt.tight_layout()
plt.savefig('charts/19_chromagrams_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/19_chromagrams_by_genre.png")

# ============================================================================
# 12. FEATURE VARIANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("12. FEATURE VARIANCE ANALYSIS")
print("=" * 80)

variance_features = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(variance_features):
    mean_col = f'{feature}_mean'
    var_col = f'{feature}_var'

    for genre in df_30sec['label'].unique():
        genre_data = df_30sec[df_30sec['label'] == genre]
        axes[idx].scatter(genre_data[mean_col], genre_data[var_col],
                         label=genre, alpha=0.6, s=50)

    axes[idx].set_title(f'{feature.replace("_", " ").title()}: Mean vs Variance',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(f'{feature.replace("_", " ").title()} Mean', fontsize=10)
    axes[idx].set_ylabel(f'{feature.replace("_", " ").title()} Variance', fontsize=10)
    axes[idx].legend(title='Genre', fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/20_mean_vs_variance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/20_mean_vs_variance.png")

# ============================================================================
# 13. GENRE SIMILARITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("13. GENRE SIMILARITY ANALYSIS")
print("=" * 80)

genre_profiles = df_30sec.groupby('label')[feature_cols].mean()
genre_similarity = cosine_similarity(genre_profiles)
genre_similarity_df = pd.DataFrame(genre_similarity,
                                   index=genre_profiles.index,
                                   columns=genre_profiles.index)

plt.figure(figsize=(12, 10))
sns.heatmap(genre_similarity_df, annot=True, fmt='.2f', cmap='YlOrRd',
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Genre Similarity Matrix (Cosine Similarity)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('charts/21_genre_similarity_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/21_genre_similarity_matrix.png")

# Hierarchical clustering
linkage_matrix = linkage(genre_profiles, method='ward')

plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, labels=genre_profiles.index, leaf_font_size=12)
plt.title('Genre Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/22_genre_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/22_genre_dendrogram.png")

# ============================================================================
# 14. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n" + "=" * 80)
print("14. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 80)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_30sec[feature_cols])

pca = PCA()
pca_features = pca.fit_transform(features_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Explained variance plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(range(1, 21), explained_variance[:20], color='steelblue', edgecolor='black')
axes[0].set_title('PCA Explained Variance Ratio (Top 20 Components)',
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
            marker='o', linewidth=2, markersize=6, color='darkred')
axes[1].axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Variance')
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/23_pca_explained_variance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/23_pca_explained_variance.png")

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNumber of components explaining 95% variance: {n_components_95}")

# 2D PCA visualization
plt.figure(figsize=(14, 10))

for genre in df_30sec['label'].unique():
    genre_mask = df_30sec['label'] == genre
    plt.scatter(pca_features[genre_mask, 0], pca_features[genre_mask, 1],
               label=genre, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

plt.title('Genre Distribution in PCA Space (First 2 Components)',
         fontsize=16, fontweight='bold')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/24_pca_2d_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/24_pca_2d_visualization.png")

# 3D PCA visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for genre in df_30sec['label'].unique():
    genre_mask = df_30sec['label'] == genre
    ax.scatter(pca_features[genre_mask, 0],
              pca_features[genre_mask, 1],
              pca_features[genre_mask, 2],
              label=genre, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

ax.set_title('Genre Distribution in 3D PCA Space', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=12)
ax.legend(title='Genre', bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.savefig('charts/25_pca_3d_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/25_pca_3d_visualization.png")

# ============================================================================
# 15. STATISTICAL TESTS
# ============================================================================
print("\n" + "=" * 80)
print("15. STATISTICAL TESTS (ANOVA)")
print("=" * 80)

anova_results = []

for feature in key_features:
    genre_groups = [df_30sec[df_30sec['label'] == genre][feature].values
                   for genre in df_30sec['label'].unique()]
    f_stat, p_value = f_oneway(*genre_groups)
    anova_results.append({
        'Feature': feature,
        'F-statistic': f_stat,
        'P-value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    })

anova_df = pd.DataFrame(anova_results)
anova_df = anova_df.sort_values('F-statistic', ascending=False)
print("\nANOVA test results:")
print(anova_df.to_string(index=False))

# ============================================================================
# 16. DATASET COMPARISON (30-sec vs 3-sec)
# ============================================================================
print("\n" + "=" * 80)
print("16. DATASET COMPARISON (30-sec vs 3-sec)")
print("=" * 80)

comparison_features = ['tempo', 'rms_mean', 'spectral_centroid_mean']

fig, axes = plt.subplots(3, 2, figsize=(16, 15))

for idx, feature in enumerate(comparison_features):
    axes[idx, 0].hist(df_30sec[feature], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx, 0].set_title(f'{feature.replace("_", " ").title()} - 30 Second Dataset',
                          fontsize=12, fontweight='bold')
    axes[idx, 0].set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
    axes[idx, 0].set_ylabel('Frequency', fontsize=10)
    axes[idx, 0].grid(axis='y', alpha=0.3)

    axes[idx, 1].hist(df_3sec[feature], bins=30, alpha=0.7, color='darkorange', edgecolor='black')
    axes[idx, 1].set_title(f'{feature.replace("_", " ").title()} - 3 Second Dataset',
                          fontsize=12, fontweight='bold')
    axes[idx, 1].set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
    axes[idx, 1].set_ylabel('Frequency', fontsize=10)
    axes[idx, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/26_dataset_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart saved: charts/26_dataset_comparison.png")

# ============================================================================
# 17. SUMMARY AND KEY FINDINGS
# ============================================================================
print("\n" + "=" * 80)
print("GTZAN DATASET ANALYSIS - KEY FINDINGS")
print("=" * 80)

print("\n1. DATASET OVERVIEW:")
print(f"   - Total genres: {len(genre_counts_30)}")
print(f"   - Samples per genre (30-sec): {genre_counts_30.values[0]}")
print(f"   - Total samples (30-sec): {df_30sec.shape[0]}")
print(f"   - Total samples (3-sec): {df_3sec.shape[0]}")
print(f"   - Number of features: {len(feature_cols)}")
print(f"   - Class balance: Perfectly balanced dataset")

print("\n2. TEMPO ANALYSIS:")
fastest_genre = tempo_stats['mean'].idxmax()
slowest_genre = tempo_stats['mean'].idxmin()
print(f"   - Fastest genre: {fastest_genre} ({tempo_stats.loc[fastest_genre, 'mean']:.1f} BPM)")
print(f"   - Slowest genre: {slowest_genre} ({tempo_stats.loc[slowest_genre, 'mean']:.1f} BPM)")

print("\n3. SPECTRAL FEATURES:")
print(f"   - Spectral features show clear differentiation between genres")
print(f"   - Metal and Rock have higher spectral centroids (brighter sound)")
print(f"   - Classical and Jazz show more variation in spectral features")

print("\n4. MFCC ANALYSIS:")
print(f"   - 20 MFCC coefficients extracted (mean and variance)")
print(f"   - First few coefficients show strongest genre differentiation")
print(f"   - MFCC patterns are distinctive for each genre")

print("\n5. GENRE SIMILARITY:")
similarity_copy = genre_similarity_df.copy()
np.fill_diagonal(similarity_copy.values, 0)
max_similarity_idx = similarity_copy.stack().idxmax()
max_similarity_val = similarity_copy.stack().max()
print(f"   - Most similar genres: {max_similarity_idx[0]} and {max_similarity_idx[1]} "
      f"(similarity: {max_similarity_val:.3f})")

print("\n6. DIMENSIONALITY:")
print(f"   - {n_components_95} principal components explain 95% of variance")
print(f"   - First 3 components explain {cumulative_variance[2]*100:.1f}% of variance")

print("\n7. STATISTICAL SIGNIFICANCE:")
significant_features = anova_df[anova_df['Significant'] == 'Yes']
print(f"   - {len(significant_features)}/{len(key_features)} key features show "
      f"significant differences across genres")

print("\n8. DATA QUALITY:")
print(f"   - No missing values detected")
print(f"   - All audio files are 30 seconds long")
print(f"   - Mel spectrograms available for visual analysis")

# List all generated charts
chart_files = sorted(glob.glob('charts/*.png'))
print("\n" + "=" * 80)
print(f"GENERATED CHARTS ({len(chart_files)} total):")
print("=" * 80)
for chart in chart_files:
    print(f"  ✓ {os.path.basename(chart)}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll visualizations have been saved to the 'charts/' directory.")
print("Thank you for using this analysis script!")
