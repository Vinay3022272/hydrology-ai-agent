import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
path = r"C:\Users\ps302\OneDrive\Desktop\Hydrology\src\Flood_Model\raw_data\flood_training_data_10k_clean.csv"
df = pd.read_csv(path)

print("Dataset Info:")
print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns) - 1}")
print("\nClass distribution:")
print(df['flood'].value_counts())

# Separate features and target
X = df.drop('flood', axis=1)
y = df['flood']
y = df['flood']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression with balanced class weights
lr_balanced = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    solver='liblinear'
)

lr_balanced.fit(X_train_scaled, y_train)

# Get predictions and probabilities
y_pred = lr_balanced.predict(X_test_scaled)
y_pred_proba = lr_balanced.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# VISUALIZATION 1: ROC-AUC CURVE
# ============================================================================
plt.figure(figsize=(10, 8))

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

# Add threshold annotations at key points
threshold_points = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for thresh in threshold_points:
    idx = np.argmin(np.abs(thresholds - thresh))
    plt.annotate(f'θ={thresh}', 
                (fpr[idx], tpr[idx]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 2: CLASS DISTRIBUTION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
ax1 = axes[0]
colors = ['#66b3ff', '#ff9999']
flood_counts = df['flood'].value_counts()
ax1.pie(flood_counts, labels=['Non-Flood (0)', 'Flood (1)'], 
        colors=colors, autopct='%1.1f%%', startangle=90, 
        explode=(0.05, 0.1), shadow=True)
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')

# Bar chart
ax2 = axes[1]
bars = ax2.bar(['Non-Flood (0)', 'Flood (1)'], flood_counts.values, color=colors)
ax2.set_title('Class Counts', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count')
ax2.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for bar, count in zip(bars, flood_counts.values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 3: CONFUSION MATRIX HEATMAP
# ============================================================================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Non-Flood', 'Predicted Flood'],
            yticklabels=['Actual Non-Flood', 'Actual Flood'],
            annot_kws={'size': 16, 'fontweight': 'bold'})

plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)

# Add metrics annotations
metrics_text = f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}"
plt.text(1.5, 2.5, metrics_text, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
         fontsize=11, verticalalignment='center')

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 4: PRECISION-RECALL CURVE
# ============================================================================
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, color='green', lw=2, 
         label=f'PR curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 5: PREDICTION PROBABILITY DISTRIBUTION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of prediction probabilities
ax1 = axes[0]
ax1.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, color='blue', 
         label='Actual Non-Flood', edgecolor='black')
ax1.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, color='red', 
         label='Actual Flood', edgecolor='black')
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax1.set_xlabel('Predicted Probability of Flood', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Prediction Probabilities', fontsize=14, fontweight='bold')
ax1.legend(loc="upper center", fontsize=10)
ax1.grid(True, alpha=0.3)

# Box plot of probabilities
ax2 = axes[1]
data_to_plot = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
bp = ax2.boxplot(data_to_plot, labels=['Non-Flood', 'Flood'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax2.set_ylabel('Predicted Probability', fontsize=12)
ax2.set_title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 6: FEATURE IMPORTANCE (TOP 15)
# ============================================================================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_balanced.coef_[0]
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False).head(15)

plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'green' for x in feature_importance['coefficient']]
bars = plt.barh(range(len(feature_importance)), feature_importance['coefficient'], color=colors)
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Top 15 Feature Importances (Logistic Regression Coefficients)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (coef, bar) in enumerate(zip(feature_importance['coefficient'], bars)):
    width = bar.get_width()
    label_x = width + 0.01 if width > 0 else width - 0.05
    plt.text(label_x, bar.get_y() + bar.get_height()/2, 
             f'{coef:.3f}', ha='left' if width > 0 else 'right', 
             va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================================
# VISUALIZATION 7: THRESHOLD OPTIMIZATION
# ============================================================================
thresholds = np.arange(0.1, 0.9, 0.02)
precision_thresh = []
recall_thresh = []
f1_thresh = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    precision_thresh.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recall_thresh.append(recall_score(y_test, y_pred_thresh, zero_division=0))
    f1_thresh.append(f1_score(y_test, y_pred_thresh, zero_division=0))

plt.figure(figsize=(12, 6))
plt.plot(thresholds, precision_thresh, label='Precision', linewidth=2, color='blue')
plt.plot(thresholds, recall_thresh, label='Recall', linewidth=2, color='red')
plt.plot(thresholds, f1_thresh, label='F1-Score', linewidth=2, color='green')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Effect of Threshold on Performance Metrics', fontsize=14, fontweight='bold')
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)

# Find optimal threshold for F1
optimal_idx = np.argmax(f1_thresh)
optimal_threshold = thresholds[optimal_idx]
plt.scatter(optimal_threshold, f1_thresh[optimal_idx], color='green', s=200, 
            marker='*', zorder=5, label=f'Optimal Threshold (F1={f1_thresh[optimal_idx]:.3f})')
plt.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.5)
plt.legend(loc="best", fontsize=10)

plt.tight_layout()
plt.show()

print(f"\nOptimal Threshold based on F1-Score: {optimal_threshold:.3f}")
print(f"F1-Score at optimal threshold: {f1_thresh[optimal_idx]:.4f}")
print(f"Precision at optimal threshold: {precision_thresh[optimal_idx]:.4f}")
print(f"Recall at optimal threshold: {recall_thresh[optimal_idx]:.4f}")

# ============================================================================
# VISUALIZATION 8: CORRELATION HEATMAP (Top features with target)
# ============================================================================
# Get top 10 most important features
top_features = feature_importance['feature'].head(10).tolist()
top_features.append('flood')

# Calculate correlations
corr_df = df[top_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Top 10 Features with Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY DASHBOARD
# ============================================================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('FLOOD PREDICTION MODEL - PERFORMANCE DASHBOARD', fontsize=16, fontweight='bold')

# Subplot 1: Metrics gauge
ax1 = plt.subplot(2, 3, 1)
metrics = ['Precision', 'Recall', 'F1', 'ROC-AUC']
values = [precision, recall, f1, roc_auc]
colors_metrics = ['blue', 'red', 'green', 'purple']
bars = ax1.bar(metrics, values, color=colors_metrics)
ax1.set_ylim([0, 1])
ax1.set_ylabel('Score')
ax1.set_title('Key Metrics', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.3f}', ha='center', fontweight='bold')

# Subplot 2: Confusion matrix percentages
ax2 = plt.subplot(2, 3, 2)
cm_percent = cm.astype('float') / cm.sum() * 100
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=['Pred Non-Flood', 'Pred Flood'],
            yticklabels=['Actual Non-Flood', 'Actual Flood'],
            ax=ax2, cbar=False, annot_kws={'size': 12})
ax2.set_title('Confusion Matrix (%)', fontweight='bold')

# Subplot 3: Class distribution pie
ax3 = plt.subplot(2, 3, 3)
ax3.pie(flood_counts, labels=['Non-Flood', 'Flood'], 
        autopct='%1.1f%%', colors=['lightblue', 'salmon'],
        startangle=90, explode=(0, 0.1))
ax3.set_title('Class Distribution', fontweight='bold')

# Subplot 4: ROC curve small
ax4 = plt.subplot(2, 3, 4)
ax4.plot(fpr, tpr, 'b-', linewidth=2)
ax4.plot([0, 1], [0, 1], 'r--', linewidth=1)
ax4.set_xlabel('FPR')
ax4.set_ylabel('TPR')
ax4.set_title(f'ROC Curve (AUC={roc_auc:.3f})', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Subplot 5: Precision-Recall curve
ax5 = plt.subplot(2, 3, 5)
ax5.plot(recall_vals, precision_vals, 'g-', linewidth=2)
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title(f'PR Curve (AP={avg_precision:.3f})', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Subplot 6: Sample info
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
info_text = f"""
MODEL SUMMARY
══════════════
Training samples: {len(X_train)}
Test samples: {len(X_test)}
Total features: {X.shape[1]}

BEST METRICS
══════════════
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Optimal threshold: {optimal_threshold:.3f}
F1 at optimal: {f1_thresh[optimal_idx]:.4f}
"""
ax6.text(0.1, 0.5, info_text, fontsize=10, fontfamily='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n All visualizations complete!")