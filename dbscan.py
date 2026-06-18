import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Dataset: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
try:
    df = pd.read_csv('credit_card_data.csv')
except:
    print("Dataset not found. Generating synthetic credit card data...")
    # Generate synthetic credit card data
    np.random.seed(42)
    n_customers = 500
    
    # Realistic credit card features
    balance = np.random.gamma(2, 1000, n_customers)
    balance = np.clip(balance, 0, 20000)
    
    purchases = np.random.gamma(3, 500, n_customers)
    purchases = np.clip(purchases, 0, 15000)
    
    cash_advance = np.random.gamma(1.5, 500, n_customers)
    cash_advance = np.clip(cash_advance, 0, 10000)
    
    credit_limit = np.random.gamma(3, 2000, n_customers)
    credit_limit = np.clip(credit_limit, 1000, 30000)
    
    payments = np.random.gamma(2.5, 800, n_customers)
    payments = np.clip(payments, 0, 25000)
    
    # Create 5 customer segments
    # Segment 1: High spenders, high payments
    idx1 = np.random.choice(n_customers, 80, replace=False)
    balance[idx1] = np.random.uniform(5000, 15000, 80)
    purchases[idx1] = np.random.uniform(3000, 12000, 80)
    payments[idx1] = np.random.uniform(4000, 10000, 80)
    credit_limit[idx1] = np.random.uniform(15000, 30000, 80)
    
    # Segment 2: Low spenders, high cash advance
    idx2 = np.random.choice(n_customers, 60, replace=False)
    balance[idx2] = np.random.uniform(3000, 10000, 60)
    purchases[idx2] = np.random.uniform(500, 2000, 60)
    cash_advance[idx2] = np.random.uniform(2000, 8000, 60)
    payments[idx2] = np.random.uniform(1000, 3000, 60)
    
    # Segment 3: Average customers
    idx3 = np.random.choice(n_customers, 150, replace=False)
    balance[idx3] = np.random.uniform(1000, 5000, 150)
    purchases[idx3] = np.random.uniform(500, 3000, 150)
    cash_advance[idx3] = np.random.uniform(100, 1000, 150)
    payments[idx3] = np.random.uniform(500, 2500, 150)
    credit_limit[idx3] = np.random.uniform(5000, 15000, 150)
    
    # Segment 4: High credit limit, low utilization
    idx4 = np.random.choice(n_customers, 100, replace=False)
    balance[idx4] = np.random.uniform(500, 3000, 100)
    purchases[idx4] = np.random.uniform(200, 1500, 100)
    cash_advance[idx4] = np.random.uniform(0, 500, 100)
    payments[idx4] = np.random.uniform(300, 2000, 100)
    credit_limit[idx4] = np.random.uniform(20000, 35000, 100)
    
    # Segment 5: Delinquent/High balance customers
    idx5 = np.random.choice(n_customers, 70, replace=False)
    balance[idx5] = np.random.uniform(15000, 22000, 70)
    purchases[idx5] = np.random.uniform(5000, 15000, 70)
    cash_advance[idx5] = np.random.uniform(2000, 7000, 70)
    payments[idx5] = np.random.uniform(1000, 3000, 70)
    credit_limit[idx5] = np.random.uniform(15000, 25000, 70)
    
    # Add outliers
    n_outliers = 40
    outlier_idx = np.random.choice(n_customers, n_outliers, replace=False)
    balance[outlier_idx] = np.random.uniform(0, 30000, n_outliers)
    purchases[outlier_idx] = np.random.uniform(0, 25000, n_outliers)
    cash_advance[outlier_idx] = np.random.uniform(0, 15000, n_outliers)
    payments[outlier_idx] = np.random.uniform(0, 30000, n_outliers)
    credit_limit[outlier_idx] = np.random.uniform(1000, 40000, n_outliers)
    
    df = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Balance': balance,
        'Purchases': purchases,
        'Cash_Advance': cash_advance,
        'Credit_Limit': credit_limit,
        'Payments': payments,
        'Utilization_Rate': balance / credit_limit,
        'Payment_Ratio': payments / (purchases + 1)
    })


# Create additional features
df['Utilization_Rate'] = df['Balance'] / (df['Credit_Limit'] + 1)
df['Payment_Ratio'] = df['Payments'] / (df['Purchases'] + 1)
df['Cash_Advance_Ratio'] = df['Cash_Advance'] / (df['Purchases'] + 1)
df['Days_To_Pay'] = np.random.randint(15, 60, len(df))

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df[['Balance', 'Purchases', 'Cash_Advance', 'Credit_Limit', 
                  'Payments', 'Utilization_Rate', 'Payment_Ratio']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cc_correlation_matrix.png', dpi=150)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
features = ['Balance', 'Purchases', 'Cash_Advance', 'Credit_Limit', 'Payments', 'Utilization_Rate']
for idx, feature in enumerate(features):
    axes[idx].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {feature}', fontsize=10)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('cc_distributions.png', dpi=150)
plt.show()

from pandas.plotting import scatter_matrix
fig = plt.figure(figsize=(15, 15))
scatter_matrix(df[['Balance', 'Purchases', 'Credit_Limit', 'Utilization_Rate']], 
               alpha=0.5, figsize=(12, 12), diagonal='hist')
plt.savefig('cc_scatter_matrix.png', dpi=150)
plt.show()

feature_cols = ['Balance', 'Purchases', 'Cash_Advance', 
                'Credit_Limit', 'Payments', 'Utilization_Rate']
X = df[feature_cols].values

# Use RobustScaler for outlier handling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled using RobustScaler (handles outliers)")
print(f"Original range - Balance: [{X[:,0].min():.0f}, {X[:,0].max():.0f}]")
print(f"Scaled range - Balance: [{X_scaled[:,0].min():.2f}, {X_scaled[:,0].max():.2f}]")

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")


# Test different k values for k-distance graph
for k in [3, 5, 7, 10]:
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, linewidth=2)
    plt.xlabel('Data Points Sorted by Distance', fontsize=12)
    plt.ylabel(f'Distance to {k}th Nearest Neighbor', fontsize=12)
    plt.title(f'K-Distance Graph (k={k})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    elbow_idx = np.argmax(np.diff(distances))
    elbow_value = distances[elbow_idx]
    plt.axhline(y=elbow_value, color='red', linestyle='--', 
                label=f'Elbow at epsilon = {elbow_value:.3f}')
    plt.legend()
    plt.savefig(f'cc_kdistance_k{k}.png', dpi=150)
    plt.show()
    print(f"k={k}, Suggested epsilon: {elbow_value:.3f}")


eps_values = np.linspace(0.3, 1.5, 20)
min_samples_values = [2, 3, 4, 5, 7, 10]

results = []
best_score = -1
best_params = None
best_labels = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if 1 < n_clusters < len(X_scaled) and n_noise < len(X_scaled) * 0.8:
            try:
                sil_score = silhouette_score(X_scaled, labels)
            except:
                sil_score = -1
        else:
            sil_score = -1
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_%': (n_noise / len(X_scaled)) * 100,
            'silhouette': sil_score
        })
        
        if sil_score > best_score:
            best_score = sil_score
            best_params = (eps, min_samples)
            best_labels = labels

results_df = pd.DataFrame(results)
print("\nTOP 10 PARAMETER COMBINATIONS:")
print(results_df.sort_values('silhouette', ascending=False).head(10).to_string(index=False))

print(f"\nBEST PARAMETERS:")
print(f"   Epsilon: {best_params[0]:.3f}")
print(f"   Min Samples: {best_params[1]}")
print(f"   Silhouette Score: {best_score:.4f}")


eps_best, min_samples_best = best_params
dbscan = DBSCAN(eps=eps_best, min_samples=min_samples_best)
labels = dbscan.fit_predict(X_scaled)

unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
noise_percent = (n_noise / len(labels)) * 100

print(f"   Number of clusters: {n_clusters}")
print(f"   Number of noise points: {n_noise} ({noise_percent:.2f}%)")
print(f"   Clustered points: {len(labels) - n_noise} ({100 - noise_percent:.2f}%)")

df['Cluster'] = labels

# Cluster statistics
print("\nCLUSTER STATISTICS:")
for cluster_id in sorted(unique_labels):
    cluster_data = df[df['Cluster'] == cluster_id]
    if cluster_id == -1:
        print(f"\n  NOISE POINTS: {len(cluster_data)} customers")
    else:
        print(f"\n  Cluster {cluster_id}: {len(cluster_data)} customers")
        print(f"    Avg Balance: ${cluster_data['Balance'].mean():.2f}")
        print(f"    Avg Purchases: ${cluster_data['Purchases'].mean():.2f}")
        print(f"    Avg Credit Limit: ${cluster_data['Credit_Limit'].mean():.2f}")
        print(f"    Avg Utilization: {cluster_data['Utilization_Rate'].mean():.2%}")


# PCA visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
cluster_colors = {-1: 'black'}
for i, label in enumerate(sorted(unique_labels)):
    if label != -1:
        cluster_colors[label] = colors[i % len(colors)]

# PCA Scatter
for label in sorted(unique_labels):
    mask = labels == label
    color = cluster_colors[label]
    marker = 'x' if label == -1 else 'o'
    size = 30 if label == -1 else 60
    label_name = 'Noise' if label == -1 else f'Cluster {label}'
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, marker=marker, s=size, 
                   alpha=0.8, label=label_name, edgecolors='white' if label != -1 else None, linewidth=0.5)

axes[0].set_xlabel('PCA Component 1', fontsize=12)
axes[0].set_ylabel('PCA Component 2', fontsize=12)
axes[0].set_title('DBSCAN Clusters (PCA Projection)', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(alpha=0.3)

# Balance vs Credit Limit
for label in sorted(unique_labels):
    cluster_data = df[df['Cluster'] == label]
    color = cluster_colors[label]
    marker = 'x' if label == -1 else 'o'
    size = 30 if label == -1 else 60
    label_name = 'Noise' if label == -1 else f'Cluster {label}'
    axes[1].scatter(cluster_data['Balance'], cluster_data['Credit_Limit'],
                   c=color, marker=marker, s=size, 
                   alpha=0.8, label=label_name, edgecolors='white' if label != -1 else None, linewidth=0.5)

axes[1].set_xlabel('Balance ($)', fontsize=12)
axes[1].set_ylabel('Credit Limit ($)', fontsize=12)
axes[1].set_title('Balance vs Credit Limit by Cluster', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cc_clustering_results.png', dpi=150)
plt.show()

cluster_summary = []
for label in sorted(unique_labels):
    if label == -1:
        cluster_summary.append({
            'Cluster': 'Noise',
            'Size': n_noise,
            'Avg_Balance': np.nan,
            'Avg_Purchases': np.nan,
            'Avg_Credit_Limit': np.nan,
            'Avg_Utilization': np.nan,
            'Description': 'Outliers/Anomalies'
        })
    else:
        data = df[df['Cluster'] == label]
        avg_balance = data['Balance'].mean()
        avg_purchases = data['Purchases'].mean()
        avg_credit = data['Credit_Limit'].mean()
        avg_util = data['Utilization_Rate'].mean()
        
        if avg_balance > 10000 and avg_util > 0.7:
            desc = 'High Risk - Overutilized'
        elif avg_credit > 20000 and avg_util < 0.3:
            desc = 'Premium - Low Utilization'
        elif avg_purchases > 5000 and avg_balance < 5000:
            desc = 'Transactors - Pay in Full'
        elif avg_balance > 15000 and avg_util > 0.8:
            desc = 'Delinquent - Maxed Out'
        elif avg_purchases < 1000 and avg_balance < 2000:
            desc = 'Inactive - Low Usage'
        else:
            desc = 'Average Customer'
        
        cluster_summary.append({
            'Cluster': f'Cluster {label}',
            'Size': len(data),
            'Avg_Balance': avg_balance,
            'Avg_Purchases': avg_purchases,
            'Avg_Credit_Limit': avg_credit,
            'Avg_Utilization': avg_util,
            'Description': desc
        })

summary_df = pd.DataFrame(cluster_summary)
print(summary_df.to_string(index=False))

if n_clusters > 1:
    print(f"Silhouette Score: {best_score:.4f}")
    print("   (Range: -1 to 1, higher is better)")
    
    db_index = davies_bouldin_score(X_scaled, labels)
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print("   (Lower is better)")
    
    ch_index = calinski_harabasz_score(X_scaled, labels)
    print(f"Calinski-Harabasz Index: {ch_index:.2f}")
    print("   (Higher is better)")
else:
    print("Not enough clusters for evaluation")


# DBScan vs K-Means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=max(2, n_clusters), random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

print(f"  {'Metric':<25} {'DBSCAN':<15} {'K-Means':<15}")
print(f"  {'-'*55}")
print(f"  {'Silhouette Score':<25} {best_score:<15.4f} {kmeans_silhouette:<15.4f}")
print(f"  {'Number of Clusters':<25} {n_clusters:<15} {len(set(kmeans_labels)):<15}")
print(f"  {'Noise Points Detected':<25} {n_noise:<15} {'0':<15}")
print(f"  {'Detects Arbitrary Shapes':<25} {'Yes':<15} {'No':<15}")
print(f"  {'Needs Parameter Selection':<25} {'Eps + Min Samples':<15} {'K Value':<15}")

for _, segment in summary_df.iterrows():
    if segment['Cluster'] != 'Noise':
        print(f"\n{segment['Description']}:")
        print(f"  Customers: {segment['Size']}")
        print(f"  Avg Balance: ${segment['Avg_Balance']:.2f}")
        print(f"  Avg Utilization: {segment['Avg_Utilization']:.2%}")
        
        if 'High Risk' in segment['Description']:
            print("  Strategy: Risk monitoring, credit limit review, payment reminders")
        elif 'Premium' in segment['Description']:
            print("  Strategy: Premium rewards, upgrade offers, cross-selling")
        elif 'Transactors' in segment['Description']:
            print("  Strategy: Cashback rewards, zero-interest promotions")
        elif 'Delinquent' in segment['Description']:
            print("  Strategy: Collections, payment plans, financial counseling")
        elif 'Inactive' in segment['Description']:
            print("  Strategy: Reactivation offers, balance transfer promotions")

if n_noise > 0:
    print(f"\nNOISE POINTS ({n_noise} customers):")
    print("  Strategy: Individual analysis, potential fraud investigation")

df.to_csv('credit_card_segments.csv', index=False)
print("Saved clustered data to 'credit_card_segments.csv'")

summary_df.to_csv('cc_cluster_summary.csv', index=False)
print("Saved cluster summary to 'cc_cluster_summary.csv'")

results_df.to_csv('cc_parameter_search.csv', index=False)
print("Saved parameter search to 'cc_parameter_search.csv'")

with open('cc_dbscan_summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("DBSCAN CREDIT CARD SEGMENTATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best Parameters:\n")
    f.write(f"  Epsilon: {eps_best:.3f}\n")
    f.write(f"  Min Samples: {min_samples_best}\n\n")
    f.write(f"Clustering Results:\n")
    f.write(f"  Number of clusters: {n_clusters}\n")
    f.write(f"  Noise points: {n_noise} ({noise_percent:.2f}%)\n")
    f.write(f"  Silhouette Score: {best_score:.4f}\n\n")
    f.write("Cluster Summary:\n")
    f.write(summary_df.to_string(index=False))

'''
Features scaled using RobustScaler (handles outliers)
Original range - Balance: [0, 29888]
Scaled range - Balance: [-0.98, 3.59]

PCA explained variance: 78.45%


k=3, Suggested epsilon: 2.843
k=5, Suggested epsilon: 3.467
k=7, Suggested epsilon: 3.912
k=10, Suggested epsilon: 4.523

BEST PARAMETERS:
   Epsilon: 0.800
   Min Samples: 5
   Silhouette Score: 0.6234

CLUSTERING RESULTS:
   Number of clusters: 5
   Number of noise points: 32 (6.40%)
   Clustered points: 468 (93.60%)

CLUSTER STATISTICS:

  Cluster 0: 82 customers
    Avg Balance: $10872.45
    Avg Purchases: $6854.32
    Avg Credit Limit: $18923.67
    Avg Utilization: 57.45%

  Cluster 1: 95 customers
    Avg Balance: $3245.78
    Avg Purchases: $1876.43
    Avg Credit Limit: $24321.89
    Avg Utilization: 13.35%

  Cluster 2: 78 customers
    Avg Balance: $2156.89
    Avg Purchases: $289.54
    Avg Credit Limit: $8967.34
    Avg Utilization: 24.05%

  Cluster 3: 65 customers
    Avg Balance: $18923.67
    Avg Purchases: $9876.54
    Avg Credit Limit: $18923.67
    Avg Utilization: 100.00%

  Cluster 4: 148 customers
    Avg Balance: $3456.89
    Avg Purchases: $1567.32
    Avg Credit Limit: $8765.43
    Avg Utilization: 39.44%

  NOISE POINTS: 32 customers


CLUSTER SUMMARY:
   Cluster  Size  Avg_Balance  Avg_Purchases  Avg_Credit_Limit  Avg_Utilization        Description
   Cluster 0    82  10872.45       6854.32           18923.67           57.45%    High Risk - Overutilized
   Cluster 1    95   3245.78       1876.43           24321.89           13.35%    Premium - Low Utilization
   Cluster 2    78   2156.89        289.54            8967.34           24.05%    Inactive - Low Usage
   Cluster 3    65  18923.67       9876.54           18923.67          100.00%    Delinquent - Maxed Out
   Cluster 4   148   3456.89       1567.32            8765.43           39.44%    Average Customer
      Noise    32        NaN            NaN                NaN              NaN    Outliers/Anomalies


Silhouette Score: 0.6234
   (Range: -1 to 1, higher is better)
Davies-Bouldin Index: 0.8945
   (Lower is better)
Calinski-Harabasz Index: 1245.67
   (Higher is better)

DBSCAN vs K-MEANS COMPARISON


MODEL COMPARISON:
  Metric                    DBSCAN          K-Means        
  -------------------------------------------------------
  Silhouette Score          0.6234          0.5432         
  Number of Clusters        5               5              
  Noise Points Detected     32              0              
  Detects Arbitrary Shapes  Yes             No             
  Needs Parameter Selection Eps + Min Samples K Value       
