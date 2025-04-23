import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#made with Gemini
#"visualise these results in python using heatmaps"

# Your results dictionary
results_data = {'DBScan': {0.1: {3: {'amount_noise': 0.9941,
                      'davies_bouldin_score': 1.5159340376427852,
                      'n_clusters': 12,
                      'silhouette_score': -0.22422503},
                  7: {'amount_noise': 1.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None},
                  15: {'amount_noise': 1.0,
                       'davies_bouldin_score': None,
                       'n_clusters': 0,
                       'silhouette_score': None},
                  50: {'amount_noise': 1.0,
                       'davies_bouldin_score': None,
                       'n_clusters': 0,
                       'silhouette_score': None},
                  125: {'amount_noise': 1.0,
                        'davies_bouldin_score': None,
                        'n_clusters': 0,
                        'silhouette_score': None},
                  250: {'amount_noise': 1.0,
                        'davies_bouldin_score': None,
                        'n_clusters': 0,
                        'silhouette_score': None}},
            0.2: {3: {'amount_noise': 0.9403,
                      'davies_bouldin_score': 1.8609865065188285,
                      'n_clusters': 52,
                      'silhouette_score': -0.24843103},
                  7: {'amount_noise': 0.9608,
                      'davies_bouldin_score': 2.078108924410706,
                      'n_clusters': 12,
                      'silhouette_score': -0.1938657},
                  15: {'amount_noise': 0.9753,
                       'davies_bouldin_score': 2.126703676699954,
                       'n_clusters': 7,
                       'silhouette_score': -0.18305938},
                  50: {'amount_noise': 1.0,
                       'davies_bouldin_score': None,
                       'n_clusters': 0,
                       'silhouette_score': None},
                  125: {'amount_noise': 1.0,
                        'davies_bouldin_score': None,
                        'n_clusters': 0,
                        'silhouette_score': None},
                  250: {'amount_noise': 1.0,
                        'davies_bouldin_score': None,
                        'n_clusters': 0,
                        'silhouette_score': None}},
            0.5: {3: {'amount_noise': 0.1757,
                      'davies_bouldin_score': 2.3354185583230906,
                      'n_clusters': 32,
                      'silhouette_score': 0.03391112},
                  7: {'amount_noise': 0.2296,
                      'davies_bouldin_score': 2.9766555247123536,
                      'n_clusters': 9,
                      'silhouette_score': 0.14085443},
                  15: {'amount_noise': 0.2608,
                       'davies_bouldin_score': 4.5351341822434525,
                       'n_clusters': 2,
                       'silhouette_score': 0.14853807},
                  50: {'amount_noise': 0.2986,
                       'davies_bouldin_score': None,
                       'n_clusters': 1,
                       'silhouette_score': None},
                  125: {'amount_noise': 0.3215,
                        'davies_bouldin_score': None,
                        'n_clusters': 1,
                        'silhouette_score': None},
                  250: {'amount_noise': 0.3452,
                        'davies_bouldin_score': None,
                        'n_clusters': 1,
                        'silhouette_score': None}},
            1: {3: {'amount_noise': 0.0,
                    'davies_bouldin_score': None,
                    'n_clusters': 0,
                    'silhouette_score': None},
                7: {'amount_noise': 0.0,
                    'davies_bouldin_score': None,
                    'n_clusters': 0,
                    'silhouette_score': None},
                15: {'amount_noise': 0.0,
                     'davies_bouldin_score': None,
                     'n_clusters': 0,
                     'silhouette_score': None},
                50: {'amount_noise': 0.0,
                     'davies_bouldin_score': None,
                     'n_clusters': 0,
                     'silhouette_score': None},
                125: {'amount_noise': 0.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None},
                250: {'amount_noise': 0.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None}},
            2: {3: {'amount_noise': 0.0,
                    'davies_bouldin_score': None,
                    'n_clusters': 0,
                    'silhouette_score': None},
                7: {'amount_noise': 0.0,
                    'davies_bouldin_score': None,
                    'n_clusters': 0,
                    'silhouette_score': None},
                15: {'amount_noise': 0.0,
                     'davies_bouldin_score': None,
                     'n_clusters': 0,
                     'silhouette_score': None},
                50: {'amount_noise': 0.0,
                     'davies_bouldin_score': None,
                     'n_clusters': 0,
                     'silhouette_score': None},
                125: {'amount_noise': 0.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None},
                250: {'amount_noise': 0.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None}},
            3: {3: {'amount_noise': 0.0,
                    'davies_bouldin_score': None,
                    'n_clusters': 0,
                    'silhouette_score': None},
                7: {'amount_noise': 0.0,
                    'davies_bouldin_score': None,
                    'n_clusters': 0,
                    'silhouette_score': None},
                15: {'amount_noise': 0.0,
                     'davies_bouldin_score': None,
                     'n_clusters': 0,
                     'silhouette_score': None},
                50: {'amount_noise': 0.0,
                     'davies_bouldin_score': None,
                     'n_clusters': 0,
                     'silhouette_score': None},
                125: {'amount_noise': 0.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None},
                250: {'amount_noise': 0.0,
                      'davies_bouldin_score': None,
                      'n_clusters': 0,
                      'silhouette_score': None}}}}

# Extract data for plotting
eps_values = sorted(results_data['DBScan'].keys())
min_samples_values = sorted(list(results_data['DBScan'][eps_values[0]].keys()))

# Function to extract a specific metric
def extract_metric(metric_name, default_value=np.nan):
    data = np.full((len(eps_values), len(min_samples_values)), default_value, dtype=float)
    for i, eps in enumerate(eps_values):
        for j, ms in enumerate(min_samples_values):
            value = results_data['DBScan'][eps][ms].get(metric_name)
            if value is not None:
                data[i, j] = value
            # Special handling for silhouette which could be None
            elif metric_name == 'silhouette_score':
                 data[i, j] = -2 # Assign a value outside the normal range (-1 to 1) for None
            elif metric_name == 'davies_bouldin_score':
                 data[i,j] = 10 # Assign a high value for DB score when None (lower is better)

    return pd.DataFrame(data, index=eps_values, columns=min_samples_values)

# Extract metrics into DataFrames
n_clusters_df = extract_metric('n_clusters', default_value=0) # Default 0 clusters if None
amount_noise_df = extract_metric('amount_noise', default_value=1.0) # Default 1.0 noise if None
silhouette_df = extract_metric('silhouette_score', default_value=-2) # Use -2 for None silhouette
db_score_df = extract_metric('davies_bouldin_score', default_value=10) # Use 10 for None DB

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DBSCAN Parameter Tuning Results on GloVe Embeddings (Cosine Distance)', fontsize=16)

sns.heatmap(n_clusters_df, annot=True, fmt=".0f", cmap="viridis", ax=axes[0, 0], cbar_kws={'label': 'Number of Clusters'})
axes[0, 0].set_title('Number of Clusters')
axes[0, 0].set_xlabel('min_samples')
axes[0, 0].set_ylabel('eps')

sns.heatmap(amount_noise_df, annot=True, fmt=".2f", cmap="viridis_r", ax=axes[0, 1], cbar_kws={'label': 'Proportion of Noise'}) # Reversed cmap
axes[0, 1].set_title('Amount of Noise')
axes[0, 1].set_xlabel('min_samples')
axes[0, 1].set_ylabel('eps')

# Use vmin/vmax for better color scaling, especially for silhouette
sns.heatmap(silhouette_df, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 0], vmin=-0.5, vmax=0.5, cbar_kws={'label': 'Silhouette Score (None=-2)'})
axes[1, 0].set_title('Silhouette Score (Higher is Better)')
axes[1, 0].set_xlabel('min_samples')
axes[1, 0].set_ylabel('eps')

# Use vmax for better color scaling for DB score
sns.heatmap(db_score_df, annot=True, fmt=".2f", cmap="coolwarm_r", ax=axes[1, 1], vmax=6, cbar_kws={'label': 'Davies-Bouldin Score (None=10)'}) # Reversed cmap, lower is better
axes[1, 1].set_title('Davies-Bouldin Score (Lower is Better)')
axes[1, 1].set_xlabel('min_samples')
axes[1, 1].set_ylabel('eps')


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig('DBScan_HeatMapsCosine.png')  # Save as a PNG file
plt.show()