import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

results = {'GMM': {3: {'davies_bouldin_score': 5.274182218645962,
             'silhouette_score': 0.07473756},
         4: {'davies_bouldin_score': 5.6341188290628414,
             'silhouette_score': 0.06497207},
         5: {'davies_bouldin_score': 3.9687476515317273,
             'silhouette_score': -0.027800066},
         6: {'davies_bouldin_score': 5.749521413824362,
             'silhouette_score': 0.0525212},
         7: {'davies_bouldin_score': 5.345027321916791,
             'silhouette_score': 0.034214925},
         8: {'davies_bouldin_score': 6.539582703856707,
             'silhouette_score': 0.0352406},
         9: {'davies_bouldin_score': 6.042852959759275,
             'silhouette_score': -0.057228718},
         10: {'davies_bouldin_score': 6.113458258147597,
              'silhouette_score': -0.020584928},
         11: {'davies_bouldin_score': 5.840495112244724,
              'silhouette_score': -0.051833455},
         12: {'davies_bouldin_score': 5.173633444185909,
              'silhouette_score': 0.02210028},
         13: {'davies_bouldin_score': 5.904673506339386,
              'silhouette_score': -0.04759026},
         14: {'davies_bouldin_score': 5.009360700116555,
              'silhouette_score': -0.07220053}},
 'KMeans': {3: {'davies_bouldin_score': 4.710887968053205,
                'silhouette_score': 0.12409525},
            4: {'davies_bouldin_score': 4.750136589089264,
                'silhouette_score': 0.00030361544},
            5: {'davies_bouldin_score': 4.668779647754294,
                'silhouette_score': 0.011572539},
            6: {'davies_bouldin_score': 4.404818194173554,
                'silhouette_score': 0.0039514597},
            7: {'davies_bouldin_score': 4.532430717313027,
                'silhouette_score': 0.0020868937},
            8: {'davies_bouldin_score': 4.488529318882314,
                'silhouette_score': -0.0006590467},
            9: {'davies_bouldin_score': 4.417787842604017,
                'silhouette_score': 0.0012648372},
            10: {'davies_bouldin_score': 4.209069472237935,
                 'silhouette_score': 0.0015956322},
            11: {'davies_bouldin_score': 4.482569072021378,
                 'silhouette_score': -0.0025755446},
            12: {'davies_bouldin_score': 4.574517390403753,
                 'silhouette_score': -0.0042376444},
            13: {'davies_bouldin_score': 4.345784710169638,
                 'silhouette_score': -0.0014827935},
            14: {'davies_bouldin_score': 4.321644392545351,
                 'silhouette_score': -0.0007718298}}}


# Extract data for plotting
k_values = sorted(results['KMeans'].keys()) # Use KMeans keys (assuming same range)

gmm_silhouette = [results['GMM'][k]['silhouette_score'] for k in k_values]
kmeans_silhouette = [results['KMeans'][k]['silhouette_score'] for k in k_values]

gmm_db = [results['GMM'][k]['davies_bouldin_score'] for k in k_values]
kmeans_db = [results['KMeans'][k]['davies_bouldin_score'] for k in k_values]

# Create the plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Clustering Evaluation Metrics vs. Number of Clusters (k)', fontsize=16)

# --- Silhouette Score Plot ---
axes[0].plot(k_values, gmm_silhouette, marker='o', linestyle='-', label='GMM')
axes[0].plot(k_values, kmeans_silhouette, marker='x', linestyle='-', label='K-Means')
axes[0].set_title('Silhouette Score')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Silhouette Score (Higher is Better)')
axes[0].axhline(0, color='grey', linestyle='--', linewidth=0.8)
axes[0].axhline(1, color='red', linestyle=':', linewidth=1, label='Max Possible Score (1.0)')
axes[0].axhline(-1, color='red', linestyle=':', linewidth=1, label='Min Possible Score (-1.0)')

axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle=':')
axes[0].set_xticks(k_values) # Ensure all k values are shown as ticks

# --- Davies-Bouldin Score Plot ---
axes[1].plot(k_values, gmm_db, marker='o', linestyle='--', label='GMM')
axes[1].plot(k_values, kmeans_db, marker='x', linestyle='-', label='K-Means')
axes[1].set_title('Davies-Bouldin Score')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Davies-Bouldin Score (Lower is Better)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle=':')
axes[1].set_xticks(k_values) # Ensure all k values are shown as ticks

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
plt.savefig('k-means&GMM_graphs.png')  # Save as a PNG file
plt.show()