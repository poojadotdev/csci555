import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# Baseline data (representative from runs with 0 prefetches)
baseline_hits = 100
baseline_misses = 180 # Using an average/representative value
baseline_prefetches = 0
baseline_interactions = baseline_hits + baseline_misses
baseline_hit_rate = (baseline_hits / baseline_interactions) * 100 if baseline_interactions > 0 else 0

# LSTM data (from your latest successful run)
lstm_hits = 241
lstm_misses = 110
lstm_prefetches = 204
lstm_interactions = lstm_hits + lstm_misses
lstm_hit_rate = (lstm_hits / lstm_interactions) * 100 if lstm_interactions > 0 else 0

# Labels for the charts
labels = ['Baseline (No Prefetching)', 'LSTM Prefetching']

# --- Graph 1: Overall Cache Hit Rate ---
plt.figure(figsize=(6, 5))
hit_rates = [baseline_hit_rate, lstm_hit_rate]
bars1 = plt.bar(labels, hit_rates, color=['#ff9999', '#66b3ff']) # Light Red, Light Blue

plt.ylabel('Cache Hit Rate (%)')
plt.title('Overall Cache Hit Rate Comparison')
plt.ylim(0, 100) # Set Y-axis limit to 100%
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', va='bottom', ha='center')

plt.tight_layout()
plt.savefig('cache_hit_rate_comparison.png') # Save the plot
# plt.show() # Uncomment to display plot interactively

# --- Graph 2: Cache Interaction Breakdown (Hits vs. Misses) ---
plt.figure(figsize=(7, 5))
hits_data = [baseline_hits, lstm_hits]
misses_data = [baseline_misses, lstm_misses]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

rects1 = plt.bar(x - width/2, hits_data, width, label='Cache Hits', color='#66b3ff')
rects2 = plt.bar(x + width/2, misses_data, width, label='Cache Misses', color='#ff9999')

# Add some text for labels, title and axes ticks
plt.ylabel('Count')
plt.title('Cache Interactions: Hits vs. Misses')
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Function to add labels on bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('cache_interactions_breakdown.png') # Save the plot
# plt.show() # Uncomment to display plot interactively


# --- Graph 3: Successful Prefetch Count ---
plt.figure(figsize=(6, 5))
prefetch_counts = [baseline_prefetches, lstm_prefetches]
bars3 = plt.bar(labels, prefetch_counts, color=['#c2c2f0','#99ff99']) # Light Purple, Light Green

plt.ylabel('Count')
plt.title('Successful Prefetch Operations')
# Adjust ylim if needed, add some padding
plt.ylim(0, max(prefetch_counts) * 1.15 if max(prefetch_counts) > 0 else 10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars3:
    yval = bar.get_height()
    # Only add label if value > 0 for clarity
    if yval >= 0:
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(prefetch_counts)*0.01 , f'{yval}', va='bottom', ha='center')


plt.tight_layout()
plt.savefig('successful_prefetches.png') # Save the plot
# plt.show() # Uncomment to display plot interactively

print("Graphs saved as PNG files.")