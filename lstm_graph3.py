import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# --- Data ---
# Manually extracted prefetch counts for files 0000-0099
# IMPORTANT: Please verify this list against your actual statistics output!
prefetch_counts_per_file = [
    1, 1, 1, 1, 2, 3, 2, 2, 1, 3, 3, 1, 2, 1, 2, 1, 1, 3, 1, 2, # 0-19
    2, 1, 4, 4, 2, 3, 3, 1, 3, 1, 3, 1, 2, 1, 1, 2, 2, 3, 4, 4, # 20-39
    2, 2, 4, 3, 4, 3, 3, 3, 2, 2, 2, 3, 2, 3, 3, 4, 3, 3, 2, 2, # 40-59
    4, 3, 3, 2, 0, 3, 1, 2, 3, 1, 1, 2, 1, 1, 3, 2, 2, 2, 1, 3, # 60-79
    1, 1, 3, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 2, 2, 0, 1, 1, 1, 1  # 80-99
]

# Verify total
if len(prefetch_counts_per_file) != 100:
    print(f"Warning: Expected 100 prefetch counts, but got {len(prefetch_counts_per_file)}")
calculated_total = sum(prefetch_counts_per_file)
print(f"Sum of extracted prefetches: {calculated_total}") # Should be 204 if correct

# --- Calculate Frequency Distribution ---
prefetch_distribution = Counter(prefetch_counts_per_file)

# Prepare data for plotting (ensure all possible counts 0-4 are included)
max_prefetches_observed = max(prefetch_counts_per_file) if prefetch_counts_per_file else 0
labels = [str(i) for i in range(max_prefetches_observed + 1)] # Labels '0', '1', '2', ...
frequencies = [prefetch_distribution.get(i, 0) for i in range(max_prefetches_observed + 1)] # Get count for each label

# --- Create Histogram Plot ---
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, frequencies, color='skyblue')

plt.xlabel('Number of Successful Prefetches per File')
plt.ylabel('Number of Files')
plt.title('Distribution of Successful Prefetches (LSTM Run)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    if yval > 0: # Only label bars with counts
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval}', va='bottom', ha='center') # Adjust offset as needed

plt.tight_layout()
plt.savefig('prefetch_distribution_histogram.png')
# plt.show()

print("Prefetch distribution histogram saved as prefetch_distribution_histogram.png")

# --- (Optional) Code for Detailed Bar Chart ---
# This might be too wide for a slide but useful for analysis

# plt.figure(figsize=(20, 6)) # Make figure wider
# file_indices = np.arange(100)
# plt.bar(file_indices, prefetch_counts_per_file, color='lightcoral')
# plt.xlabel('File Index (benchmark_file_XXXX)')
# plt.ylabel('Prefetch Count')
# plt.title('Successful Prefetches per Benchmark File (LSTM Run)')
# plt.xticks(np.arange(0, 100, 5), rotation=45, ha="right") # Show ticks every 5 files
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig('prefetch_per_file_detail.png')
# # plt.show()
# print("Detailed prefetch-per-file bar chart saved as prefetch_per_file_detail.png")