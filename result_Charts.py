import matplotlib.pyplot as plt
import numpy as np

# Data
strategies = ['LRU Baseline', 'LRU + Markov']
hits = [100, 209]
misses = [201, 52]
prefetches = [0, 192]
hit_rates = [33.22, 80.08]

# Pastel colors
pastel_blue = '#A3BFFA'  # Hits
pastel_red = '#FCA5A5'   # Misses
pastel_green = '#A7F3D0' # Prefetches

# Option 1: Grouped Bar Chart
plt.figure(figsize=(10, 6))
bar_width = 0.25
r1 = np.arange(len(strategies))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.bar(r1, hits, color=pastel_blue, width=bar_width, label='Hits')
plt.bar(r2, misses, color=pastel_red, width=bar_width, label='Misses')
plt.bar(r3, prefetches, color=pastel_green, width=bar_width, label='Prefetches')

# Add Hit Rate annotations
for i, strategy in enumerate(strategies):
    plt.text(i + bar_width, max(hits[i], misses[i], prefetches[i]) + 10, 
            f'Hit Rate: {hit_rates[i]}%', ha='center', fontsize=10)

plt.xlabel('Caching Strategy')
plt.ylabel('Count')
plt.title('Cache Performance Comparison: LRU vs Markov')
plt.xticks([r + bar_width for r in range(len(strategies))], strategies)
plt.legend()
plt.tight_layout()

# Save the grouped bar chart
plt.savefig('grouped_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Option 2: Separate Bar Charts
# Chart A: Hit Rate
plt.figure(figsize=(6, 4))
plt.bar(strategies, hit_rates, color=pastel_blue)
plt.ylabel('Hit Rate (%)')
plt.title('Cache Hit Rate Improvement')
for i, v in enumerate(hit_rates):
    plt.text(i, v + 2, f'{v}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('hit_rate_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Chart B: Misses
plt.figure(figsize=(6, 4))
plt.bar(strategies, misses, color=pastel_red)
plt.ylabel('Number of Misses')
plt.title('Cache Miss Reduction')
for i, v in enumerate(misses):
    plt.text(i, v + 5, f'{v}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('misses_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Chart C: Prefetches
plt.figure(figsize=(6, 4))
plt.bar(strategies, prefetches, color=pastel_green)
plt.ylabel('Number of Prefetches')
plt.title('Prefetch Activity')
for i, v in enumerate(prefetches):
    plt.text(i, v + 5, f'{v}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('prefetches_chart.png', dpi=300, bbox_inches='tight')
plt.show()