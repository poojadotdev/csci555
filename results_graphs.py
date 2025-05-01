import matplotlib.pyplot as plt

strategies = ['LRU', 'Markov']
misses = [201, 52]

plt.figure(figsize=(6, 4))
bars = plt.bar(strategies, misses, color=['#A3BFFA', '#A7F3D0'])
plt.ylabel('Total Cache Misses')
plt.title('Cache Misses and Read Time Impact')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{int(yval)}', ha='center')
plt.text(0.5, max(misses) * 0.8, 'Negligible Read Time Impact\n(~12.61s for both)', ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('miss_impact.png', dpi=300, bbox_inches='tight')
plt.show()