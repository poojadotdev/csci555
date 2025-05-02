import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# Write Phase
write_data_kb = 500.0
write_duration_s = 4.46
write_data_mb = write_data_kb / 1024.0
write_throughput_mb_s = write_data_mb / write_duration_s

# Read Phases
read_phases_labels = ['Sequential Read', 'Reverse Read', 'Random Read']
read_durations_s = [16.40, 15.39, 15.99]
read_data_mb = 4.88
read_throughputs_mb_s = [read_data_mb / d for d in read_durations_s]

# Prepare for plotting
all_labels = ['Write'] + read_phases_labels
all_throughputs = [write_throughput_mb_s] + read_throughputs_mb_s

# Colors (optional - one for write, another for reads)
colors = ['#ffcc99'] + ['#66b3ff'] * len(read_phases_labels) # Orange, Blue, Blue, Blue

# --- Create Throughput Comparison Plot ---
plt.figure(figsize=(8, 5))
bars = plt.bar(all_labels, all_throughputs, color=colors)

plt.ylabel('Throughput (MB/s)')
plt.title('Write vs. Read Phase Throughput (LSTM Run)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f'{yval:.3f} MB/s', # Show 3 decimal places
            va='bottom', ha='center', fontsize=9)

# Adjust ylim for better visibility, start near 0
plt.ylim(0, max(all_throughputs) * 1.15)
plt.xticks(rotation=10) # Slightly rotate labels if needed

plt.tight_layout()
plt.savefig('write_vs_read_throughput.png')
# plt.show()

print(f"Write throughput: {write_throughput_mb_s:.3f} MB/s")
print(f"Sequential Read throughput: {read_throughputs_mb_s[0]:.3f} MB/s")
print(f"Reverse Read throughput: {read_throughputs_mb_s[1]:.3f} MB/s")
print(f"Random Read throughput: {read_throughputs_mb_s[2]:.3f} MB/s")
print("\nWrite vs. Read throughput graph saved as write_vs_read_throughput.png")