import matplotlib.pyplot as plt
import numpy as np

# --- Data from Benchmark Output ---
phases = ['Write', 'Sequential Read', 'Reverse Read', 'Random Read']
durations = [4.46, 16.40, 15.39, 15.99] # Durations in seconds

read_phases = ['Sequential Read', 'Reverse Read', 'Random Read']
read_durations = [16.40, 15.39, 15.99] # Durations in seconds
data_read_mb = 4.88 # MB read in each phase

# Calculate throughput (MB/s)
throughputs = [data_read_mb / d for d in read_durations]

# Colors for consistency (optional)
colors_phases = ['#ffcc99', '#66b3ff', '#99ff99', '#ff9999'] # Orange, Blue, Green, Red
colors_throughput = ['#66b3ff', '#99ff99', '#ff9999'] # Blue, Green, Red

# --- Graph 1: Benchmark Phase Durations ---
plt.figure(figsize=(8, 5))
bars1 = plt.bar(phases, durations, color=colors_phases)

plt.ylabel('Duration (seconds)')
plt.title('Benchmark Phase Durations (LSTM Run)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=10) # Slightly rotate labels if needed

# Add value labels on bars
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f} s', va='bottom', ha='center')

plt.tight_layout()
plt.savefig('benchmark_phase_durations.png') # Save the plot
# plt.show()

# --- Graph 2: Read Phase Throughput ---
plt.figure(figsize=(7, 5))
bars2 = plt.bar(read_phases, throughputs, color=colors_throughput)

plt.ylabel('Throughput (MB/s)')
plt.title('Read Phase Throughput Comparison (LSTM Run)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=10)

# Add value labels on bars
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.2f} MB/s', va='bottom', ha='center')

# Adjust ylim for better visibility
plt.ylim(0, max(throughputs) * 1.15)

plt.tight_layout()
plt.savefig('benchmark_read_throughput.png') # Save the plot
# plt.show()

print("Benchmark timing graphs saved as PNG files.")