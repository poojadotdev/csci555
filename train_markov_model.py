#!/usr/bin/env python3

import json
import os
from collections import defaultdict

# --- Configuration ---
LOG_FILE = "access_log.txt"
PROB_SAVE_PATH = "markov_probabilities.json"
# ---------------------

def train_markov_model(log_file):
    """
    Trains a first-order Markov model from the access log.
    Returns a dictionary representing transition probabilities:
    { previous_file: { next_file: probability } }
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found. Cannot train.")
        return None

    try:
        with open(log_file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading log file '{log_file}': {e}")
        return None


    if len(paths) < 2:
        print(f"Warning: Not enough data (found {len(paths)} accesses, need >= 2) in '{log_file}' to build transitions.")
        return {} # Return empty dict if no transitions possible

    # Count transitions: transitions[prev][next] = count
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(paths) - 1):
        prev_path = paths[i]
        next_path = paths[i+1]
        # Basic check to avoid self-loops if desired, or allow them
        # if prev_path != next_path: # Uncomment to ignore self-transitions
        transitions[prev_path][next_path] += 1

    # Calculate probabilities: probabilities[prev][next] = probability
    probabilities = defaultdict(dict)
    transition_count = 0
    for prev_path, next_counts in transitions.items():
        total_transitions_from_prev = sum(next_counts.values())
        if total_transitions_from_prev > 0:
            for next_path, count in next_counts.items():
                probabilities[prev_path][next_path] = count / total_transitions_from_prev
                transition_count += count

    print(f"Processed {len(paths)} accesses from '{log_file}'.")
    print(f"Generated {transition_count} transitions from {len(probabilities)} unique files.")
    return dict(probabilities) # Convert back to regular dict for JSON

def main():
    print(f"Starting Markov Model training using '{LOG_FILE}'...")
    probabilities = train_markov_model(LOG_FILE)

    if probabilities is None:
        print("Exiting due to log file issue.")
        return
    elif not probabilities:
        print("No transition probabilities were generated (likely insufficient data).")
        print(f"Skipping save to '{PROB_SAVE_PATH}'.")
        return

    print(f"Saving transition probabilities to '{PROB_SAVE_PATH}'...")
    try:
        with open(PROB_SAVE_PATH, 'w') as f:
            json.dump(probabilities, f, indent=4) # indent for readability
        print("Training complete. Probabilities saved.")
    except Exception as e:
        print(f"Error saving probabilities to '{PROB_SAVE_PATH}': {e}")


if __name__ == "__main__":
    main()