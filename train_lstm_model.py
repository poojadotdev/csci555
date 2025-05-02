#!/usr/bin/env python3
import sys
import json
import numpy as np
import random
from collections import defaultdict, Counter # Use Counter for frequency
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout # Added Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping # Added EarlyStopping
from sklearn.model_selection import train_test_split # For splitting data

# --- Constants ---
PAD_IDX = 0 # Explicitly define padding/OOV index
DEFAULT_SEQ_LENGTH = 5 # Match FUSE script default

# --- Data Preparation Functions ---

def build_vocabulary(access_sequence, min_freq=1):
    """Builds vocabulary mapping paths to indices, starting from 1. Index 0 is reserved."""
    print("Building vocabulary...")
    path_counts = Counter(access_sequence)
    # Filter paths by minimum frequency if desired
    filtered_paths = [path for path, count in path_counts.items() if count >= min_freq]
    # Sort paths for consistent vocabulary generation (optional, but good practice)
    # Sorting alphabetically might be more stable than frequency if counts are similar
    sorted_unique_paths = sorted(filtered_paths)

    # Start mapping from index 1, reserving 0 for padding/OOV
    vocab = {path: i + 1 for i, path in enumerate(sorted_unique_paths)}
    # Add PAD token mapping for clarity, though not strictly used by vocab lookup
    # vocab['<PAD>'] = PAD_IDX # You generally don't look up '<PAD>'
    vocab_size = len(vocab) + 1 # +1 for the reserved index 0
    print(f"Vocabulary size: {vocab_size} (including padding/OOV token)")
    # Extract only the benchmark file paths for pattern generation
    benchmark_files = sorted([p for p in vocab.keys() if 'benchmark_file_' in p])
    return vocab, vocab_size, benchmark_files

def create_sequences_from_list(path_list, seq_length, vocab):
    """Creates input sequences (X) and target sequences (y) from a list of paths."""
    sequences = []
    next_paths = []
    if len(path_list) <= seq_length:
        return [], []

    for i in range(len(path_list) - seq_length):
        seq = path_list[i:i+seq_length]
        next_p = path_list[i+seq_length]
        # Convert paths to indices, using PAD_IDX (0) if OOV (though vocab should contain all)
        # We strictly check here to only include sequences with known paths.
        # During inference, we *will* map unknown paths to PAD_IDX.
        if all(p in vocab for p in seq) and next_p in vocab:
             sequences.append([vocab.get(p, PAD_IDX) for p in seq]) # Use .get for safety
             next_paths.append(vocab.get(next_p, PAD_IDX)) # Should always be found here

    return sequences, next_paths

def generate_synthetic_sequences(benchmark_files, seq_length, vocab, num_sequences=500):
    """Generates synthetic sequences mimicking benchmark patterns."""
    print(f"Generating synthetic sequences (sequential, reverse)...")
    sequences = []
    next_paths = []
    num_files = len(benchmark_files)
    if num_files <= seq_length:
        print("Warning: Not enough unique benchmark files found to generate synthetic sequences.")
        return [], []

    # 1. Sequential Patterns
    for _ in range(num_sequences // 2): # Generate half sequential
        start_index = random.randint(0, num_files - seq_length - 1)
        seq_paths = benchmark_files[start_index : start_index + seq_length]
        next_p = benchmark_files[start_index + seq_length]
        # Convert to indices
        if all(p in vocab for p in seq_paths) and next_p in vocab:
            sequences.append([vocab[p] for p in seq_paths])
            next_paths.append(vocab[next_p])

    # 2. Reverse Patterns
    for _ in range(num_sequences // 2): # Generate half reverse
        # Need to pick end index such that start index is >= 0
        end_index = random.randint(seq_length, num_files - 1)
        # Sequence goes from end_index down to end_index - seq_length + 1
        seq_paths = benchmark_files[end_index - seq_length + 1 : end_index + 1]
        seq_paths.reverse() # Make it reverse order
        # The path *before* the start of the sequence is the next one in reverse
        next_p = benchmark_files[end_index - seq_length]
         # Convert to indices
        if all(p in vocab for p in seq_paths) and next_p in vocab:
            sequences.append([vocab[p] for p in seq_paths])
            next_paths.append(vocab[next_p])

    print(f"Generated {len(sequences)} synthetic sequences.")
    return sequences, next_paths


# --- Model Building Function ---

def build_lstm_model(vocab_size, embedding_size, lstm_units, seq_length):
    """Builds the LSTM model."""
    print("Building LSTM model...")
    model = Sequential([
        # Input dim is vocab_size (includes index 0)
        # mask_zero=True can help LSTM ignore padding if sequences vary in length,
        # but less critical here since we pad to fixed seq_length.
        Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=seq_length, mask_zero=False),
        LSTM(lstm_units),
        Dropout(0.2), # Add dropout for regularization
        # Output layer size must match vocab_size to predict probabilities for all tokens (incl. 0)
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', # Correct for integer targets
                  metrics=['accuracy'])
    model.summary() # Print model summary
    return model

# --- Main Training Function ---

def train_lstm_model(log_file_path, model_path, embedding_size=64, lstm_units=128, seq_length=DEFAULT_SEQ_LENGTH, epochs=50, batch_size=128):
    """Loads data, prepares sequences, builds, trains, and saves the LSTM model."""
    print(f"Starting LSTM training process...")
    print(f"Log file: '{log_file_path}'")
    print(f"Model output path: '{model_path}'")
    print(f"Sequence length: {seq_length}")

    # 1. Load Access Log
    try:
        with open(log_file_path, 'r') as f:
            # Read and filter out empty lines immediately
            access_sequence = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(access_sequence)} accesses from log file.")
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)

    if len(access_sequence) < seq_length + 1:
        print(f"Error: Not enough data in log file. Need at least {seq_length + 1} accesses.")
        sys.exit(1)

    # 2. Build Vocabulary
    # Consider using min_freq > 1 if your log has many rarely accessed files you want to ignore
    vocab, vocab_size, benchmark_files = build_vocabulary(access_sequence, min_freq=1)

    # 3. Create Training Sequences
    print("Creating sequences from log file...")
    log_sequences, log_next_paths = create_sequences_from_list(access_sequence, seq_length, vocab)
    print(f"Created {len(log_sequences)} sequences from log data.")

    # 4. Generate Synthetic Sequences (Optional but Recommended)
    synth_sequences, synth_next_paths = generate_synthetic_sequences(benchmark_files, seq_length, vocab, num_sequences=len(log_sequences)//2 or 500) # Generate proportional or fixed amount

    # 5. Combine and Prepare Final Data
    all_sequences = log_sequences + synth_sequences
    all_next_paths = log_next_paths + synth_next_paths

    if not all_sequences:
        print("Error: No valid training sequences could be generated.")
        sys.exit(1)

    print(f"Total training sequences: {len(all_sequences)}")

    # Pad sequences (input X)
    X = pad_sequences(all_sequences, maxlen=seq_length, padding='pre', value=PAD_IDX)
    # Convert targets (y) to numpy array
    y = np.array(all_next_paths)

    # 6. Split Data (Train/Validation)
    # Using validation_split in model.fit is okay, but explicit split gives more control
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42) # 15% validation split
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # 7. Build Model
    model = build_lstm_model(vocab_size, embedding_size, lstm_units, seq_length)

    # 8. Train Model
    print(f"Starting training (max epochs={epochs}, batch_size={batch_size})...")
    # Early stopping: monitor validation loss, stop if no improvement after 'patience' epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1) # Set verbose=1 or 2 to see progress

    print("Training finished.")

    # 9. Evaluate (Optional)
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nEvaluation on validation set:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # 10. Save Model and Vocabulary
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    vocab_path = f"{model_path}_vocab.json"
    print(f"Saving vocabulary to {vocab_path}...")
    try:
        with open(vocab_path, 'w') as f:
            # Save vocabulary that maps path -> index (starting from 1)
            json.dump(vocab, f, indent=4)
        print("Model and vocabulary saved successfully.")
    except Exception as e:
        print(f"Error saving vocabulary: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Example Usage: python train_script.py access_log.txt lstm_model.h5
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <access_log.txt> <output_model.h5> [seq_length]")
        sys.exit(1)

    log_path = sys.argv[1]
    model_output_path = sys.argv[2]
    sequence_length = DEFAULT_SEQ_LENGTH
    if len(sys.argv) > 3:
        try:
            sequence_length = int(sys.argv[3])
            if sequence_length <= 0:
                raise ValueError("Sequence length must be positive.")
        except ValueError as e:
            print(f"Error: Invalid sequence length '{sys.argv[3]}'. {e}")
            sys.exit(1)

    train_lstm_model(log_path, model_output_path, seq_length=sequence_length)