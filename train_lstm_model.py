import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense



# Read and map file paths to IDs
with open("access_log.txt", "r") as f:
    paths = [line.strip() for line in f.readlines()]

unique_paths = sorted(set(paths))
path_to_idx = {p: i for i, p in enumerate(unique_paths)}
idx_to_path = {i: p for p, i in path_to_idx.items()}

# Convert paths to numeric sequence
seq = [path_to_idx[p] for p in paths]

# Split sequence
def splitSequence(seq, n_steps):
    X, y = [], []
    for i in range(len(seq)):
        lastIndex = i + n_steps
        if lastIndex > len(seq) - 1:
            break
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 5
X, y = splitSequence(seq, n_steps)

# Build model
model = Sequential()
model.add(Embedding(input_dim=len(unique_paths), output_dim=32, input_length=n_steps))
model.add(LSTM(64))
model.add(Dense(len(unique_paths), activation='softmax'))  # class prediction

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=16)

# Save model + lookup tables
model.save("lstm_cache_model.h5")
with open("path_to_index.pkl", "wb") as f:
    pickle.dump(path_to_idx, f)
with open("index_to_path.pkl", "wb") as f:
    pickle.dump(idx_to_path, f)

