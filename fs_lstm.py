#!/usr/bin/env python3

import os
import sys
import errno
import threading
import time
from fuse import FUSE, FuseOSError, Operations
from stat import S_IFDIR, S_IFREG
from collections import OrderedDict, deque
import json
import numpy as np # Needed for LSTM prediction
import traceback # For printing exception details
import tensorflow as tf # Needed for LSTM model
from tensorflow.keras.preprocessing.sequence import pad_sequences # Needed for LSTM input padding

# --- Constants ---
LOG_FILE = "access_log.txt"             # Logs file accesses
# PROB_PATH = "markov_probabilities.json" # Markov model path (REMOVED)
STATS_FILE = "filesystem_stats.txt"     # Stores the final run statistics
PREFETCH_COUNT = 1                      # How many files to prefetch based on prediction
DEBUG_CACHE = True                      # Enable verbose cache/prefetch logging to console

# --- LSTM Specific Constants ---
LSTM_MODEL_PATH = 'lstm_model.h5'           # Path to the trained LSTM model file
LSTM_VOCAB_PATH = 'lstm_model_vocab.json'   # Path to the model's vocabulary file
LSTM_SEQ_LENGTH = 5                         # Input sequence length for the LSTM model
# ---------------------------

class MemoryFS(Operations):
    def __init__(self):
        self.files = {}  # Stores file metadata {path: {st_mode, st_size, ...}}
        self.data = {}   # Stores file content {path: b'bytes'}
        self.fd = 0      # Simple file descriptor counter
        self.cache = OrderedDict() # LRU Cache {path: b'bytes'}
        self.cache_size = 10       # Max number of items in cache
        self.stats = {             # Dictionary to hold runtime statistics
            'access_count': {},    # {path: count}
            'cache_hits': {},      # {path: count}
            'cache_misses': {},    # {path: count}
            'prefetches': {}       # {path: count}
        }
        now = time.time()
        # Initialize root directory
        root_path = self._normalize_path('/')
        self.files[root_path] = {
            'st_mode': (S_IFDIR | 0o755), 'st_ctime': now, 'st_mtime': now,
            'st_atime': now, 'st_nlink': 2
        }
        self.shutdown_flag = threading.Event() # Used to signal shutdown between threads

        # --- LSTM Related Attributes ---
        self.lstm_model = None                     # Holds the loaded LSTM model
        self.seq_length = LSTM_SEQ_LENGTH          # Sequence length expected by LSTM
        self.recent_paths = deque(maxlen=self.seq_length) # Stores recent normalized paths
        self.vocab = {}                            # Maps path -> index
        self.inverse_vocab = {}                    # Maps index -> path
        self.load_lstm_model()                     # Attempt to load LSTM model on startup
        # --- End LSTM Attributes ---

        # Removed Markov attributes:
        # self.markov_probabilities = {}
        # self.last_accessed_path = None
        # self.load_markov_model()

    # --- Path Normalization Helper ---
    def _normalize_path(self, path):
        """Ensure path starts with a single '/' and has no trailing slash (unless root)."""
        norm_path = '/' + path.lstrip('/')
        if len(norm_path) > 1:
            norm_path = norm_path.rstrip('/')
        return norm_path

    # --- Cache Helper Methods ---
    def _update_cache(self, path, data):
        """Adds/updates an item in the cache, handling LRU eviction. Uses normalized path."""
        norm_path = self._normalize_path(path)
        if norm_path in self.cache:
            self.cache.move_to_end(norm_path)
            # Only update data if it's different? For prefetch, data should be same. For write, it'll differ.
            # Let's assume we always want to update/overwrite here.
            self.cache[norm_path] = data
            if DEBUG_CACHE: print(f"DEBUG: Cache UPDATE for {norm_path} (Size: {len(self.cache)})")
        else:
            if len(self.cache) >= self.cache_size:
                evicted_path, _ = self.cache.popitem(last=False) # Evict oldest
                if DEBUG_CACHE: print(f"DEBUG: Cache EVICTING {evicted_path} (Cache Size: {self.cache_size})")
            self.cache[norm_path] = data
            self.cache.move_to_end(norm_path) # Mark as newest
            if DEBUG_CACHE: print(f"DEBUG: Cache ADD for {norm_path} (New Size: {len(self.cache)})")

    def _get_from_cache(self, path):
        """Retrieves from cache, updates stats, and LRU order. Uses normalized path."""
        norm_path = self._normalize_path(path)
        if norm_path in self.cache:
            self.cache.move_to_end(norm_path) # Mark as recently used
            self.stats['cache_hits'][norm_path] = self.stats['cache_hits'].get(norm_path, 0) + 1
            if DEBUG_CACHE: print(f"DEBUG: Cache HIT for {norm_path} (Total Hits: {self.stats['cache_hits'].get(norm_path, 0)})")
            return self.cache[norm_path]
        else:
            if norm_path in self.data or norm_path in self.files:
                self.stats['cache_misses'][norm_path] = self.stats['cache_misses'].get(norm_path, 0) + 1
                if DEBUG_CACHE: print(f"DEBUG: Cache MISS for {norm_path} (Total Misses: {self.stats['cache_misses'].get(norm_path, 0)})")
            return None # Not in cache

    # --- LSTM Model Loading and Prediction ---
    def load_lstm_model(self):
        """Loads the trained LSTM model and its vocabulary."""
        print("Attempting to load LSTM model and vocabulary...")
        model_loaded = False
        vocab_loaded = False
        self.lstm_model = None # Ensure it starts as None here

        # --- Model Loading ---
        model_load_error = None
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                print(f"DEBUG: Attempting tf.keras.models.load_model('{LSTM_MODEL_PATH}')...")
                loaded_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
                print(f"DEBUG: load_model call finished. Type: {type(loaded_model)}") # Check type
                self.lstm_model = loaded_model # Assign to self.lstm_model
                print(f"DEBUG: Assigned model to self.lstm_model. Current value type: {type(self.lstm_model)}")
                # Check if it's usable (optional basic check)
                # print(loaded_model.summary()) # Uncomment to see summary if load seems ok
                print(f"LSTM model loaded successfully assigned from '{LSTM_MODEL_PATH}'.")
                model_loaded = True
                with open(LSTM_VOCAB_PATH, 'r') as f:
                    loaded_vocab = json.load(f)
                # ====> THIS IS THE LIKELY CULPRIT <====
                # Ensure keys loaded from JSON are normalized before storing in self.vocab
                self.vocab = {self._normalize_path(k): v for k, v in loaded_vocab.items()}
                # =======================================
                self.inverse_vocab = {str(v): k for k, v in self.vocab.items()} # Uses normalized keys from self.vocab
                print(f"DEBUG: self.vocab created. Size: {len(self.vocab)}. Sample: {list(self.vocab.items())[:3]}") # Add this
                print(f"DEBUG: self.inverse_vocab created. Size: {len(self.inverse_vocab)}. Sample: {list(self.inverse_vocab.items())[:3]}") # Add this
                print("LSTM vocabulary loaded successfully...")
                vocab_loaded = True
            except Exception as e:
                print(f"ERROR during LSTM model loading from '{LSTM_MODEL_PATH}': {e}")
                import traceback
                print("--- Traceback ---")
                traceback.print_exc() # Print full error traceback
                print("--- End Traceback ---")
                model_load_error = e
                self.lstm_model = None # Explicitly set to None on error
        else:
            print(f"LSTM model file not found: '{LSTM_MODEL_PATH}'.")
            model_load_error = FileNotFoundError(f"File not found: {LSTM_MODEL_PATH}")

        # --- Vocabulary Loading (Keep as is, or add similar debug prints) ---
        # ... (vocabulary loading code) ...
        # Example check:
        # print(f"DEBUG: Vocabulary loaded. Size: {len(self.vocab)}")

        # --- Final Check ---
        if not model_loaded or not vocab_loaded:
            print("LSTM prefetching disabled due to loading errors.")
            # Ensure model is None if *either* part failed, especially if vocab failed after model seemed okay
            if not model_loaded:
                    print(f"  Reason: Model loading failed ({model_load_error})")
            if not vocab_loaded:
                    print("  Reason: Vocabulary loading failed.")
            self.lstm_model = None # Explicitly set to None if anything failed
        else:
            print("DEBUG: load_lstm_model function finished successfully.")

        # Add a final check of the variable state before function exits
        print(f"DEBUG: Exiting load_lstm_model. self.lstm_model is type: {type(self.lstm_model)}")

    def _predict_and_prefetch_lstm(self):
        print("DEBUG: Entered _predict_and_prefetch_lstm") 
        """Uses LSTM model to predict and prefetch next likely files based on recent history."""
        if not self.lstm_model or len(self.recent_paths) < self.seq_length:
            # Not enough history or model not loaded
            # if DEBUG_CACHE: print(f"DEBUG: LSTM Prefetch skipped (Model: {self.lstm_model is not None}, History: {len(self.recent_paths)}/{self.seq_length})")
            print("NO MODEL!!!!")
            return

        # Convert recent paths (normalized) to sequence of indices
        # Use 0 or a specific index for paths not in vocab (Out-Of-Vocabulary)
        # Assuming index 0 might represent padding or OOV depending on training
        current_sequence_indices = [self.vocab.get(p, 0) for p in self.recent_paths]

        # Pad sequence (though deque with mafxlen should ensure length, it's good practice)
        # The model expects input shape like (batch_size, seq_length), so wrap in another list
        padded_sequence = pad_sequences([current_sequence_indices], maxlen=self.seq_length, padding='pre', value = 0) # 'pre' or 'post' depending on training

        if DEBUG_CACHE: print(f"DEBUG: LSTM Predicting based on sequence indices: {padded_sequence}")

        try:
            # Get model predictions (output probabilities for each item in vocab)
            predictions = self.lstm_model.predict(padded_sequence, verbose=0)[0] # Get prediction for the first (only) batch item

            # Get indices of top N predictions (highest probability)
            # Argsort returns indices from lowest to highest, so take the last N
            top_indices = np.argsort(predictions)[-PREFETCH_COUNT:]

            if DEBUG_CACHE:
                # Show top few predicted indices and their probabilities
                top_preds_debug = [(idx, f"{predictions[idx]:.4f}") for idx in reversed(top_indices)] # Show highest first
                print(f"DEBUG: LSTM Top {PREFETCH_COUNT} predicted indices (index, prob): {top_preds_debug}")

            prefetched_count = 0
            # Iterate from highest probability downwards
            for index in reversed(top_indices):
                if prefetched_count >= PREFETCH_COUNT: break

                predicted_path = self.inverse_vocab.get(str(index), None) # Lookup index (as string) in inverse vocab

                if predicted_path and predicted_path != self._normalize_path('/'): # Ensure prediction is a valid, known path and not root
                    in_data = predicted_path in self.data
                    in_cache = predicted_path in self.cache

                    if DEBUG_CACHE: print(f"DEBUG: LSTM Trying prefetch {predicted_path} (Index: {index}, Prob: {predictions[index]:.4f}). In data? {in_data}. In cache? {in_cache}")

                    # Condition: File must exist in our main data store and not already be in cache
                    if in_data and not in_cache:
                        if DEBUG_CACHE: print(f"DEBUG: ---> LSTM Prefetching {predicted_path}")
                        data_to_cache = self.data[predicted_path]
                        self._update_cache(predicted_path, data_to_cache) # Add to cache
                        self.stats['prefetches'][predicted_path] = self.stats['prefetches'].get(predicted_path, 0) + 1
                        prefetched_count += 1
                    # else:
                        # if DEBUG_CACHE: print(f"DEBUG: ---> LSTM Skipping prefetch for {predicted_path}")
                # else:
                    # if DEBUG_CACHE and index != 0: print(f"DEBUG: LSTM Predicted index {index} not found in inverse vocab or was root.")

        except Exception as e:
            print(f"Error during LSTM prediction or prefetching: {e}")
            traceback.print_exc() # Print full traceback for debugging

    def _log_access_lstm(self, current_path):
        """Logs access, updates recent path list, and triggers LSTM prediction."""
        norm_path = self._normalize_path(current_path)

        # Don't log or predict based on root accesses if not desired
        if norm_path == '/': return

        # Log access to file
        try:
            log_dir = os.path.dirname(LOG_FILE)
            if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
            with open(LOG_FILE, "a") as log: log.write(f"{norm_path}\n")
        except Exception as e: print(f"Error writing to access log '{LOG_FILE}': {e}")

        # Add normalized path to recent history (deque handles max length)
        self.recent_paths.append(norm_path)
        if DEBUG_CACHE: print(f"DEBUG: LSTM History Updated. Current: {list(self.recent_paths)}")

        # Trigger prediction based on the *updated* sequence
        self._predict_and_prefetch_lstm()

    # --- Filesystem Operations ---
    def _is_ignored_path(self, path):
        """Check if the path corresponds to an ignorable macOS metadata file."""
        basename = os.path.basename(path)
        if basename.startswith('._'): return True
        if basename == '.DS_Store': return True
        return False

    # --- Methods Modified for Path Normalization and Ignoring ._* ---
    def getattr(self, path, fh=None):
        if self._is_ignored_path(path): raise FuseOSError(errno.ENOENT)
        norm_path = self._normalize_path(path)
        if norm_path not in self.files: raise FuseOSError(errno.ENOENT)
        return self.files[norm_path]

    def access(self, path, amode):
        if self._is_ignored_path(path): raise FuseOSError(errno.ENOENT)
        norm_path = self._normalize_path(path)
        if norm_path not in self.files: raise FuseOSError(errno.ENOENT)
        return 0 # Success

    def readdir(self, path, fh):
        norm_path = self._normalize_path(path)
        dirents = ['.', '..']
        all_entries_in_dir = set()
        for entry_path in self.files: # Keys are normalized
            if entry_path == '/': continue
            parent_dir = self._normalize_path(os.path.dirname(entry_path)) # Ensure dirname result is normalized
            if parent_dir == norm_path:
                base_name = os.path.basename(entry_path)
                if not self._is_ignored_path(base_name): # Check basename
                    all_entries_in_dir.add(base_name)
        dirents.extend(sorted(list(all_entries_in_dir)))
        self.stats['access_count'][norm_path] = self.stats['access_count'].get(norm_path, 0) + 1
        # Optionally log directory access for LSTM if needed, but usually focus on files
        # self._log_access_lstm(norm_path)
        return dirents

    def create(self, path, mode, fi=None):
        if self._is_ignored_path(path): raise FuseOSError(errno.EACCES)
        norm_path = self._normalize_path(path)
        if norm_path == '/': raise FuseOSError(errno.EISDIR)
        if norm_path in self.files: raise FuseOSError(errno.EEXIST)

        now = time.time()
        uid, gid = os.getuid(), os.getgid()
        self.files[norm_path] = { 'st_mode': (S_IFREG | mode), 'st_nlink': 1, 'st_size': 0,
                                'st_ctime': now, 'st_mtime': now, 'st_atime': now, 'st_uid': uid, 'st_gid': gid }
        self.data[norm_path] = b''
        parent_path_norm = self._normalize_path(os.path.dirname(norm_path))
        # Check parent existence and type before updating times
        if parent_path_norm in self.files and (self.files[parent_path_norm]['st_mode'] & S_IFDIR):
            self.files[parent_path_norm]['st_mtime'] = now
            self.files[parent_path_norm]['st_atime'] = now
            self.files[parent_path_norm]['st_nlink'] = self.files[parent_path_norm].get('st_nlink', 2) # Safety check? mkdir handles link count

        self._update_cache(norm_path, b'') # Cache the empty file
        self.stats['access_count'][norm_path] = self.stats['access_count'].get(norm_path, 0) + 1
        self._log_access_lstm(norm_path) # Log access which triggers prediction
        self.fd += 1
        return self.fd

    def open(self, path, flags):
        # access() or getattr() will have already handled ignored paths and existence
        norm_path = self._normalize_path(path)
        # Simplified access check for open flags could be added here if needed
        self.stats['access_count'][norm_path] = self.stats['access_count'].get(norm_path, 0) + 1
        self.files[norm_path]['st_atime'] = time.time() # Update access time on open
        # Log access on open? Could lead to different prediction patterns than logging on read/write.
        # Let's stick to logging on read/write for now unless desired otherwise.
        # self._log_access_lstm(norm_path)
        self.fd += 1
        return self.fd

    def read(self, path, length, offset, fh):
        norm_path = self._normalize_path(path)
        if DEBUG_CACHE: print(f"DEBUG: Entering READ for {norm_path} (offset={offset}, length={length})")

        if norm_path not in self.files or (self.files[norm_path]['st_mode'] & S_IFDIR):
            raise FuseOSError(errno.EIO) # Should not happen with valid fh

        # --- Log access FIRST (before cache check) to ensure history is up-to-date for prediction ---
        self.stats['access_count'][norm_path] = self.stats['access_count'].get(norm_path, 0) + 1
        self._log_access_lstm(norm_path) # Log access which triggers prediction
        # ------------------------------------------------------------------------------------------

        cached_data = self._get_from_cache(norm_path) # Updates hit/miss stats & LRU

        if cached_data is not None:
            data_to_read = cached_data
            if DEBUG_CACHE: print(f"DEBUG: READ served from cache for {norm_path}")
        else:
            # Miss occurred (and was counted in _get_from_cache if file exists)
            data_to_read = self.data.get(norm_path, None)
            if data_to_read is None:
                print(f"ERROR: Read requested for path '{norm_path}' but not found in self.data!")
                raise FuseOSError(errno.EIO)
            if DEBUG_CACHE: print(f"DEBUG: READ cache miss for {norm_path}, fetching from main store.")
            self._update_cache(norm_path, data_to_read) # Cache the data on miss

        self.files[norm_path]['st_atime'] = time.time() # Update access time
        result_data = data_to_read[offset:offset + length]
        if DEBUG_CACHE: print(f"DEBUG: Exiting READ for {norm_path} (returning {len(result_data)} bytes)")
        return result_data

    def write(self, path, buf, offset, fh):
        norm_path = self._normalize_path(path)
        if DEBUG_CACHE: print(f"DEBUG: Entering WRITE for {norm_path} (offset={offset}, length={len(buf)})")

        if norm_path not in self.files or not (self.files[norm_path]['st_mode'] & S_IFREG):
            if norm_path in self.files and (self.files[norm_path]['st_mode'] & S_IFDIR): raise FuseOSError(errno.EISDIR)
            raise FuseOSError(errno.EIO) # Should not happen with valid fh

        # --- Log access FIRST ---
        self.stats['access_count'][norm_path] = self.stats['access_count'].get(norm_path, 0) + 1
        self._log_access_lstm(norm_path) # Log access which triggers prediction
        # ------------------------

        # Check cache (updates LRU, counts hit if present) before modifying data
        # We need to invalidate/update the cache *after* the write completes.
        # _get_from_cache here mainly serves to update LRU if it was a hit before write.
        self._get_from_cache(norm_path)

        current_data_main = self.data.get(norm_path, b'') # Write based on main data

        # Perform the write operation logic
        data_len = len(current_data_main)
        write_end = offset + len(buf)
        if offset > data_len:
            # Writing past the end with a gap - fill gap with null bytes
            new_data = current_data_main + b'\0' * (offset - data_len) + buf
        elif write_end > data_len:
            # Overwriting end and extending
            new_data = current_data_main[:offset] + buf
        else:
            # Overwriting existing data within bounds
            new_data = current_data_main[:offset] + buf + current_data_main[write_end:]

        self.data[norm_path] = new_data # Update main store first
        self._update_cache(norm_path, new_data) # Update cache with the NEW data

        # Update metadata
        self.files[norm_path]['st_size'] = len(new_data)
        now = time.time()
        self.files[norm_path]['st_mtime'] = now
        self.files[norm_path]['st_atime'] = now # Write also updates atime

        if DEBUG_CACHE: print(f"DEBUG: Exiting WRITE for {norm_path} (new size={len(new_data)})")
        return len(buf)

    def unlink(self, path):
        if self._is_ignored_path(path): raise FuseOSError(errno.ENOENT)
        norm_path = self._normalize_path(path)
        if norm_path == '/': raise FuseOSError(errno.EISDIR)
        if norm_path not in self.files: raise FuseOSError(errno.ENOENT)
        if (self.files[norm_path]['st_mode'] & S_IFDIR): raise FuseOSError(errno.EISDIR)

        # Remove data first
        if norm_path in self.data: del self.data[norm_path]
        if norm_path in self.cache:
            if DEBUG_CACHE: print(f"DEBUG: Removing {norm_path} from cache due to unlink.")
            del self.cache[norm_path]
        # Remove stats entries
        for stat_type in list(self.stats.keys()):
            if norm_path in self.stats[stat_type]: del self.stats[stat_type][norm_path]
        # Remove from recent paths history if present (though this might slightly alter prediction)
        # Optional: remove occurrences from self.recent_paths
        # temp_deque = deque(maxlen=self.seq_length)
        # for p in self.recent_paths:
        #     if p != norm_path:
        #         temp_deque.append(p)
        # self.recent_paths = temp_deque
        # Consider if removing history is desired or if keeping it reflects past reality better.
        # For simplicity, let's not remove it from history for now.

        # Remove file metadata last
        del self.files[norm_path]

        # Update parent directory times and link count
        parent_path_norm = self._normalize_path(os.path.dirname(norm_path))
        if parent_path_norm in self.files and (self.files[parent_path_norm]['st_mode'] & S_IFDIR):
            now = time.time()
            self.files[parent_path_norm]['st_mtime'] = now
            self.files[parent_path_norm]['st_atime'] = now
            # Decrement link count of parent (for the entry being removed) - nlink is handled by mkdir/rmdir for subdirs
            # For files, only the file's own nlink matters usually (starts at 1).

        return 0 # Success for unlink

    def rmdir(self, path):
        if self._is_ignored_path(path): raise FuseOSError(errno.ENOENT)
        norm_path = self._normalize_path(path)
        if norm_path == '/': raise FuseOSError(errno.EBUSY) # Cannot remove root
        if norm_path not in self.files: raise FuseOSError(errno.ENOENT)
        if not (self.files[norm_path]['st_mode'] & S_IFDIR): raise FuseOSError(errno.ENOTDIR)

        # Check if directory is empty (ignoring ._* files potentially)
        is_empty = True
        for entry_path in self.files:
            # Check if entry_path is a direct child of norm_path
            if entry_path != '/' and self._normalize_path(os.path.dirname(entry_path)) == norm_path:
                # Now check if this child should be ignored
                base_name = os.path.basename(entry_path)
                if not self._is_ignored_path(base_name):
                    is_empty = False
                    break # Found a non-ignored entry, directory is not empty

        if not is_empty:
            raise FuseOSError(errno.ENOTEMPTY)

        # If empty, proceed with removal
        del self.files[norm_path] # Remove dir metadata
        # Clean up stats (less critical for dirs)
        for stat_type in list(self.stats.keys()):
            if norm_path in self.stats[stat_type]: del self.stats[stat_type][norm_path]
        # Remove from recent paths history? Less likely for dirs, same logic as unlink applies.

        # Update parent directory metadata
        parent_path_norm = self._normalize_path(os.path.dirname(norm_path))
        if parent_path_norm in self.files and (self.files[parent_path_norm]['st_mode'] & S_IFDIR):
            now = time.time()
            self.files[parent_path_norm]['st_nlink'] -= 1 # Link count decreases as subdir is removed
            self.files[parent_path_norm]['st_mtime'] = now
            self.files[parent_path_norm]['st_atime'] = now

        return 0 # Success for rmdir

    def mkdir(self, path, mode):
        if self._is_ignored_path(path): raise FuseOSError(errno.EACCES)
        norm_path = self._normalize_path(path)
        if norm_path in self.files: raise FuseOSError(errno.EEXIST)
        parent_path_norm = self._normalize_path(os.path.dirname(norm_path))

        if parent_path_norm not in self.files or not (self.files[parent_path_norm]['st_mode'] & S_IFDIR):
            raise FuseOSError(errno.ENOENT) # Parent directory doesn't exist or isn't a directory

        # Create the new directory entry
        now = time.time()
        uid, gid = os.getuid(), os.getgid()
        self.files[norm_path] = {
            'st_mode': (S_IFDIR | mode), 'st_nlink': 2, # A new directory has 2 links: one for itself, one for '..' from parent
            'st_size': 0, # Directories have size 0 or block size depending on system; 0 is common
            'st_ctime': now, 'st_mtime': now, 'st_atime': now, 'st_uid': uid, 'st_gid': gid
        }

        # Update parent directory's link count and times
        self.files[parent_path_norm]['st_nlink'] += 1
        self.files[parent_path_norm]['st_mtime'] = now
        self.files[parent_path_norm]['st_atime'] = now

        return 0 # Success for mkdir

    def truncate(self, path, length, fh=None):
        if self._is_ignored_path(path): raise FuseOSError(errno.EACCES)
        norm_path = self._normalize_path(path)
        if norm_path not in self.files or not (self.files[norm_path]['st_mode'] & S_IFREG):
            if norm_path in self.files and (self.files[norm_path]['st_mode'] & S_IFDIR): raise FuseOSError(errno.EISDIR)
            raise FuseOSError(errno.EINVAL) # Use EINVAL for truncate on non-file

        self.stats['access_count'][norm_path] = self.stats['access_count'].get(norm_path, 0) + 1
        # Log truncate access? Similar reasoning to open - maybe not critical for prediction model.
        # self._log_access_lstm(norm_path)

        # Get data, preferring cache but falling back to main store
        current_data = self._get_from_cache(norm_path) # Updates LRU if hit
        if current_data is None:
            current_data = self.data.get(norm_path, b'')
            # No need to cache here unless we want to count this as a 'read' for caching purposes

        # Perform truncation
        if length < len(current_data):
            new_data = current_data[:length]
        else:
            new_data = current_data + b'\0' * (length - len(current_data))

        self.data[norm_path] = new_data # Update main store
        self._update_cache(norm_path, new_data) # Update cache

        # Update metadata
        self.files[norm_path]['st_size'] = length
        now = time.time()
        self.files[norm_path]['st_mtime'] = now
        self.files[norm_path]['st_atime'] = now

        return 0 # Success for truncate

    def statfs(self, path):
        # Provide somewhat realistic stats based on current memory usage
        total_data_size = sum(len(d) for d in self.data.values())
        total_meta_size_est = len(self.files) * 128 # Rough estimate metadata size per entry
        total_size = total_data_size + total_meta_size_est

        block_size = 4096
        # Simulate a larger total capacity, e.g., 1GB
        total_blocks = (1024 * 1024 * 1024) // block_size
        used_blocks = -(-total_size // block_size) # Ceiling division
        free_blocks = max(0, total_blocks - used_blocks)
        # Estimate inodes (files/dirs)
        total_inodes = total_blocks # Simplistic: assume max 1 file per block potential
        used_inodes = len(self.files)
        free_inodes = max(0, total_inodes - used_inodes)


        return dict(f_bsize=block_size, f_frsize=block_size, # Fragment size, usually same as block size
                    f_blocks=total_blocks, f_bfree=free_blocks, f_bavail=free_blocks, # Total, free, avail blocks
                    f_files=total_inodes, f_ffree=free_inodes, # Total, free inodes (file entries)
                    f_favail=free_inodes, # Available inodes for non-root
                    f_namemax=255, # Max filename length
                    f_flag=0) # Mount flags (e.g., ST_NOSUID, ST_NODEV, ST_RDONLY) - 0 is common default


    def utimens(self, path, times=None):
        norm_path = self._normalize_path(path)
        if norm_path not in self.files: raise FuseOSError(errno.ENOENT)
        now = time.time()
        # times is tuple (atime, mtime) in nanoseconds or None
        if times:
            # FUSE passes ns, but Python time() is seconds. Convert ns to seconds.
            atime_ns, mtime_ns = times
            atime = atime_ns / 1e9
            mtime = mtime_ns / 1e9
        else:
            atime, mtime = now, now

        self.files[norm_path]['st_atime'] = atime
        self.files[norm_path]['st_mtime'] = mtime
        # ctime is 'change time' - typically updated when metadata *or* content changes.
        # utimens traditionally only updates atime/mtime, but some systems might update ctime too.
        # Let's update ctime as well for consistency with other modifications.
        self.files[norm_path]['st_ctime'] = now
        return 0


# --- Shared Shutdown Logic ---
# (Shutdown logic remains the same, no changes needed here)
def perform_shutdown_tasks(fs, mountpoint, stats_file_path, triggered_by="Unknown"):
    """Writes stats and attempts to unmount."""
    # Prevent potential race conditions or duplicate calls
    if not hasattr(perform_shutdown_tasks, "shutdown_in_progress"):
        perform_shutdown_tasks.shutdown_in_progress = False
    if perform_shutdown_tasks.shutdown_in_progress:
        print(f"Shutdown already in progress, skipping call by {triggered_by}.")
        return
    perform_shutdown_tasks.shutdown_in_progress = True

    print(f"\nPerforming shutdown tasks (triggered by {triggered_by})...")
    # --- Write Stats to File ---
    stats = {
        'access_count': dict(fs.stats['access_count']),
        'cache_hits': dict(fs.stats['cache_hits']),
        'cache_misses': dict(fs.stats['cache_misses']),
        'prefetches': dict(fs.stats['prefetches'])
    }
    all_paths_norm = set(stats['access_count'].keys()) \
            .union(set(stats['cache_hits'].keys())) \
            .union(set(stats['cache_misses'].keys())) \
            .union(set(stats['prefetches'].keys())) \
            .union(set(f for f in fs.files.keys() if f != '/'))

    try:
        with open(stats_file_path, 'w') as stats_f:
            stats_f.write("Filesystem Statistics:\n")
            stats_f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            stats_f.write(f"Cache Size: {fs.cache_size}\n")
            stats_f.write(f"Prefetch Strategy: LSTM (SeqLen={fs.seq_length}, Count={PREFETCH_COUNT})\n") # Added strategy info
            stats_f.write(f"Triggered By: {triggered_by}\n")
            stats_f.write("-" * 80 + "\n")
            stats_f.write(f"{'Path':<40} {'Accesses':>10} {'Hits':>8} {'Misses':>8} {'Prefetches':>10}\n")
            stats_f.write("=" * 80 + "\n")

            if not all_paths_norm: stats_f.write("  (No activity recorded)\n")
            else:
                total_accesses = 0; total_hits = 0; total_misses = 0; total_prefetches = 0
                for path in sorted(list(all_paths_norm)):
                    acc = stats['access_count'].get(path, 0); hit = stats['cache_hits'].get(path, 0)
                    miss = stats['cache_misses'].get(path, 0); pref = stats['prefetches'].get(path, 0)
                    is_dir = path in fs.files and (fs.files[path]['st_mode'] & S_IFDIR)
                    path_display = f"{path}/" if is_dir else path
                    # Only print rows with some activity
                    if acc > 0 or hit > 0 or miss > 0 or pref > 0:
                        stats_f.write(f"{path_display:<40} {acc:>10} {hit:>8} {miss:>8} {pref:>10}\n")
                        total_accesses += acc; total_hits += hit; total_misses += miss; total_prefetches += pref
                stats_f.write("=" * 80 + "\n")
                stats_f.write(f"{'TOTALS':<40} {total_accesses:>10} {total_hits:>8} {total_misses:>8} {total_prefetches:>10}\n")
                cache_interactions = total_hits + total_misses
                if cache_interactions > 0:
                    hit_rate = (total_hits / cache_interactions) * 100
                    stats_f.write(f"\nOverall Cache Hit Rate: {hit_rate:.2f}% ({total_hits} Hits / {cache_interactions} Interactions)\n")
                else: stats_f.write("\nOverall Cache Hit Rate: N/A (No cache hits or misses)\n")
        print(f"Filesystem statistics written to '{stats_file_path}'.")
    except IOError as e: print(f"Error: Could not write statistics to file '{stats_file_path}': {e}")

    # --- Unmounting Logic ---
    print("\nAttempting to unmount...")
    time.sleep(0.2) # Short pause before unmount
    # Determine unmount command based on platform
    umount_cmd = f"fusermount -u {mountpoint}" # Linux default
    if sys.platform == 'darwin': # macOS
        umount_cmd = f"umount {mountpoint}"
    elif sys.platform == 'win32': # Windows (requires winfsp or similar) - Placeholder
        # Unmounting might need specific library calls or commands on Windows
        print("Warning: Automatic unmounting on Windows is not implemented in this script.")
        umount_cmd = None # Indicate unmount command is not standard

    result = -1 # Default to error
    if umount_cmd:
        result = os.system(umount_cmd)
        retry_count = 0
        while result != 0 and retry_count < 3: # Retry unmount a few times if it fails
            retry_count += 1
            print(f"Warning: Failed to unmount {mountpoint} (cmd: '{umount_cmd}'). Retrying ({retry_count}/3)...")
            time.sleep(retry_count) # Wait longer each time
            result = os.system(umount_cmd)
    else:
        print(f"Skipping automatic unmount for platform {sys.platform}.")


    if result == 0: print(f"Successfully unmounted {mountpoint}.")
    elif umount_cmd: print(f"ERROR: Failed to unmount {mountpoint} after multiple attempts.")

    # Reset flag after completion
    perform_shutdown_tasks.shutdown_in_progress = False


# --- Shutdown Monitor Thread ---
# (Shutdown monitor remains the same, no changes needed here)
def shutdown_monitor(fs, mountpoint):
    stats_file_path = os.path.join(os.getcwd(), STATS_FILE)
    shutdown_file_path = os.path.join(os.getcwd(), "shutdown")
    monitor_name = threading.current_thread().name
    if not monitor_name: monitor_name = "ShutdownMonitorThread" # Default name

    while not fs.shutdown_flag.is_set():
        if os.path.exists(shutdown_file_path):
            print(f"\nShutdown signal received via file ({monitor_name})...")
            if not fs.shutdown_flag.is_set(): # Check again before acting
                fs.shutdown_flag.set() # Set flag immediately
                # Call shutdown tasks (which handles unmount and stats)
                perform_shutdown_tasks(fs, mountpoint, stats_file_path, triggered_by=f"{monitor_name} (File Trigger)")
                try:
                    if os.path.exists(shutdown_file_path): os.remove(shutdown_file_path)
                except OSError as e: print(f"Warning: Could not remove shutdown file {shutdown_file_path}: {e}")
                print(f"Exiting FUSE process via {monitor_name}.")
                # Use os._exit to force exit from this thread, as FUSE might be blocking main
                os._exit(0)
            else:
                # Shutdown was already initiated by another thread (e.g., main finally block after Ctrl+C)
                print(f"{monitor_name} detected shutdown already in progress, exiting loop.")
                break # Exit the loop gracefully
        try:
            # Check more frequently without busy-waiting too much
            fs.shutdown_flag.wait(timeout=0.5) # Wait for event or timeout
        except KeyboardInterrupt:
            # This thread might catch Ctrl+C if it's active during the signal
            print(f"\n{monitor_name} received KeyboardInterrupt.")
            if not fs.shutdown_flag.is_set():
                fs.shutdown_flag.set() # Ensure flag is set if Ctrl+C is caught here
                # Let main thread handle cleanup via its finally block if possible
                # perform_shutdown_tasks(fs, mountpoint, stats_file_path, triggered_by=f"{monitor_name} (KeyboardInterrupt)")
            break # Exit loop


# --- Main Function ---
# (Main function structure remains largely the same)
def main(mountpoint):
    abs_mountpoint = os.path.abspath(mountpoint)
    if not os.path.isdir(abs_mountpoint):
        try:
            print(f"Mountpoint '{abs_mountpoint}' doesn't exist. Creating it.")
            os.makedirs(abs_mountpoint)
        except OSError as e:
            print(f"Error: Could not create mountpoint directory '{abs_mountpoint}': {e}", file=sys.stderr)
            sys.exit(1)

    fs = MemoryFS() # Initialize MemoryFS with LSTM components
    print(f"Mounting LSTM-Prefetch FS filesystem at {abs_mountpoint}...")
    print(f"Log file: {os.path.abspath(LOG_FILE)}")
    print(f"LSTM Model: {os.path.abspath(LSTM_MODEL_PATH)}")
    print(f"LSTM Vocab: {os.path.abspath(LSTM_VOCAB_PATH)}")
    print(f"Statistics will be saved to: {os.path.abspath(STATS_FILE)}")
    if DEBUG_CACHE: print("****** CACHE DEBUGGING ENABLED ******")
    print("Ready for operations.")
    # Assuming you have a benchmark script:
    # print(f"Run benchmark: python3 your_benchmark_script.py {mountpoint}")
    print("To stop: Create a file named 'shutdown' in the current directory or press Ctrl+C.")

    # Start the shutdown monitor thread
    monitor_thread = threading.Thread(target=shutdown_monitor, args=(fs, abs_mountpoint), name="ShutdownMonitorThread", daemon=True)
    monitor_thread.start()

    # FUSE options
    # foreground=True: Keep FUSE in the foreground
    # nothreads=False: Allow FUSE to use multiple threads (potentially better performance, but need thread-safe Operations)
    #                Set to True if Operations are not thread-safe. Our current implementation uses shared dicts/lists
    #                without explicit locks, so nothreads=True might be safer, although performance might suffer.
    #                Let's try False first, assuming Python's GIL provides some safety for basic dict ops,
    #                but be mindful of potential race conditions in complex updates (like cache LRU).
    #                For simplicity and safety, let's revert to nothreads=True unless performance is paramount and thread safety is verified.
    # allow_other=False: Restrict access to the mounting user only (safer default)
    fuse_kwargs = dict(foreground=True, nothreads=True, allow_other=False)
    stats_file_path = os.path.join(os.getcwd(), STATS_FILE) # Define here for finally block

    try:
        print(f"Starting FUSE main loop... (PID: {os.getpid()}) Press Ctrl+C to interrupt.")
        # Pass the MemoryFS instance, mountpoint, and keyword arguments to FUSE
        FUSE(fs, abs_mountpoint, **fuse_kwargs)
        # This line is reached if FUSE exits *normally*, e.g., unmounted externally AND foreground=False (unlikely with our setup)
        # Or if the mount fails very early before the main loop truly starts.
        print(f"FUSE main loop finished.") # Usually not reached if foreground=True and shutdown via signal/file

    except BaseException as e: # Catch BaseException including KeyboardInterrupt and SystemExit
        # This block catches Ctrl+C (KeyboardInterrupt) or other unexpected errors during FUSE setup/runtime
        print(f"\n*** FUSE MAIN LOOP EXCEPTION: {type(e).__name__}: {e} ***", file=sys.stderr)
        if isinstance(e, SystemExit):
            print("SystemExit caught, likely from os._exit in monitor thread.", file=sys.stderr)
            # Don't re-raise SystemExit, allow finally block to run if needed, but monitor likely handled it.
        elif isinstance(e, Exception): # Print traceback only for actual programming errors
            print("Traceback:", file=sys.stderr)
            traceback.print_exc()
        # Ensure shutdown flag is set upon any main loop termination
        if not fs.shutdown_flag.is_set():
            fs.shutdown_flag.set()

    finally:
        # This block executes reliably when the try block exits,
        # either normally, via exception, or KeyboardInterrupt (unless os._exit was called).
        print("Entering main finally block...")
        # Attempt shutdown tasks ONLY if the flag wasn't already set and handled by the monitor thread.
        # The perform_shutdown_tasks function itself has a guard against multiple executions.
        if not perform_shutdown_tasks.shutdown_in_progress:
            print("Initiating cleanup from main finally block...")
            # Ensure flag is set before calling tasks from here
            fs.shutdown_flag.set()
            perform_shutdown_tasks(fs, abs_mountpoint, stats_file_path, triggered_by="MainThreadFinally")
        else:
            print("Shutdown already handled or in progress (likely by monitor thread).")

        # Wait briefly for the monitor thread to potentially finish its own exit/cleanup if needed
        # This might help ensure messages aren't interleaved badly in the console.
        if monitor_thread.is_alive():
            # print("Waiting for monitor thread to exit...") # Optional: wait for monitor
            monitor_thread.join(timeout=1.0) # Wait max 1 second

        print("Exiting main function.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mountpoint>")
        sys.exit(1)
    main(sys.argv[1])