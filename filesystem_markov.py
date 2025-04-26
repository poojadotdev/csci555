#!/usr/bin/env python3

import os
import sys
import errno
import threading
import time
from fuse import FUSE, FuseOSError, Operations
from stat import S_IFDIR, S_IFREG
from collections import OrderedDict, deque # Kept deque just in case for future Order-k
import json
import numpy as np # Still useful for sorting probabilities if needed

# --- Constants ---
LOG_FILE = "access_log.txt"
PROB_PATH = "markov_probabilities.json" # Path to saved probabilities
PREFETCH_COUNT = 1                      # How many top predictions to prefetch
# -----------------

class MemoryFS(Operations):
    def __init__(self):
        self.files = {}
        self.data = {}
        self.fd = 0
        self.cache = OrderedDict()
        self.cache_size = 10 # Example size - Adjust if needed
        self.stats = {
            'access_count': {},
            'cache_hits': {},
            'cache_misses': {},
            'prefetches': {} # Track prefetched items
        }
        now = time.time()
        self.files['/'] = {
            'st_mode': (S_IFDIR | 0o755),
            'st_ctime': now,
            'st_mtime': now,
            'st_atime': now,
            'st_nlink': 2
        }
        self.shutdown_flag = threading.Event()

        # --- Markov Model Cache Initialization ---
        self.markov_probabilities = {}
        self.last_accessed_path = None # Store the path of the last accessed file for prediction
        self.load_markov_model()
        # ---------------------------------------

    def load_markov_model(self):
        """Loads the trained Markov model probabilities."""
        print("Attempting to load Markov model probabilities...")
        if os.path.exists(PROB_PATH):
            try:
                with open(PROB_PATH, 'r') as f:
                    self.markov_probabilities = json.load(f)
                print(f"Markov probabilities loaded successfully from '{PROB_PATH}'.")
                print(f"Model knows transitions from {len(self.markov_probabilities)} files.")
            except Exception as e:
                print(f"Error loading Markov probabilities: {e}")
                self.markov_probabilities = {} # Disable Markov if loading fails
        else:
            print(f"Markov probabilities file not found: '{PROB_PATH}'. Markov prefetching disabled.")
            self.markov_probabilities = {}

    def _update_cache(self, path, data):
        """Adds/updates an item in the cache, handling LRU eviction."""
        if path in self.cache:
            self.cache.move_to_end(path)
            self.cache[path] = data
        else:
            if len(self.cache) >= self.cache_size:
                evicted_path, _ = self.cache.popitem(last=False)
                # print(f"Cache full. Evicted (LRU): {evicted_path}") # Optional: for debugging
            self.cache[path] = data
            self.cache.move_to_end(path)

    def _get_from_cache(self, path):
        """Retrieves from cache, updates stats, and LRU order."""
        if path in self.cache:
            self.cache.move_to_end(path)
            self.stats['cache_hits'][path] = self.stats['cache_hits'].get(path, 0) + 1
            return self.cache[path]
        self.stats['cache_misses'][path] = self.stats['cache_misses'].get(path, 0) + 1
        return None

    def _predict_and_prefetch_markov(self):
        """Uses Markov model to predict and prefetch next likely files."""
        if not self.markov_probabilities or self.last_accessed_path is None:
            return # No model or no history

        next_file_probs = self.markov_probabilities.get(self.last_accessed_path, {})
        if not next_file_probs:
            return # No known transitions from the last file

        sorted_predictions = sorted(next_file_probs.items(), key=lambda item: item[1], reverse=True)

        prefetched_count = 0
        for predicted_path, probability in sorted_predictions:
            if prefetched_count >= PREFETCH_COUNT:
                break
            if predicted_path in self.data and predicted_path not in self.cache:
                # print(f"Markov Prefetching: {predicted_path} (Prob: {probability:.2f}) based on previous: {self.last_accessed_path}")
                data_to_cache = self.data[predicted_path]
                self._update_cache(predicted_path, data_to_cache) # Add to cache
                self.stats['prefetches'][predicted_path] = self.stats['prefetches'].get(predicted_path, 0) + 1
                prefetched_count += 1

    def _log_access_markov(self, current_path):
        """Logs access to file and updates state for Markov prediction."""
        try:
            # Ensure the directory for the log file exists (optional, good practice)
            # log_dir = os.path.dirname(LOG_FILE)
            # if log_dir and not os.path.exists(log_dir):
            #     os.makedirs(log_dir)
            with open(LOG_FILE, "a") as log:
                log.write(f"{current_path}\n")
        except Exception as e:
            print(f"Error writing to access log '{LOG_FILE}': {e}")

        previous_path_for_pred = self.last_accessed_path
        self.last_accessed_path = current_path # Update state

        if previous_path_for_pred is not None:
            self._predict_and_prefetch_markov() # Call prediction

    # --- Filesystem Operations ---
    # (Includes implementations for access, getattr, create, open, read, write,
    #  truncate, unlink, rmdir, mkdir, utimens, readdir, statfs - see previous
    #  answers for the full code for each method. Ensure relevant methods like
    #  read, write, create call self._log_access_markov(path) and others like
    #  unlink, rmdir clean up state/stats if necessary)

    # --- Minimal required methods for benchmark ---
    def getattr(self, path, fh=None):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        return self.files[path]

    def readdir(self, path, fh):
        # Simplified readdir for benchmark's os.listdir
        dirents = ['.', '..']
        if path == '/':
            # Add top-level file/dir names (excluding leading '/')
            dirents.extend(name[1:] for name in self.files if '/' in name[1:] and name != '/')
            dirents.extend(name[1:] for name in self.files if '/' not in name[1:] and name != '/') # files in root
        else:
            prefix = path.rstrip('/') + '/'
            dirents.extend(entry[len(prefix):].split('/')[0] for entry in self.files if entry.startswith(prefix))
            # Remove duplicates that might arise from files vs dirs within path
            dirents = list(OrderedDict.fromkeys(dirents))


        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        return dirents

    def create(self, path, mode, fi=None):
        now = time.time()
        self.files[path] = {
            'st_mode': (S_IFREG | mode), 'st_ctime': now, 'st_mtime': now,
            'st_atime': now, 'st_nlink': 1, 'st_size': 0
        }
        self.data[path] = b''
        self._update_cache(path, b'') # Add empty file to cache
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        self._log_access_markov(path) # Log and potentially predict
        self.fd += 1
        return self.fd

    def open(self, path, flags):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        # Check permissions based on flags (simplified check)
        # if (flags & os.O_WRONLY or flags & os.O_RDWR) and not (os.access(path, os.W_OK)):
        #      raise FuseOSError(errno.EACCES)
        # if (flags & os.O_RDONLY or flags & os.O_RDWR) and not (os.access(path, os.R_OK)):
        #      raise FuseOSError(errno.EACCES)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        self.fd += 1
        return self.fd

    def read(self, path, length, offset, fh):
        if path not in self.files or not (self.files[path]['st_mode'] & S_IFREG):
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        self._log_access_markov(path) # Log access and trigger prediction/prefetch
        cached_data = self._get_from_cache(path)
        data_to_read = cached_data if cached_data is not None else self.data.get(path, b'')
        if cached_data is None and path in self.data: # If miss but data exists, cache it
            self._update_cache(path, data_to_read)

        return data_to_read[offset:offset + length]

    def write(self, path, buf, offset, fh):
        if path not in self.files or not (self.files[path]['st_mode'] & S_IFREG):
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        self._log_access_markov(path) # Log access and trigger prediction/prefetch
        current_data = self._get_from_cache(path)
        if current_data is None:
            current_data = self.data.get(path, b'')
        data_len = len(current_data)
        write_end = offset + len(buf)
        if offset > data_len: new_data = current_data + b'\0' * (offset - data_len) + buf
        elif write_end > data_len: new_data = current_data[:offset] + buf
        else: new_data = current_data[:offset] + buf + current_data[write_end:]
        self.data[path] = new_data
        self._update_cache(path, new_data)
        self.files[path]['st_size'] = len(new_data)
        now = time.time()
        self.files[path]['st_mtime'] = now
        self.files[path]['st_atime'] = now
        return len(buf)

    def unlink(self, path):
        if path not in self.files: raise FuseOSError(errno.ENOENT)
        if not (self.files[path]['st_mode'] & S_IFREG): raise FuseOSError(errno.EISDIR)
        del self.files[path]
        if path in self.data: del self.data[path]
        if path in self.cache: del self.cache[path]
        for stat_type in list(self.stats.keys()): # Use list() to avoid modifying during iteration
            if path in self.stats[stat_type]: del self.stats[stat_type][path]
        if self.last_accessed_path == path: self.last_accessed_path = None

    def truncate(self, path, length, fh=None):
        if path not in self.files or not (self.files[path]['st_mode'] & S_IFREG):
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        # Optionally log truncate: self._log_access_markov(path)
        current_data = self._get_from_cache(path)
        if current_data is None: current_data = self.data.get(path, b'')
        if length < len(current_data): new_data = current_data[:length]
        else: new_data = current_data + b'\0' * (length - len(current_data))
        self.data[path] = new_data
        self._update_cache(path, new_data)
        self.files[path]['st_size'] = length
        now = time.time(); self.files[path]['st_mtime'] = now; self.files[path]['st_atime'] = now

    def statfs(self, path):
         # Return dummy filesystem stats
         return dict(f_bsize=512, f_blocks=4096*10, f_bavail=2048*10,
                     f_bfree=2048*10, f_files=10000, f_ffree=9000, f_namelen=255)

    # Add other methods like rmdir, mkdir, utimens if needed for more complex tests


# --- Shutdown Monitor (Identical to previous versions) ---
def shutdown_monitor(fs, mountpoint):
    # print("Shutdown monitor started...") # Debug
    while not fs.shutdown_flag.is_set():
        shutdown_file_path = os.path.join(os.getcwd(), "shutdown")
        if os.path.exists(shutdown_file_path):
            print("\nShutdown signal received...")
            stats = { # Combine stats
                'access_count': dict(fs.stats['access_count']),
                'cache_hits': dict(fs.stats['cache_hits']),
                'cache_misses': dict(fs.stats['cache_misses']),
                'prefetches': dict(fs.stats['prefetches'])
            }
            print("Filesystem Statistics:")
            all_paths = set().union(*[set(v.keys()) for v in stats.values()]) # More robust union
            if not all_paths:
                 print("  (No activity recorded)")
            else:
                 print(f"{'Path':<40} {'Accesses':>10} {'Hits':>8} {'Misses':>8} {'Prefetches':>10}")
                 print("-" * 80)
                 for path in sorted(list(all_paths)):
                     # Ensure path key exists before accessing file metadata (it might be deleted)
                     is_file = path in fs.files and (fs.files[path]['st_mode'] & S_IFREG)
                     path_display = path if is_file else f"{path}/" # Indicate dirs visually
                     acc = stats['access_count'].get(path, 0)
                     hit = stats['cache_hits'].get(path, 0)
                     miss = stats['cache_misses'].get(path, 0)
                     pref = stats['prefetches'].get(path, 0)
                     print(f"{path_display:<40} {acc:>10} {hit:>8} {miss:>8} {pref:>10}")

            print("\nAttempting to unmount...")
            # Ensure fusermount is used, especially on Linux
            umount_cmd = f"fusermount -u {mountpoint}"
            if sys.platform == 'darwin': # macOS might just use umount
                umount_cmd = f"umount {mountpoint}"

            result = os.system(umount_cmd)
            if result != 0:
                print(f"Failed to unmount {mountpoint} (cmd: '{umount_cmd}'). Trying lazy unmount (Linux)...")
                if sys.platform.startswith('linux'):
                    os.system(f"fusermount -u -z {mountpoint}") # Lazy unmount

            fs.shutdown_flag.set()
            try: os.remove(shutdown_file_path)
            except OSError: pass
            print("Exiting FUSE process.")
            os._exit(0) # Force exit
        time.sleep(1)


# --- Main Function (Identical to previous versions) ---
def main(mountpoint):
    if not os.path.isdir(mountpoint):
        print(f"Error: Mountpoint '{mountpoint}' is not an existing directory.", file=sys.stderr)
        sys.exit(1)

    fs = MemoryFS()
    print(f"Mounting MarkovFS filesystem at {mountpoint}...")
    print(f"Log file: {os.path.abspath(LOG_FILE)}")
    print(f"Probabilities file: {os.path.abspath(PROB_PATH)}")
    print("Ready for operations. Use benchmark script or manual interaction.")
    print("Create a file named 'shutdown' in the current working directory to stop.")


    monitor_thread = threading.Thread(target=shutdown_monitor, args=(fs, mountpoint), daemon=True)
    monitor_thread.start()

    try:
        # Consider allow_other=True if you need access from users other than the one mounting
        FUSE(fs, mountpoint, nothreads=True, foreground=True, allow_other=False)
    except RuntimeError as e:
        print(f"\nError initializing FUSE: {e}", file=sys.stderr)
        print("Attempting cleanup..."); os.system(f"fusermount -u {mountpoint}" if sys.platform.startswith('linux') else f"umount {mountpoint}"); sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during FUSE operation: {e}");
        print("Attempting cleanup..."); os.system(f"fusermount -u {mountpoint}" if sys.platform.startswith('linux') else f"umount {mountpoint}"); raise
    finally:
        # This code might not be reached due to os._exit in monitor
        print("Filesystem main loop finished.")
        if not fs.shutdown_flag.is_set():
            print("Performing final unmount attempt...")
            os.system(f"fusermount -u {mountpoint}" if sys.platform.startswith('linux') else f"umount {mountpoint}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mountpoint>")
        sys.exit(1)
    main(sys.argv[1])