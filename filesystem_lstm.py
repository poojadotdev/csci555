#!/usr/bin/env python3

import os
import sys
import errno
import threading
import time
import pickle
import stat
import numpy as np
from fuse import FUSE, FuseOSError, Operations
from stat import S_IFDIR, S_IFREG
from collections import OrderedDict, defaultdict, deque
from tensorflow.keras.models import load_model


class MemoryFS(Operations):
    def __init__(self):
        now = time.time()
        self.files = {
            '/': dict(st_mode=(S_IFDIR | 0o755), st_nlink=2, st_ctime=now, st_mtime=now, st_atime=now)
        }
        self.data = {}
        self.fd = 0
        self.cache = OrderedDict()
        self.cache_size = 10
        self.was_prefetched = defaultdict(bool)
        self.stats = defaultdict(lambda: defaultdict(int))
        self.shutdown_flag = threading.Event()

        #loading LSTM model and vocab
        model_path = "csci555/lstm_cache_model.h5"
        vocab_path = "csci555/path_to_index.pkl"
        inv_vocab_path = "csci555/index_to_path.pkl"

        if os.path.exists(model_path):
            self.lstm_model = load_model(model_path)
            print("[LSTM] Loaded trained model.")
        else:
            self.lstm_model = None
            print("[LSTM] No trained model found, starting without prefetching.")

        if os.path.exists(vocab_path) and os.path.exists(inv_vocab_path):
            with open(vocab_path, "rb") as f:
                self.path_to_index = pickle.load(f)
            with open(inv_vocab_path, "rb") as f:
                self.index_to_path = pickle.load(f)
        else:
            self.path_to_index = {}
            self.index_to_path = {}
            print("[LSTM] No path-to-index mappings found.")

        self.n_steps = 5
        self.recent_access_ids = deque(maxlen=self.n_steps)
        self.training_mode = not bool(self.lstm_model)

    def _normalize_path(self, path):
        norm = '/' + path.lstrip('/')
        return norm if norm == '/' else norm.rstrip('/')

    def _update_cache(self, path, data):
        norm = self._normalize_path(path)
        if norm in self.cache:
            self.cache.move_to_end(norm)
        else:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[norm] = data

    def _get_from_cache(self, path):
        norm = self._normalize_path(path)
        if norm in self.cache:
            self.cache.move_to_end(norm)
            self.stats['cache_hits'][norm] += 1
            return self.cache[norm]
        else:
            self.stats['cache_misses'][norm] += 1
            return None

    def predict_next_file(self):
        if not self.lstm_model or len(self.recent_access_ids) < self.n_steps:
            return None
        try:
            input_seq = np.array(self.recent_access_ids).reshape((1, self.n_steps))
            probs = self.lstm_model.predict(input_seq, verbose=0)[0]
            pred_idx = np.argmax(probs)
            return self.index_to_path.get(pred_idx)
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None

    def prefetch_next_file(self):
        predicted = self.predict_next_file()
        if predicted and predicted in self.data and predicted not in self.cache:
            self._update_cache(predicted, self.data[predicted])
            self.was_prefetched[predicted] = True
            self.stats['prefetches'][predicted] += 1
            print(f"[PREFETCHED]: {predicted}")

    def _log_and_predict(self, path):
        norm = self._normalize_path(path)
        if self.training_mode:
            with open("access_log.txt", "a") as log:
                log.write(f"{norm}\n")
        if not self.training_mode and norm in self.path_to_index:
            self.recent_access_ids.append(self.path_to_index[norm])
            self.prefetch_next_file()

    def access(self, path, mode):
        norm = self._normalize_path(path)
        if norm not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][norm] += 1
        return 0

    def getattr(self, path, fh=None):
        norm = self._normalize_path(path)
        if norm not in self.files:
            raise FuseOSError(errno.ENOENT)
        return self.files[norm]

    def readdir(self, path, fh):
        norm = self._normalize_path(path)
        if norm not in self.files or not stat.S_ISDIR(self.files[norm]['st_mode']):
            raise FuseOSError(errno.ENOENT)
        dirents = ['.', '..']
        for name in self.files:
            if name != '/' and os.path.dirname(name) == norm:
                dirents.append(os.path.basename(name))
        return dirents

    def create(self, path, mode):
        norm = self._normalize_path(path)
        now = time.time()
        self.files[norm] = dict(st_mode=(S_IFREG | 0o644), st_nlink=1, st_size=0, st_ctime=now, st_mtime=now, st_atime=now)
        self.data[norm] = b""
        self.stats['access_count'][norm] += 1
        self.fd += 1
        self._log_and_predict(norm)
        return self.fd

    def open(self, path, flags):
        norm = self._normalize_path(path)
        self.stats['access_count'][norm] += 1
        self._log_and_predict(norm)
        self.fd += 1
        return self.fd

    def read(self, path, length, offset, fh):
        print(f"[READ OPERATION]: {path}")
        norm = self._normalize_path(path)
        self.stats['access_count'][norm] += 1
        self._log_and_predict(norm)
        if self.was_prefetched[norm]:
            self.stats['prefetches'][norm] += 1
            self.was_prefetched[norm] = False
        cached = self._get_from_cache(norm)
        if cached is not None:
            return cached[offset:offset + length]
        if norm not in self.data:
            raise FuseOSError(errno.ENOENT)
        data = self.data[norm]
        self._update_cache(norm, data)
        return data[offset:offset + length]

    def write(self, path, buf, offset, fh):
        print(f"[WRITE OPERATION]: {path}")
        norm = self._normalize_path(path)
        self.stats['access_count'][norm] += 1
        self._log_and_predict(norm)
        current = self._get_from_cache(norm)
        if current is None:
            current = self.data.get(norm, b'')
        new_data = current[:offset] + buf + current[offset + len(buf):]
        self.data[norm] = new_data
        self._update_cache(norm, new_data)
        self.files[norm]['st_size'] = len(new_data)
        self.files[norm]['st_mtime'] = time.time()
        return len(buf)

    def unlink(self, path):
        norm = self._normalize_path(path)
        for store in [self.files, self.data, self.cache]:
            store.pop(norm, None)
        for stat_type in self.stats:
            self.stats[stat_type].pop(norm, None)

            
def shutdown_monitor(fs, mountpoint):
    print("To stop the filesystem, create a file named 'shutdown'")
    while not fs.shutdown_flag.is_set():
        if os.path.exists("shutdown"):
            fs.shutdown_flag.set()
            print("\nShutdown signal received...")
            print("\nFilesystem Statistics:")
            all_paths = set().union(*[fs.stats[stat].keys() for stat in fs.stats])
            with open("lstm_result.txt", "w") as f:
                for path in sorted(all_paths):
                    acc = fs.stats['access_count'][path]
                    hit = fs.stats['cache_hits'][path]
                    miss = fs.stats['cache_misses'].get(path, 0)
                    pref = fs.stats['prefetches'][path]
                    line = f"{path:<40} {acc:>10} {hit:>8} {miss:>8} {pref:>10}"
                    print(line)
                    f.write(line + "\n")
            os._exit(0)
        time.sleep(1)


def main(mountpoint):
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    fs = MemoryFS()
    monitor_thread = threading.Thread(target=shutdown_monitor, args=(fs, mountpoint), daemon=True)
    monitor_thread.start()
    FUSE(fs, mountpoint, foreground=True, allow_other=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mountpoint>")
        sys.exit(1)
    main(sys.argv[1])