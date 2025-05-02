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
from collections import OrderedDict
from tensorflow.keras.models import load_model
from pathlib import Path
from collections import defaultdict




class MemoryFS(Operations):
    def __init__(self):
        # self.files = {}
        self.files = {
            '/': dict(st_mode=(stat.S_IFDIR | 0o755), st_nlink=2, st_ctime=time.time(), st_mtime=time.time(), st_atime=time.time())
        }
        self.data = {}
        self.fd = 0
        self.cache = OrderedDict()
        self.cache_size = 10
        self.was_prefetched = defaultdict(bool)
        self.stats = {
            'access_count': {},
            'cache_hits': {},
            'cache_misses': {},
            'prefetches': {}, 
        }
        now = time.time()
        # self.files['/'] = {
        #     'st_mode': (S_IFDIR | 0o755),
        #     'st_ctime': now,
        #     'st_mtime': now,
        #     'st_atime': now,
        #     'st_nlink': 2
        # }
        self.shutdown_flag = threading.Event()  



        #loading LSTM model
        if os.path.exists("csci555/lstm_cache_model.h5"):
            self.lstm_model = load_model("csci555/lstm_cache_model.h5")
            print("[LSTM] Loaded trained model.")
        else:
            self.lstm_model = None
            print("[LSTM] No trained model found, starting without prefetching.")

        
        
        self.training_mode = not bool(self.lstm_model)
        self.log_access_enabled = True 
        print(f"[INIT] LSTM model loaded: {bool(self.lstm_model)}")
        print(f"[INIT] Training mode: {self.training_mode}")


        #load path mappings if available
        if os.path.exists("csci555/path_to_index.pkl") and os.path.exists("csci555/index_to_path.pkl"):
            with open("csci555/path_to_index.pkl", "rb") as f:
                self.path_to_index = pickle.load(f)
            with open("csci555/index_to_path.pkl", "rb") as f:
                self.index_to_path = pickle.load(f)
        else:
            self.path_to_index = {}
            self.index_to_path = {}
            print("[LSTM] No path-to-index mappings found, starting fresh.")

        #initializes the recent access list
        self.n_steps = 5
        self.recent_access_ids = []

    
    #cache keeping the predicted file in memory and evict something else instead
    def _update_cache(self, path, data):
        if path in self.cache:
            self.cache.move_to_end(path)
        else:
            if len(self.cache) >= self.cache_size:
                predicted_path = self.predict_next_file()
                for evict_path in self.cache:
                    if evict_path != predicted_path:
                        print(f"[LSTM] Evicting {evict_path}, keeping predicted {predicted_path}")
                        del self.cache[evict_path]
                        break
            self.cache[path] = data

    def _get_from_cache(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            self.stats['cache_hits'][path] = self.stats['cache_hits'].get(path, 0) + 1
            return self.cache[path]
        self.stats['cache_misses'][path] = self.stats['cache_misses'].get(path, 0) + 1
        return None
    
    

    def predict_next_file(self):
        if len(self.recent_access_ids) < self.n_steps:
            print(f"[LSTM] Skipping prediction: only {len(self.recent_access_ids)} accesses (need {self.n_steps})")
            return None
        input_seq = np.array(self.recent_access_ids).reshape((1, self.n_steps))
        probs = self.lstm_model.predict(input_seq, verbose=0)
        pred_idx = np.argmax(probs)
        print(f"[LSTM] Predicted index: {pred_idx} → {self.index_to_path.get(pred_idx)}")
        return self.index_to_path.get(pred_idx)


    def prefetch_next_file(self):
        """Prefetch the next predicted file into the cache based on LSTM prediction."""
        if len(self.recent_access_ids) < self.n_steps:
            print(f"[LSTM] Skipping prefetch: only {len(self.recent_access_ids)} access IDs (need {self.n_steps})")
            return  

        predicted_path = self.predict_next_file()
        # print(f"[PREFETCH] Trying: {predicted_path}")
        # print(f"    In data? {predicted_path in self.data}")
        # print(f"    In cache? {predicted_path in self.cache}")

        if predicted_path in self.data and predicted_path not in self.cache:
            self.cache[predicted_path] = self.data[predicted_path]
            self.was_prefetched[predicted_path] = True
            # print(f"[PREFETCHED]: {predicted_path}")



    def _get_path(self, partial):
        if partial.startswith("/"):
            partial = partial[1:]
        return '/' + partial

        
    def get_stats(self, path=None):
        if path:
            return {
                'access_count': self.stats['access_count'].get(path, 0),
                'cache_hits': self.stats['cache_hits'].get(path, 0),
                'cache_misses': self.stats['cache_misses'].get(path, 0),
                'prefetches': self.stats['prefetches'].get(path, 0),
            }
        else:
            return {
                'access_count': dict(self.stats['access_count']),
                'cache_hits': dict(self.stats['cache_hits']),
                'cache_misses': dict(self.stats['cache_misses']),
                'prefetches': dict(self.stats['prefetches']),  
            }


    def access(self, path, mode):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        return 0

    def chmod(self, path, mode):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.files[path]['st_mode'] &= 0o770000
        self.files[path]['st_mode'] |= mode
        return 0

    def chown(self, path, uid, gid):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.files[path]['st_uid'] = uid
        self.files[path]['st_gid'] = gid
        return 0


    def getattr(self, path, fh=None):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        return self.files[path]
    
   
    
    def readdir(self, path, fh):
       
        if path not in self.files or not stat.S_ISDIR(self.files[path]['st_mode']):
           
            raise FuseOSError(errno.ENOENT)

        dirents = ['.', '..']
        for name in self.files:
            if name != '/':
                # Check if file is directly under current directory
                parent = os.path.dirname(name)
                if parent == path:
                    basename = os.path.basename(name)
                    if basename not in dirents:
                        dirents.append(basename)

        return dirents

    
    def create(self, path, mode):
        
        now = time.time()
        self.files[path] = dict(
            st_mode=(stat.S_IFREG | 0o644),
            st_nlink=1,
            st_size=0,
            st_ctime=now,
            st_mtime=now,
            st_atime=now
        )
  

        self.data[path] = b""
        self.stats['access_count'][path] = 1
        self.fd += 1
        return self.fd


    def mkdir(self, path, mode):
        now = time.time()
        self.files[path] = {
            'st_mode': (S_IFDIR | mode),
            'st_ctime': now,
            'st_mtime': now,
            'st_atime': now,
            'st_nlink': 2
        }
        self.files[path[:-len(os.path.basename(path))]]['st_nlink'] += 1
        return 0

    def rmdir(self, path):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        if not self.files[path]['st_mode'] & S_IFDIR:
            raise FuseOSError(errno.ENOTDIR)
        if any(k.startswith(path + '/') for k in self.files.keys()):
            raise FuseOSError(errno.ENOTEMPTY)
        self.files[path[:-len(os.path.basename(path))]]['st_nlink'] -= 1
        del self.files[path]
        if path in self.cache:
            del self.cache[path]
        return 0

    def statfs(self, path):
        return {
            'f_bsize': 512,
            'f_blocks': 4096,
            'f_bavail': 2048,
            'f_bfree': 2048,
            'f_files': 1024,
            'f_ffree': 1000,
            'f_namelen': 255
        }

    
    def open(self, path, flags):
        print(f"OPEN OPERATION: {path}")  

        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1

        # also log the file access during open()
        if self.log_access_enabled:
            print(f"Logging open access: {path}") 

            with open("access_log.txt", "a") as log:
                log.write(f"{path}\n")

        self.fd += 1
        return self.fd


    def read(self, path, length, offset, fh):
        print(f"[READ OPERATION]: {path}")

        if path not in self.files:
            raise FuseOSError(errno.ENOENT)

        #access count
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1

        #checking if this was a successful prefetch
        if self.was_prefetched.get(path, False):
            self.stats['prefetches'][path] = self.stats['prefetches'].get(path, 0) + 1
            print(f"[PREFETCH SUCCESSFUL]: {path}")
            self.was_prefetched[path] = False 

        # --- Training mode logging ---
        if self.training_mode:
            print(f"[TRAINING] Logging read access: {path}")
            with open("access_log.txt", "a") as log:
                log.write(f"{path}\n")

        # --- Inference mode prefetching ---
        if not self.training_mode:
            if path in self.path_to_index:
                idx = self.path_to_index[path]
                self.recent_access_ids.append(idx)
                if len(self.recent_access_ids) > self.n_steps:
                    self.recent_access_ids.pop(0)
            self.prefetch_next_file()

        #trying to serve from cache
        cached_data = self._get_from_cache(path)
        if cached_data is not None:
            self.stats['cache_hits'][path] = self.stats['cache_hits'].get(path, 0) + 1
            print(f"[CACHE HIT]: {path}")
            return cached_data[offset:offset + length]

        #cache miss fallback
        if path not in self.data:
            raise FuseOSError(errno.ENOENT)

        data = self.data[path]
        self._update_cache(path, data)
        return data[offset:offset + length]




    def write(self, path, buf, offset, fh):
        print(f"WRITE OPERATION: {path}")  
        print(f"[WRITE DEBUG]: {path}, offset: {offset}, len(buf): {len(buf)}")


        if path not in self.files:
            raise FuseOSError(errno.ENOENT)

        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1

        # --- Logging phase ---
        if self.training_mode:
            print(f"Logging write access: {path}")
            with open("access_log.txt", "a") as log:
                log.write(f"{path}\n")

        # --- Prediction/prefetch phase ---
        if not self.training_mode:
            if path in self.path_to_index:
                idx = self.path_to_index[path]
                self.recent_access_ids.append(idx)
                if len(self.recent_access_ids) > self.n_steps:
                    self.recent_access_ids.pop(0)

        cached = self._get_from_cache(path)
        current_data = cached if cached is not None else self.data.get(path, b'')


        if offset + len(buf) > len(current_data):
            current_data = current_data + b'\0' * (offset + len(buf) - len(current_data))

        new_data = current_data[:offset] + buf + current_data[offset + len(buf):]
        self.data[path] = new_data
        self._update_cache(path, new_data)
        self.files[path]['st_size'] = len(new_data)
        self.files[path]['st_mtime'] = time.time()

        return len(buf)


    def truncate(self, path, length, fh=None):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        current_data = self._get_from_cache(path)
        if current_data is None:
            current_data = self.data.get(path, b'')
        new_data = current_data[:length]
        self.data[path] = new_data
        self._update_cache(path, new_data)
        self.files[path]['st_size'] = length
        self.files[path]['st_mtime'] = time.time()

    def unlink(self, path):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        del self.files[path]
        if path in self.data:
            del self.data[path]
        if path in self.cache:
            del self.cache[path]
        for stat_type in ['access_count', 'cache_hits', 'cache_misses']:
            if path in self.stats[stat_type]:
                del self.stats[stat_type][path]

    def utimens(self, path, times=None):
        now = time.time()
        atime, mtime = times if times else (now, now)
        self.files[path]['st_atime'] = atime
        self.files[path]['st_mtime'] = mtime

        
def shutdown_monitor(fs, mountpoint):
    print("To stop the filesystem and see stats, create a file named 'shutdown' in the current directory")
    while not fs.shutdown_flag.is_set():
        if os.path.exists("shutdown"):
            print("\nShutdown signal received...")
            print("\nFilesystem Statistics:")

            all_paths = set().union(
                fs.stats['access_count'].keys(),
                fs.stats['cache_hits'].keys(),
                fs.stats['cache_misses'].keys(),
                fs.stats['prefetches'].keys(),
            )

            print(f"{'Path':<40} {'Accesses':>10} {'Hits':>8} {'Misses':>8} {'Prefetches':>10}")
            print("-" * 80)

            for path in sorted(all_paths):
                acc = fs.stats['access_count'].get(path, 0)
                hit = fs.stats['cache_hits'].get(path, 0)
                miss = acc - hit if path in fs.data else 0
                pref = fs.stats['prefetches'].get(path, 0)
                print(f"{path:<40} {acc:>10} {hit:>8} {miss:>8} {pref:>10}")
                with open("lstm_result.txt", "a") as f:
                    f.write(f"{path:<40} {acc:>10} {hit:>8} {miss:>8} {pref:>10}\n")


            print("\nAttempting to unmount...")
            result = os.system(f"umount -f {mountpoint}")
            if result != 0:
                print("Failed to unmount, trying lazy unmount...")
                os.system(f"umount -l {mountpoint}")
            fs.shutdown_flag.set()
            os._exit(0)
        time.sleep(1)

        
        
def main(mountpoint):
    print(f"[DEBUG] Entered main() — trying to mount {mountpoint}")

    # Ensure the mountpoint directory exists
    if not os.path.exists(mountpoint):
        print(f"[DEBUG] Creating mountpoint directory: {mountpoint}")
        os.makedirs(mountpoint)

    fs = MemoryFS()
    print(f"Mounting filesystem at {mountpoint}...")

    # Start shutdown monitor in a separate thread
    monitor_thread = threading.Thread(target=shutdown_monitor, args=(fs, mountpoint))
    monitor_thread.daemon = True  # Dies with main thread
    monitor_thread.start()

    try:
        print(f"[DEBUG] Calling FUSE() on: {mountpoint}")
        # FUSE(fs, mountpoint, nothreads=True, foreground=True)
        FUSE(fs, mountpoint, foreground=True, allow_other=True)

        print(f"[SUCCESS] FUSE mounted successfully on {mountpoint}.")
    except Exception as e:
        print("[FATAL ERROR] FUSE failed to mount!")
        print(f"Reason: {repr(e)}")
        os.system(f"umount -f {mountpoint}")
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <mountpoint>'.format(sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1])
