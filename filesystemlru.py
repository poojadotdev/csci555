#!/usr/bin/env python3

import os
import sys
import errno
import threading
import time
from fuse import FUSE, FuseOSError, Operations
from stat import S_IFDIR, S_IFREG
from collections import OrderedDict

class MemoryFS(Operations):
    def __init__(self):
        self.files = {}
        self.data = {}
        self.fd = 0
        self.cache = OrderedDict()
        self.cache_size = 10
        self.stats = {
            'access_count': {},
            'cache_hits': {},
            'cache_misses': {}
        }
        now = time.time()
        self.files['/'] = {
            'st_mode': (S_IFDIR | 0o755),
            'st_ctime': now,
            'st_mtime': now,
            'st_atime': now,
            'st_nlink': 2
        }
        self.shutdown_flag = threading.Event()  # For graceful shutdown

    def _update_cache(self, path, data):
        if path in self.cache:
            self.cache.move_to_end(path)
        else:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[path] = data

    def _get_from_cache(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            self.stats['cache_hits'][path] = self.stats['cache_hits'].get(path, 0) + 1
            return self.cache[path]
        self.stats['cache_misses'][path] = self.stats['cache_misses'].get(path, 0) + 1
        return None

    def _get_path(self, partial):
        if partial.startswith("/"):
            partial = partial[1:]
        return '/' + partial

    def get_stats(self, path=None):
        if path:
            return {
                'access_count': self.stats['access_count'].get(path, 0),
                'cache_hits': self.stats['cache_hits'].get(path, 0),
                'cache_misses': self.stats['cache_misses'].get(path, 0)
            }
        else:
            return {
                'access_count': dict(self.stats['access_count']),
                'cache_hits': dict(self.stats['cache_hits']),
                'cache_misses': dict(self.stats['cache_misses'])
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
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        return self.files[path]

    def readdir(self, path, fh):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        dirents = ['.', '..']
        if path == '/':
            dirents.extend([name[1:] for name in self.files if name != '/'])
        else:
            base = path + '/'
            dirents.extend([name[len(base):] for name in self.files if name.startswith(base)])
        return dirents

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

    def create(self, path, mode, fi=None):
        now = time.time()
        self.files[path] = {
            'st_mode': (S_IFREG | mode),
            'st_ctime': now,
            'st_mtime': now,
            'st_atime': now,
            'st_nlink': 1,
            'st_size': 0
        }
        self.data[path] = b''
        self._update_cache(path, b'')
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        self.fd += 1
        return self.fd

    def open(self, path, flags):
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        self.fd += 1
        return self.fd

    def read(self, path, length, offset, fh):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        cached_data = self._get_from_cache(path)
        if cached_data is None:
            if path not in self.data:
                raise FuseOSError(errno.ENOENT)
            cached_data = self.data[path]
            self._update_cache(path, cached_data)
        return cached_data[offset:offset + length]

    def write(self, path, buf, offset, fh):
        if path not in self.files:
            raise FuseOSError(errno.ENOENT)
        self.stats['access_count'][path] = self.stats['access_count'].get(path, 0) + 1
        current_data = self._get_from_cache(path)
        if current_data is None:
            current_data = self.data.get(path, b'')
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
            stats = fs.get_stats()
            print("Filesystem Statistics:")
            for stat_type in ['access_count', 'cache_hits', 'cache_misses']:
                print(f"{stat_type}:")
                if stats[stat_type]:
                    for path, count in stats[stat_type].items():
                        print(f"  {path}: {count}")
                else:
                    print("  (none)")
            print("Attempting to unmount...")
            result = os.system(f"umount -f {mountpoint}")
            if result == 0:
                print(f"Unmounted {mountpoint} successfully")
            else:
                print(f"Failed to unmount {mountpoint}, forcing lazy unmount...")
                os.system(f"umount -l {mountpoint}")
            fs.shutdown_flag.set()
            os._exit(0)  # Force exit since FUSE might not terminate
        time.sleep(1)

def main(mountpoint):
    fs = MemoryFS()
    print(f"Mounting filesystem at {mountpoint}...")
    # Start shutdown monitor in a separate thread
    monitor_thread = threading.Thread(target=shutdown_monitor, args=(fs, mountpoint))
    monitor_thread.daemon = True  # Dies with main thread
    monitor_thread.start()
    
    try:
        FUSE(fs, mountpoint, nothreads=True, foreground=True)
    except Exception as e:
        print(f"Error during FUSE operation: {e}")
        os.system(f"umount -f {mountpoint}")
        raise

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <mountpoint>'.format(sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1])