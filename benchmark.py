#!/usr/bin/env python3

import os
import sys
import time

def benchmark_fuse_filesystem(mountpoint, num_files=15, reads_per_file=5, writes_per_file=2):
    """
    Benchmark the FUSE filesystem by creating files, writing to them, and reading them repeatedly.
    
    Args:
        mountpoint (str): Path where the FUSE filesystem is mounted (e.g., '/private/tmp/fuse_mount').
        num_files (int): # of files to create (default: 15, exceeds cache size 10).
        reads_per_file (int): # of times to read each file.
        writes_per_file (int): # of time to write to each file.
    """

    if not os.path.exists(mountpoint):
        print(f"Error: Mountpoint {mountpoint} does not exist.")
        sys.exit(1)

    print(f"Starting benchmark on {mountpoint}...")
    print(f"Creating {num_files} files, {writes_per_file} writes and {reads_per_file} reads per file.")

    start_time = time.time()
    for i in range(num_files):
        file_path = os.path.join(mountpoint, f"file{i}.txt")
        for _ in range(writes_per_file):
            with open(file_path, 'ab') as f: 
                f.write(f"Data for file {i}, write {_}\n".encode('utf-8'))
        print(f"Created and wrote to {file_path}")

    write_time = time.time() - start_time
    print(f"Write phase completed in {write_time:.2f} seconds")


    start_time = time.time()
    for i in range(num_files):
        file_path = os.path.join(mountpoint, f"file{i}.txt")
        for _ in range(reads_per_file):
            with open(file_path, 'rb') as f:
                content = f.read() 
            # print(f"Read {file_path}: {content.decode('utf-8')[:20]}...")
        print(f"Read {file_path} {reads_per_file} times")

    read_time = time.time() - start_time
    print(f"Read phase completed in {read_time:.2f} seconds")

    start_time = time.time()
    for _ in range(3):  # Simulate multiple ls -l
        files = os.listdir(mountpoint)
        # print(f"Directory listing: {files}")
    list_time = time.time() - start_time
    print(f"Directory listing phase completed in {list_time:.2f} seconds")



    print("Triggering shutdown to display stats...")
    shutdown_file = os.path.join(os.getcwd(), "shutdown")
    with open(shutdown_file, 'w') as f:
        f.write("shutdown")
    print(f"Created {shutdown_file}. Waiting for stats...")

    time.sleep(2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 benchmark.py <mountpoint>")
        sys.exit(1)

    mountpoint = sys.argv[1]
    benchmark_fuse_filesystem(mountpoint)