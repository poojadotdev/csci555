#!/usr/bin/env python3

import os
import sys
import time
import random
import math

# --- Configuration ---
NUM_FILES = 100          # Number of files to create
WRITES_PER_FILE = 5      # Number of write operations per file
READS_PER_FILE = 10     # Number of read operations per file (per pattern)
DATA_SIZE_PER_WRITE = 1024 # Bytes to write in each write operation (1KB)
LISTDIR_ITERATIONS = 5   # Number of times to list the directory

# --- Helper Function ---
def format_size(size_bytes):
    """Converts bytes to a human-readable format."""
    if size_bytes == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    if size_bytes <= 0: return "0B"
    try:
        i = int(math.floor(math.log(size_bytes, 1024)))
        i = max(0, min(i, len(size_name) - 1))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"
    except ValueError: return f"{size_bytes} B"

# --- Main Benchmark Function ---
def benchmark_fuse_filesystem(mountpoint):
    """
    Performs a longer benchmark on the FUSE filesystem, including sequential,
    reverse, and random read patterns after an initial write phase.
    """
    abs_mountpoint = os.path.abspath(mountpoint)
    print(f"Targeting mountpoint: {abs_mountpoint}")

    if not os.path.isdir(abs_mountpoint):
        print(f"Error: Mountpoint '{abs_mountpoint}' is not an existing directory.")
        # Check if it *looks* like a file, which can happen if FUSE failed to mount
        if os.path.exists(abs_mountpoint):
            print("       It seems to exist but is not a directory. FUSE mount may have failed.")
        sys.exit(1)

    # Optional: Basic check if it seems mounted (might not be reliable for FUSE)
    # if not os.path.ismount(abs_mountpoint):
    #      print(f"Warning: '{abs_mountpoint}' does not appear to be an active mount point (os.path.ismount test).")

    print("="*70)
    print(f"Starting Longer Benchmark on: {abs_mountpoint}")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Files to create:         {NUM_FILES}")
    print(f"  - Writes per file:         {WRITES_PER_FILE}")
    print(f"  - Reads per file/pattern:  {READS_PER_FILE}")
    print(f"  - Data size per write:     {format_size(DATA_SIZE_PER_WRITE)}")
    print(f"  - Directory list iterations: {LISTDIR_ITERATIONS}")
    final_file_size_expected = WRITES_PER_FILE * DATA_SIZE_PER_WRITE
    total_data_written_expected = NUM_FILES * final_file_size_expected
    total_data_read_expected_per_pattern = NUM_FILES * READS_PER_FILE * final_file_size_expected
    print(f"  - Expected final size/file: {format_size(final_file_size_expected)}")
    print(f"  - Total data to write:     {format_size(total_data_written_expected)}")
    print(f"  - Total data to read (each pattern): {format_size(total_data_read_expected_per_pattern)}")
    print("-"*70)

    # --- File Paths ---
    # Use absolute paths based on the mountpoint
    file_paths = [os.path.join(abs_mountpoint, f"benchmark_file_{i:04d}.dat") for i in range(NUM_FILES)]

    # --- Cleanup Old Files ---
    print("Attempting to clean up old benchmark files...")
    cleanup_count = 0
    if os.path.isdir(abs_mountpoint): # Check again before cleanup
        try:
            # List directory to find files to remove (safer than generating all paths)
            existing_files = [os.path.join(abs_mountpoint, f) for f in os.listdir(abs_mountpoint) if f.startswith("benchmark_file_") and f.endswith(".dat")]
            for p in existing_files:
                try:
                    os.remove(p)
                    cleanup_count += 1
                except Exception as e:
                    print(f"  Warning: Could not remove old file {p}: {e}")
        except Exception as e:
             print(f"  Warning: Could not list directory {abs_mountpoint} for cleanup: {e}")

    if cleanup_count > 0: print(f"  Removed {cleanup_count} old files.")
    else: print("  No old benchmark files found or removable.")

    # --- Write Phase ---
    print(f"[1] Starting Write Phase ({NUM_FILES} files)...")
    write_phase_start_time = time.time()
    total_bytes_written_actual = 0
    write_errors = 0

    for i in range(NUM_FILES):
        file_path = file_paths[i]
        try:
            # Using 'wb+' allows writing and creates the file; ensures it's fresh.
            with open(file_path, 'wb+') as f:
                file_content_size = 0
                for w in range(WRITES_PER_FILE):
                    data_chunk = os.urandom(DATA_SIZE_PER_WRITE)
                    bytes_written = f.write(data_chunk)
                    if bytes_written != DATA_SIZE_PER_WRITE:
                        print(f"  Warning: Short write on {file_path}, iteration {w}. Expected {DATA_SIZE_PER_WRITE}, got {bytes_written}")
                        # No need to break, just record less data
                    total_bytes_written_actual += bytes_written
                    file_content_size += bytes_written
                f.flush() # Ensure data is sent to FUSE layer

            # Optional sanity check file size after writing
            # tray:
            #     written_size = os.path.getsize(file_path)
            #     if written_size != file_content_size:
            #          print(f"  Warning: Size mismatch after writing {file_path}. Expected {file_content_size}, actual {written_size}")
            # except Exception as e:
            #      print(f"  Warning: Cannot get size of {file_path} after write: {e}")


            if (i + 1) % (NUM_FILES // 10 or 1) == 0 or i == NUM_FILES - 1:
                progress = (i + 1) / NUM_FILES
                print(f"  Written {i + 1}/{NUM_FILES} files... [{int(progress * 20) * '#'}{int((1-progress)*20) * '-'}]", end='\r')
        except Exception as e:
            print(f"\nError writing to {file_path}: {e}") # Newline if error occurs
            write_errors += 1
            # Decide whether to continue or stop on error
            # continue
    print("\n" + "-"*70) # Newline after progress bar

    write_time = time.time() - write_phase_start_time
    print(f"Write Phase completed in {write_time:.2f} seconds.")
    print(f"  Total bytes written: {format_size(total_bytes_written_actual)}")
    if write_errors > 0: print(f"  WARNING: Encountered {write_errors} write errors.")
    print("-"*70)
    time.sleep(1) # Small pause

    # --- Read Phase Function ---
    def run_read_phase(phase_num, pattern_name, file_order_indices):
        print(f"[{phase_num}] Starting Read Phase ({pattern_name}, {READS_PER_FILE} reads/file)...")
        phase_start_time = time.time()
        total_bytes_read_actual = 0
        read_errors = 0

        for count, i in enumerate(file_order_indices):
            file_path = file_paths[i]
            file_read_count = 0
            file_byte_count = 0
            try:
                # Check if file exists *before* trying to open multiple times
                # This check might interact with FUSE getattr
                if not os.path.exists(file_path):
                    print(f"\n  ERROR ({pattern_name}): File disappeared before read loop: {file_path}")
                    read_errors += 1
                    continue # Skip this file

                for r in range(READS_PER_FILE):
                    try:
                    # print(f"  BENCHMARK: Attempting read {r+1}/{READS_PER_FILE} of {file_path} ({pattern_name})") # Optional debug print
                        with open(file_path, 'rb') as f:
                            content = f.read() # Read the whole file
                        read_len = len(content)
                        # Optional: Verify size roughly matches expected size
                        if read_len != final_file_size_expected:
                            print(f"\n  Warning: Size mismatch reading {file_path} ({pattern_name}, read {r+1}). Expected {final_file_size_expected}, got {read_len}")
                        total_bytes_read_actual += read_len
                        file_byte_count += read_len
                        file_read_count += 1

                        time.sleep(0.01) # <<<<< ADD THIS VERY SMALL SLEEP

                    except FileNotFoundError:
                        print(f"\n  ERROR ({pattern_name}): FileNotFoundError during read {r+1}: {file_path}")
                        read_errors += 1
                        break # Stop reading this file if it disappeared
                    except OSError as e:
                        print(f"\n  ERROR ({pattern_name}): OSError during read {r+1} of {file_path}: {e}")
                        read_errors += 1
                        break # Stop reading this file on OS error
                    except Exception as e:
                        print(f"\n  ERROR ({pattern_name}): Unexpected error during read {r+1} of {file_path}: {e}")
                        read_errors += 1
                        break # Stop reading on other errors

                # Progress indicator
                if (count + 1) % (NUM_FILES // 10 or 1) == 0 or count == NUM_FILES - 1:
                    progress = (count + 1) / NUM_FILES
                    print(f"  Read {count + 1}/{NUM_FILES} files ({pattern_name})... [{int(progress * 20) * '#'}{int((1-progress)*20) * '-'}]", end='\r')


            except Exception as e:
                print(f"\nOuter error reading file {file_path} in phase {pattern_name}: {e}")
                read_errors += 1

        print("\n" + "-"*70) # Newline after progress bar
        phase_time = time.time() - phase_start_time
        print(f"{pattern_name} Read Phase completed in {phase_time:.2f} seconds.")
        print(f"  Total bytes read: {format_size(total_bytes_read_actual)}")
        if read_errors > 0: print(f"  WARNING: Encountered {read_errors} read errors.")
        print("-"*70)
        time.sleep(1) # Small pause between read phases

    # --- Execute Read Phases ---
    # 2. Sequential Read
    sequential_indices = list(range(NUM_FILES))
    run_read_phase(2, "Sequential", sequential_indices)

    # 3. Reverse Read
    reverse_indices = list(range(NUM_FILES))
    reverse_indices.reverse()
    run_read_phase(3, "Reverse", reverse_indices)

    # 4. Random Read
    random_indices = list(range(NUM_FILES))
    random.shuffle(random_indices)
    run_read_phase(4, "Random", random_indices)

    # --- Directory Listing Phase ---
    print(f"[5] Starting Directory Listing Phase ({LISTDIR_ITERATIONS} iterations)...")
    list_phase_start_time = time.time()
    list_errors = 0
    listed_files_count = -1 # Initialize to -1
    for i in range(LISTDIR_ITERATIONS):
        try:
            # Ensure mountpoint still exists and is a dir
            if not os.path.isdir(abs_mountpoint):
                print(f"Error: Mountpoint {abs_mountpoint} disappeared before listdir iteration {i+1}")
                list_errors += LISTDIR_ITERATIONS - i # Count remaining iterations as errors
                break
            files = os.listdir(abs_mountpoint)
            listed_files_count = len(files) # Get count from last successful list
        except Exception as e:
            print(f"\nError listing directory {abs_mountpoint} on iteration {i+1}: {e}")
            list_errors += 1
            # Optional: break on error? or continue?

    print("-"*70) # End listdir section
    list_time = time.time() - list_phase_start_time
    print(f"Directory Listing Phase completed in {list_time:.2f} seconds.")
    # Adjust expected count: ignore potential hidden files unless debugging
    # FUSE readdir now filters ._* and .DS_Store, so count should be closer to NUM_FILES
    if listed_files_count != -1:
        print(f"  Entries found in last listing: {listed_files_count} (Expected ~ {NUM_FILES})")
    else:
        print(f"  Directory listing did not succeed.")
    if list_errors > 0: print(f"  WARNING: Encountered {list_errors} listdir errors.")
    print("-"*70)

    # --- Trigger Shutdown ---
    print("[6] Triggering FUSE shutdown to display stats...")
    # <<< Add pause before creating shutdown file >>>
    print("      Pausing 5 seconds before creating shutdown file...")
    time.sleep(5)

    # Ensure shutdown file is created where the FUSE script runs (usually CWD)
    shutdown_file = os.path.join(os.getcwd(), "shutdown")
    try:
        with open(shutdown_file, 'w') as f:
            f.write("shutdown")
        print(f"Created {shutdown_file}. Waiting for FUSE process to react...")
    except Exception as e:
        print(f"Error creating shutdown file {shutdown_file}: {e}")
        print("Please manually stop the FUSE process if needed.")
        sys.exit(1) # Exit if we can't signal shutdown

    # Give the FUSE process time to detect the file, print stats, and unmount
    print("Benchmark finished. Check FUSE process output and stats file.")
    print("Waiting a few seconds before exiting benchmark script...")
    time.sleep(5) # Increased sleep

# --- Main Execution Block ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 markov_benchmark.py <mountpoint>")
        sys.exit(1)

    mount_path = sys.argv[1]
    try:
        benchmark_fuse_filesystem(mount_path)
    except Exception as e:
        print(f"\n--- BENCHMARK SCRIPT FAILED ---")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"-------------------------------")
        sys.exit(1)
