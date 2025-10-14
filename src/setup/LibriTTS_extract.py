from collections import defaultdict
from pathlib import Path
import multiprocessing
import tarfile
import os
import time
from typing import List
import argparse

def extract_targz_file(targz_path: str) -> None:
        '''A synchronous helper function to extract one or more tar.gz files.'''
        if tarfile.is_tarfile(targz_path):
            with tarfile.open(targz_path, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(targz_path), filter="fully_trusted")
            print(f"[OK] Extracted {targz_path}")
            os.remove(targz_path)
            print(f"[OK] Removed {targz_path}")
        else:
            print(f"[ERROR] {targz_path} is not a valid tar file.")

def parallel_extract(targz_paths: List[str]) -> None:
    '''Extract tar.gz files in parallel using multiprocessing.'''
    num_file, num_cpus = len(targz_paths), min(multiprocessing.cpu_count(), len(targz_paths))
    if num_file == 0:
        print("No .tar.gz files found for extraction.")
        return
    
    start_time = time.perf_counter()
    print(f"(Parallel) Extracting LibriTTS tar.gz files...")
    print(f"\t- Number of files: {num_file}")
    print(f"\t- Number of CPU cores: {num_cpus}")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.map(extract_targz_file, targz_paths)
    elapsed = time.perf_counter() - start_time
    print(f"Extraction completed: Took {elapsed:.2f} seconds.")


if __name__ == "__main__":
    pass