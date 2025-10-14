from pathlib import Path
import multiprocessing
import tarfile
import os
import time
import hydra
from omegaconf import DictConfig

def extract_targz_file(targz_path: str, extract_dir: str) -> None:
        '''A synchronous helper function to extract one or more tar.gz files.'''
        if tarfile.is_tarfile(targz_path):
            with tarfile.open(targz_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir, filter="fully_trusted")
            print(f"[OK] Extracted {targz_path} to {extract_dir}")
            os.remove(targz_path)
            print(f"[OK] Removed {targz_path}")
        else:
            print(f"[ERROR] {targz_path} is not a valid tar file.")

@hydra.main(version_base=None, config_path="../../configs/setup", config_name="LibriTTS_extract")
def parallel_extract(cfg: DictConfig) -> None:
    '''Extract tar.gz files in parallel using multiprocessing.'''
    # Config parameters
    data_dir = cfg.data_dir
    extract_dir = cfg.extract_dir
    extension = cfg.extension

    # Find all .tar.gz files in the data directory
    targz_paths = list(Path(data_dir).rglob(f"*{extension}"))
    extract_dir = os.path.abspath(extract_dir)
    num_file, num_cpus = len(targz_paths), min(multiprocessing.cpu_count(), len(targz_paths))
    if num_file == 0:
        print("No .tar.gz files found for extraction.")
        return
    
    start_time = time.perf_counter()
    print(f"(Parallel) Extracting LibriTTS tar.gz files...")
    print(f"\t- Number of files: {num_file}")
    print(f"\t- Number of CPU cores: {num_cpus}")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.starmap(extract_targz_file, [(path, extract_dir) for path in targz_paths])
    elapsed = time.perf_counter() - start_time
    print(f"Extraction completed: Took {elapsed:.2f} seconds.")


if __name__ == "__main__":
    parallel_extract()