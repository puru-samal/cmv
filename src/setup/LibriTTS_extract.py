from pathlib import Path
import multiprocessing
import tarfile
import os
import time
import hydra
from omegaconf import DictConfig
import logging

# Set up logging
log = logging.getLogger(__name__)

def extract_targz_file(targz_path: str, extract_dir: str, attempts: int = 3) -> None:
    '''A synchronous helper function to extract one or more tar.gz files.'''
    if not tarfile.is_tarfile(targz_path):
        log.error(f"[ERROR] {targz_path} is not a valid tar file.")
        return
    for attempt in range(attempts):
        try:
            with tarfile.open(targz_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir, filter="fully_trusted")
            log.info(f"[OK] Extracted {targz_path} to {extract_dir}")
            os.remove(targz_path)
            log.info(f"[OK] Removed {targz_path}")
            return
        except Exception as e:
            log.error(f"[ERROR] Attempt {attempt + 1}/{attempts} failed: {e}")
    log.error(f"[ERROR] Failed to extract {targz_path} after {attempts} attempts.")
    return

@hydra.main(version_base=None, config_path="../../configs/setup", config_name="LibriTTS_extract")
def parallel_extract(cfg: DictConfig) -> None:
    '''Extract tar.gz files in parallel using multiprocessing.'''
    # Config parameters
    data_dir = cfg.data_dir
    extract_dir = cfg.extract_dir
    num_retries = cfg.num_retries
    targz_paths = [os.path.join(data_dir, f) for f in cfg.files_to_extract]

    # Find all .tar.gz files in the data directory
    extract_dir = os.path.abspath(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    num_file, num_cpus = len(targz_paths), min(multiprocessing.cpu_count(), len(targz_paths))
    if num_file == 0:
        log.warning("No .tar.gz files found for extraction.")
        return
    
    # Start parallel extraction
    start_time = time.perf_counter()
    log.info(f"(Parallel) Extracting LibriTTS tar.gz files...")
    log.info(f"\t- Number of files: {num_file}")
    log.info(f"\t- Number of CPU cores used: {num_cpus}")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.starmap(extract_targz_file, [(path, extract_dir, num_retries) for path in targz_paths])
    elapsed = time.perf_counter() - start_time
    log.info(f"Extraction completed: Took {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parallel_extract()