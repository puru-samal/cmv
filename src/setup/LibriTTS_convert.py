from pathlib import Path
import multiprocessing
import tarfile
import os
import time
import hydra
from omegaconf import DictConfig
import logging
from typing import Tuple
import torchaudio
import torch
from tqdm import tqdm
from functools import partial
import warnings
warnings.simplefilter("ignore", UserWarning)

# Set up logging
log = logging.getLogger(__name__)

def convert_audio_file(file_path: Path, sample_rate_target: int, attempts: int = 3) -> Tuple[Path, bool]:
    '''A synchronous helper function to convert one or more audio files to a target sample rate.'''
    torch.set_num_threads(1)  # Limit PyTorch to use a single thread
    for attempt in range(attempts):
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != sample_rate_target:
                waveform_resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate_target)(waveform)
                torchaudio.save(file_path, waveform_resampled, sample_rate_target, bits_per_sample=16)
            return file_path, True
        except Exception as e:
            return file_path, False
    return file_path, False


@hydra.main(version_base=None, config_path="../../configs/setup", config_name="LibriTTS_convert")
def parallel_convert(cfg: DictConfig) -> None:
    '''Convert audio files in parallel using multiprocessing.'''
    root_dir = cfg.root_dir
    num_retries = cfg.num_retries
    sample_rate_target = cfg.sample_rate_target
    sub_directories_to_convert = cfg.sub_directories_to_convert
    audio_extension = cfg.audio_extension_source

    # Find all audio files in the specified subdirectories
    audio_files = []
    for sub_dir in sub_directories_to_convert:
        sub_dir_path = Path(root_dir) / sub_dir
        audio_files.extend(sub_dir_path.rglob(f"*{audio_extension}"))
    num_files, num_cpus = len(audio_files), min(multiprocessing.cpu_count(), len(audio_files))
    if num_files == 0:
        log.warning("No audio files found for conversion.")
        return
    
    # Start parallel conversion
    start_time = time.perf_counter()
    log.info(f"(Parallel) Converting audio files...")
    log.info(f"\t- Number of files: {num_files}")
    log.info(f"\t- Number of CPU cores used: {num_cpus}")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = list(tqdm(pool.imap_unordered(partial(convert_audio_file, sample_rate_target=sample_rate_target, attempts=num_retries), audio_files, chunksize=num_files // num_cpus), total=num_files, desc="[Converting audio files]"))
    elapsed = time.perf_counter() - start_time
    log.info(f"Conversion completed: Took {elapsed:.2f} seconds.")
    for file_path, success in results:
        if not success:
            log.error(f"[FAILED] Conversion failed for {file_path}")

if __name__ == "__main__":
    parallel_convert()