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
    num_file, num_cpus = len(targz_paths), multiprocessing.cpu_count()
    if num_file == 0:
        print("No .tar.gz files found for extraction.")
        return
    
    start_time = time.perf_counter()
    print(f"(Parallel) Extracting LibriTTS tar.gz files...")
    print(f"\t- Number of files: {num_file}")
    print(f"\t- Number of CPU cores: {num_cpus}")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(extract_targz_file, targz_paths)
    elapsed = time.perf_counter() - start_time
    print(f"Extraction completed: Took {elapsed:.2f} seconds.")

def summarize_libriTTS(root: str):
    """
    Fast summary for a LibriTTS / LibriSpeech-style dataset.
    Prioritizes speed — avoids reading audio metadata.
    Reports counts, size, and directory-level structure.
    """
    root = Path(root)
    if not root.exists():
        print(f"[ERROR] Directory not found: {root}")
        return

    print(f"📊 Scanning dataset at: {root}")
    print("-" * 60)

    exts = {'.wav', '.flac'}
    audio_count = 0
    txt_count = 0
    total_size = 0
    speakers = set()
    chapters = set()
    splits = defaultdict(int)

    # one pass walk
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()

            if ext in exts:
                audio_count += 1
                total_size += fpath.stat().st_size
                parts = fpath.relative_to(root).parts
                if len(parts) >= 3:
                    speakers.add(parts[-3])
                    chapters.add(parts[-2])
                if len(parts) >= 1:
                    splits[parts[0]] += 1
            elif ext == '.txt':
                txt_count += 1

    avg_size = total_size / audio_count if audio_count else 0
    gb_size = total_size / 1e9

    print(f"🎧 Audio files:     {audio_count:,}")
    print(f"🗣️  Speakers:       {len(speakers):,}")
    print(f"📚 Chapters:       {len(chapters):,}")
    print(f"📝 Transcripts:    {txt_count:,}")
    print(f"📦 Total size:     {gb_size:.2f} GB")
    print(f"📈 Avg file size:  {avg_size / 1024:.1f} KB")
    if splits:
        print("📂 Files per split:")
        for split, count in sorted(splits.items()):
            print(f"   - {split}: {count:,}")
    print("-" * 60)
    