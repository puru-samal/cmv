import glob
import os
from abc import abstractmethod
import multiprocessing as mp
from multiprocessing import Pool
import logging
from pathlib import Path
import time
from functools import partial
from tqdm import tqdm

# Set up logging
log = logging.getLogger(__name__)

class BaseExtractor:
    """Base extractor class"""

    def __init__(self, roots: list[str], extensions: list[str], num_workers: int = 1, device: str = "cpu"):
        """Base extractor class initializer

        Args:
            roots (list[str]): Root directories to recursively search and extract features from
            extensions (list[str]): List of extensions to extract features from. Must be supported by the extractor.
            num_workers (int, optional): Number of workers to use for extraction. Defaults to 1.
            device (str, optional): Device to use for extraction. Defaults to "cpu".
        """
        self.num_workers = num_workers
        self.device = device
        self.extensions = extensions
        
        self.files = []
        self.roots = [Path(root) for root in roots]
        for root in self.roots:
            for ext in extensions:
                self.files.extend(root.rglob(f"*.{ext}"))
        log.info(f"[{self.__class__.__name__}] Found {len(self.files)} files to extract features from")

    @abstractmethod
    def _run(self, files: str, rank: int = 0) -> bool:
        """
        Abstract method to run the extractor on a list of files

        Args:
            files (str): File path to extract features from
            rank (int): Rank of the current process
        """
        raise NotImplementedError(f"[{self.__class__.__name__}._run] Not implemented")

    def run(self):
        if self.num_workers <= 1:
            self._run(self.files)
            return

        start = time.perf_counter()
        log.info(f"[{self.__class__.__name__}] Running extractor with {self.num_workers} workers")
        ctx = mp.get_context("spawn")   # safer with CUDA
        with ctx.Pool(self.num_workers) as p:
            if self.device == 'cuda': # Reloading model per file too expensive!
                fn_args = [(self.files[i::self.num_workers], i % self.num_workers) for i in range(self.num_workers)]
                statuses = p.starmap(self._run, fn_args, chunksize=len(self.files) // self.num_workers)
            else:
                statuses = list(tqdm(p.imap_unordered(partial(self._run, rank=0), self.files, chunksize=len(self.files) // self.num_workers), total=len(self.files), desc=f"[{self.__class__.__name__}] Extracting features"))
            assert all(status for status in statuses), f"[{self.__class__.__name__}] Error running extractor"
        log.info(f"[{self.__class__.__name__}] Extractor finished running")
        end = time.perf_counter()
        log.info(f"[{self.__class__.__name__}] Extraction took {end - start:.2f} seconds")