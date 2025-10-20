import glob
import os
from abc import abstractmethod
from multiprocessing import Pool

class BaseExtractor:
    """Base extractor class"""

    def __init__(self, root: str, extensions: list[str], num_workers: int = 1, device: str = "cpu"):
        """Base extractor class initializer

        Args:
            root (str): Root directory to recursively search and extract features from
            extensions (list[str]): List of extensions to extract features from. Must be supported by the extractor.
            num_workers (int, optional): Number of workers to use for extraction. Defaults to 1.
            device (str, optional): Device to use for extraction. Defaults to "cpu".
        """
        self.root = root
        self.num_workers = num_workers
        self.device = device
        self.extensions = extensions
        self.files = glob.glob(os.path.join(self.root, f"**/*.{self.extensions}"), recursive=True)
        print(f"[BaseExtractor] Found {len(self.files)} files to extract features from")

    @abstractmethod
    def _run(self, files: list[str]) -> bool:
        """
        Abstract method to run the extractor on a list of files

        Args:
            files (list[str]): List of files to extract features from
        """
        raise NotImplementedError(f"[{self.__class__.__name__}._run] Not implemented")

    def run(self):
        if self.num_workers <= 1:
            self._run(self.files)
            return
        
        print(f"[BaseExtractor] Running extractor with {self.num_workers} workers")
        with Pool(self.num_workers) as p:
            statuses = p.map(self._run, self.files, chunksize = len(self.files) // self.num_workers)
            assert all(status == 0 for status in statuses), "[BaseExtractor] Error running extractor"
        print(f"[BaseExtractor] Extractor finished running")