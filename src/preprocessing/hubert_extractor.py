from .base_extractor import BaseExtractor
from .feature_extractors import Hubert
from ..utils import AudioUtils
import os
import torch
import logging

# Set up logging
log = logging.getLogger(__name__)



class HubertExtractor(BaseExtractor):
    """Hubert feature extractor class"""

    def __init__(self, root: str, num_workers: int = 1, device: str = "cpu"):
        """Hubert feature extractor class initializer

        Args:
            root (str): Root directory to recursively search and extract features from
            num_workers (int, optional): Number of workers to use for extraction. Defaults to 1.
            device (str, optional): Device to use for extraction. Defaults to "cpu".
        """
        super().__init__(root, extensions=["wav"], num_workers=num_workers, device=device)
        self.hubert = Hubert(device=self.device)

    def _run(self, files: list[str]) -> bool:
        """
        Run the Hubert extractor on a list of files

        Args:
            files (list[str]): List of files to extract features from
        """
        for file in files:
            # Load audio and convert to mono
            audio, sr = AudioUtils.load_audio(file)
            audio = AudioUtils.to_mono(audio)
            # Extract units
            units = self.hubert.extract_units(audio.unsqueeze(0), sr)  # (1, N, D)
            if units is None:
                print(f"[HubertExtractor._run] Error extracting units from {file}")
                return False
            units = units.squeeze(0)  # (N, D)
            # Save units
            fname = os.path.basename(file).rsplit(".")[0] + "_hubert_units"
            out_file = os.path.join(os.path.dirname(file), f"{fname}.pt")
            torch.save(units, out_file)
            print(f"[HubertExtractor._run] Extracted units from {file} and saved to {out_file}")
        return True