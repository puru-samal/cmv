from .base_extractor import BaseExtractor
from .feature_extractors import Energy
import os
import torch
from ..utils import AudioUtils
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
import multiprocessing
import warnings
warnings.simplefilter("ignore", UserWarning)
from typing import Union, List

# Set up logging
log = logging.getLogger(__name__)

class EnergyExtractor(BaseExtractor):
    """Energy feature extractor class"""

    def __init__(self, roots: list[str], num_workers: int = 1, device: str = "cpu"):
        """Energy feature extractor class initializer

        Args:
            roots (list[str]): Root directories to recursively search and extract features from
            num_workers (int, optional): Number of workers to use for extraction. Defaults to 1.
            device (str, optional): Device to use for extraction. Defaults to "cpu".
        """
        super().__init__(roots, extensions=["wav"], num_workers=num_workers, device=device)

    def _run(self, files: Union[List[str], str], rank: int = 0) -> bool:
        """
        Run the Energy extractor on a list of files

        Args:
            files (list[str]): List of files to extract features from
            rank (int): Rank of the current process
        """
        torch.set_num_threads(1)
        energy_extractor = Energy()
        if not isinstance(files, list):
            files = [files]
        
        for file in files:
            try:
                # Load audio and convert to mono
                audio = AudioUtils.load_audio(file, sample_rate=16000)  # (channels, frames)
                audio = AudioUtils.to_mono(audio)
                energy = energy_extractor.extract(audio, sr=16000)  # (1, N)
                
                # Save energy
                energy = energy.squeeze(0)  # (N,)
                fname = os.path.basename(file).replace(".wav", ".energy")
                out_file = os.path.join(os.path.dirname(file), f"{fname}.pt")
                torch.save(energy, out_file)
            except Exception as e:
                log.error(f"[EnergyExtractor._run] Error processing {file}: {e}")
                return False
        return True

@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="energy_extractor")
def main(cfg: DictConfig) -> None:
    """Main function to run the Hubert extractor

    Args:
        cfg (DictConfig): Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = torch.cuda.device_count() if device == "cuda" else min(8, multiprocessing.cpu_count())
    log.info(f"[main] Using device: {device} with {num_workers} workers")
    extractor = EnergyExtractor(
        roots=cfg.root_dirs,
        num_workers=num_workers,
        device=device,
    )
    extractor.run()

if __name__ == "__main__":
    main()