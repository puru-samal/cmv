from .base_extractor import BaseExtractor
from .feature_extractors import Hubert
from ..utils import AudioUtils
import os
import torch
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
import multiprocessing
import warnings
warnings.simplefilter("ignore", UserWarning)

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

    def _run(self, files: list[str], rank: int) -> bool:
        """
        Run the Hubert extractor on a list of files

        Args:
            files (list[str]): List of files to extract features from
            rank (int): Rank of the current process
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank) if self.device == "cuda" else ""
        model = Hubert(device=self.device)
        for file in files:
            try:
                # Load audio and convert to mono
                audio = AudioUtils.load_audio(file, sample_rate=16000)  # (channels, frames)
                audio = AudioUtils.to_mono(audio)
                # Extract units
                units = model.extract_units(audio.unsqueeze(0), sr=16000)  # (1, N, D)
                units = units.squeeze(0)  # (N, D)
                # Save units
                fname = os.path.basename(file).replace(".wav", ".hunits")
                out_file = os.path.join(os.path.dirname(file), f"{fname}.pt")
                torch.save(units, out_file)
            except Exception as e:
                log.error(f"[HubertExtractor._run] Error processing {file}: {e}")
                return False
        return True

@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="hubert_extractor")
def main(cfg: DictConfig) -> None:
    """Main function to run the Hubert extractor

    Args:
        cfg (DictConfig): Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = torch.cuda.device_count() if device == "cuda" else multiprocessing.cpu_count()
    log.info(f"[main] Using device: {device} with {num_workers} workers")
    extractor = HubertExtractor(
        roots=cfg.root_dirs,
        num_workers=num_workers,
        device=device,
    )
    extractor.run()

if __name__ == "__main__":
    main()