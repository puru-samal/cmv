from .base_extractor import BaseExtractor
from .feature_extractors import F0
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

class F0Extractor(BaseExtractor):
    """F0 feature extractor class"""

    def __init__(self, roots: list[str], num_workers: int = 1, device: str = "cpu"):
        """F0 feature extractor class initializer

        Args:
            roots (list[str]): Root directories to recursively search and extract features from
            num_workers (int, optional): Number of workers to use for extraction. Defaults to 1.
            device (str, optional): Device to use for extraction. Defaults to "cpu".
        """
        super().__init__(roots, extensions=["wav"], num_workers=num_workers, device=device)
    
    def _normalize(self, pitch: torch.Tensor) -> torch.Tensor:
        """Normalize pitch values to log-scale and standardize

        Args:
            pitch (torch.Tensor): Pitch tensor of shape (N,)

        Returns:
            torch.Tensor: Normalized pitch tensor of shape (N,)
        """
        non_zero = pitch > 0
        if len(non_zero) == 0:
            return pitch
        mean = pitch[non_zero].mean()
        std = pitch[non_zero].std()
        pitch[non_zero] = (pitch[non_zero] - mean) / std

        pitch_norm = torch.zeros_like(pitch)
        pitch_norm[non_zero] = pitch[non_zero]
        return pitch_norm

    def _run(self, files: Union[List[str], str], rank: int = 0) -> bool:
        """
        Run the F0 extractor on a list of files

        Args:
            files (list[str]): List of files to extract features from
            rank (int): Rank of the current process
        """
        torch.set_num_threads(1)
        f0_extractor = F0()
        if not isinstance(files, list):
            files = [files]
        
        for file in files:
            try:
                # Load audio and convert to mono
                audio = AudioUtils.load_audio(file, sample_rate=16000)  # (channels, frames)
                audio = AudioUtils.to_mono(audio)
                audio = audio.squeeze(0)  # (frames,)

                pitches, cmdf_vs, cmdf_uvs = [], [], []
                # Extract pitch, cmdf_v, cmdf_uv
                for threshold in [0.05, 0.1, 0.15]:
                    pitch, cmdf_v, cmdf_uv = f0_extractor.estimate(audio, sample_rate=16000, threshold=threshold)  # (N,), (N,), (N,)
                    pitch = self._normalize(pitch)

                    pitches.append(pitch)
                    cmdf_vs.append(cmdf_v)
                    cmdf_uvs.append(cmdf_uv)
                
                pitches = torch.stack(pitches, dim=0)   # (3, N)
                cmdf_vs = torch.stack(cmdf_vs, dim=0)   # (3, N)
                cmdf_uvs = torch.stack(cmdf_uvs, dim=0) # (3, N)
                
                # Save F0
                f0 = torch.cat([pitches, cmdf_vs, cmdf_uvs], dim=0)  # (9, N)
                fname = os.path.basename(file).replace(".wav", ".f0")
                out_file = os.path.join(os.path.dirname(file), f"{fname}.pt")
                torch.save(f0, out_file)
            except Exception as e:
                log.error(f"[F0Extractor._run] Error processing {file}: {e}")
                return False
        return True

@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="f0_extractor")
def main(cfg: DictConfig) -> None:
    """Main function to run the Hubert extractor

    Args:
        cfg (DictConfig): Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = torch.cuda.device_count() if device == "cuda" else min(8, multiprocessing.cpu_count())
    log.info(f"[main] Using device: {device} with {num_workers} workers")
    extractor = F0Extractor(
        roots=cfg.root_dirs,
        num_workers=num_workers,
        device=device,
    )
    extractor.run()

if __name__ == "__main__":
    main()