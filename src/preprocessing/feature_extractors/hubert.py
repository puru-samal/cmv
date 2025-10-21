import torch 
from typing import Literal
import os
import torch.nn as nn
import logging

# Set up logging
logger = logging.getLogger(__name__)

class Hubert:
    """
    Hubert Model used to produce soft speech units for content encoder training.
    Reference:
        https://ieeexplore.ieee.org/abstract/document/9746484
        https://github.com/bshall/hubert
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        try:
            self._load_model("soft")
        except Exception as e:
            logger.error(f"[Hubert.__init__] Error loading model: {e}")

    def _load_model(self, type: Literal["soft", "discrete"]):
        """
        Load the Hubert model
        Args:
            type (Literal["soft", "discrete"]): Type of Hubert model to load
        """
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        os.environ["TORCH_HOME"] = project_dir # Will create a "hub/bshall_hubert_main" directory here
        logger.info(f"[Hubert._load_model] TORCH_HOME set to {os.environ['TORCH_HOME']}")
        if type not in ["soft", "discrete"]:
            raise ValueError("[Hubert._load_model] Invalid type: must be 'soft' or 'discrete'")
        logger.info("Loading Hubert Model and checkpoint...")
        type = "hubert_soft" if type == "soft" else "hubert_discrete"
        self.model = torch.hub.load("bshall/hubert:main", type, trust_repo=True).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"[Hubert._load_model] Loaded Hubert ({type}) model")

    def extract_units(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract units from audio using Hubert model
        Args:
            audio (torch.Tensor): Audio tensor of shape (1, 1, T) where T is the number of samples
            sr (int): Sample rate of the audio
        Returns:
            torch.Tensor: Speech units of shape (1, N, D) where N is the number of frames and D is the number of units
                - N = T // 320 is the number of frames
                - D = 256 is the number of units
        """
        if self.model is None:
            print(f"[Hubert.extract_units] Error loading model...")
            return None
        assert sr == 16000, "[Hubert.extract_units] Sample rate must be 16000"
        assert audio.ndim == 3, "[Hubert.extract_units] Audio must be 3D (batch, channels, num_samples)"
        assert audio.shape[1] == 1, "[Hubert.extract_units] Audio must be mono"

        audio = audio.to(self.device)
        with torch.no_grad():
            units = self.model.units(audio)
        return units.cpu()


if __name__ == "__main__":
    #from data import AudioUtils
    #from pathlib import Path
    
    #audio, sr = AudioUtils.load_audio(Path("data/audio/audio.wav"))
    #audio = AudioUtils.to_mono(audio) # (1, T)
    hubert = Hubert(device="cpu")
    #units = hubert.extract_units(audio.unsqueeze(0), sr)
    #print(units.shape, audio.shape[-1] // 320)