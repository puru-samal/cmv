import torch
from ...utils import AudioUtils


class Energy:
    """
    Energy extractor from audio 
    """

    @staticmethod
    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract energy from audio
        Args:
            audio (torch.Tensor): Audio tensor of shape (1, 1, T) where T is the number of samples
            sr (int): Sample rate of the audio
        Returns:
            torch.Tensor: Energy tensor of shape (1, N) where N is the number of frames
            - N = T // 320 is the number of frames
        """
        assert sr == 16000, "[Energy.extract_energy] Sample rate must be 16000"
        assert audio.ndim == 2, "[Energy.extract_energy] Audio must be 2D (num_channels, num_samples)"
        assert audio.shape[0] == 1, "[Energy.extract_energy] Audio must be mono"

        frame_length = 1024 # 20ms for 16000kHz

        # split wav into 64ms frames with 20ms overlap
        frames = AudioUtils.to_frames(audio.squeeze(0), frame_length=frame_length, hop_length=320) # (1024, N)
        
        energy = (frames ** 2).sum(dim = 0).unsqueeze(0)
        return energy