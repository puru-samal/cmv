import torch
import torchaudio
import librosa
import numpy as np


class AudioUtils:
    """
    An Audio utilities class with static methods containing common audio operations.
    """    
    @staticmethod
    def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
        """
        Load an audio file into a tensor
        Args:
            audio_path (Path | str): The path to the audio file
            sample_rate (int): The sample rate of the audio file
        Returns:
            torch.Tensor: The audio tensor of shape (channels, frames)
        """
        audio, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
        return audio

    @staticmethod
    def to_mono(audio: torch.Tensor) -> torch.Tensor:
        """
        Convert an audio tensor to mono
        Args:
            audio (torch.Tensor): The audio tensor of shape (channels, frames)
        Returns:
            torch.Tensor: The audio tensor of shape (frames,)
        """
        return audio.mean(dim=0) if audio.shape[0] > 1 else audio

    @staticmethod
    def random_segment(audio: torch.Tensor, segment_length: int) -> torch.Tensor:
        """
        Randomly segment an audio tensor
        Args:
            audio (torch.Tensor): The audio tensor of shape (num_channels, frames,)
            segment_length (int): The length of the segment in frames
        Returns:
            torch.Tensor: The audio tensor of shape (segment_length,)
        """
        max_frames = audio.shape[1]
        if max_frames < segment_length:
            return audio
        start_frame = torch.randint(0, max_frames - segment_length)
        return audio[:, start_frame : start_frame + segment_length]

    @staticmethod
    def to_frames(audio: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor | None:
        """
        Convert an audio tensor to frames
        Args:
            audio (torch.Tensor): The audio tensor of shape (num_channels, frames)
            frame_length (int): The length of the frame in frames
            hop_length (int): The hop length in frames
        Returns:
            torch.Tensor: The audio tensor of shape (num_channels, frame_length, num_frames)
            None: If an error occurs
        """
        try:
            audio_np = audio.numpy()
            pad_length = (audio.shape[1] // hop_length) * hop_length + frame_length - audio.shape[1]
            audio_np = np.pad(audio_np, (0, pad_length), mode='constant')
            frames = librosa.util.frame(audio_np, frame_length=frame_length, hop_length=hop_length)
            return torch.from_numpy(frames)
        except Exception as e:
            print(f"Error converting audio to frames: {e}")
            return None


if __name__ == "__main__":
    import os
    path = os.path.join("data", "audio", "audio.wav")
    audio, sr = AudioUtils.load_audio(path)
    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")
    
    mono_audio = AudioUtils.to_mono(audio)
    print(f"Mono audio shape: {mono_audio.shape}")
    
    segment_dur = 1.0 # seconds
    segment_length = int(segment_dur * sr)
    seq_audio = AudioUtils.random_segment(mono_audio, segment_length)
    print(f"Sequence audio shape: {seq_audio.shape}")
    
    frames = AudioUtils.to_frames(audio, frame_length = 1024, hop_length = 512)
    print(f"Frames shape: {frames.shape}")
