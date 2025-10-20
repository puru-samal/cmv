from pathlib import Path

all_files = list(Path("data/LibriTTS").rglob("*.*"))
non_audio_files = [f for f in all_files if f.suffix != ".wav"]
print(f"Found {len(non_audio_files)} non-audio files.")
total_file_size = sum(file.stat().st_size for file in non_audio_files)
print(f"Total size of non-audio files: {total_file_size / (1024 * 1024):.2f} MB")