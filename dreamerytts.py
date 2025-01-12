import torch
from TTS.api import TTS

from config import ASSETS_DIR, MODELS_DIR


def tts(
    text: str,
    file_path: str,
    speaker_wav: str | None = None,
    speed: float = 2.0,
    language: str = "en",
) -> None:
    if speaker_wav is None:
        speaker_wav = str(ASSETS_DIR / "speaker.mp3")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODELS_DIR.absolute()
    config_path = MODELS_DIR / "config.json"

    tts = TTS(model_path=model_path, config_path=config_path).to(device)

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=file_path,
        speed=speed,
        language=language,
    )


if __name__ == "__main__":
    text = "Hello, world!"

    tts(
        text=text,
        file_path="output.wav",
    )
