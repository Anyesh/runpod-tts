import torch
from TTS.api import TTS

from config import ASSETS_DIR, MODELS_DIR


def tts(text: str, file_path: str, lang: str = "en", speaker_wav: str | None = None):
    if speaker_wav is None:
        speaker_wav = str(ASSETS_DIR / "speaker.mp3")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODELS_DIR.absolute()
    config_path = MODELS_DIR / "config.json"

    tts = TTS(model_path=model_path, config_path=config_path).to(device)

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=lang,
        file_path=file_path,
    )


if __name__ == "__main__":
    text = "Hello, world!"

    tts(
        text=text,
        file_path="output.wav",
    )
