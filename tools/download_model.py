"""Download CosyVoice2-0.5B from HuggingFace (skips non-inference files)."""
import os
import sys
import urllib.request

REPO = "FunAudioLLM/CosyVoice2-0.5B"
DEST = "pretrained_models/CosyVoice2-0.5B"
FILES = [
    "campplus.onnx",
    "config.json",
    "configuration.json",
    "cosyvoice2.yaml",
    "flow.pt",
    "hift.pt",
    "llm.pt",
    "speech_tokenizer_v2.onnx",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
]

BASE = f"https://huggingface.co/{REPO}/resolve/main/"


def fetch(remote: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  exists: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = BASE + remote
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest + ".part")
    os.rename(dest + ".part", dest)


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    only = sys.argv[1:]
    for f in FILES:
        if only and not any(p in f for p in only):
            continue
        fetch(f, os.path.join(DEST, f))
    print("done")


if __name__ == "__main__":
    main()
