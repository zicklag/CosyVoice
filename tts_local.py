"""Synthesize Chinese speech to WAV with CosyVoice2-0.5B.

Usage:
    pixi run tts                          # synthesize the default demo text
    pixi run tts -- --text "你好世界"      # custom text
    pixi run tts -- --text "..." --out hello.wav
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'Matcha-TTS'))

import torch  # noqa: E402
import torchaudio  # noqa: E402

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.file_utils import logging  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402

DEFAULT_TEXT = '你好，欢迎来学中文。今天天气很好，我们一起读课文吧。'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default=DEFAULT_TEXT)
    parser.add_argument('--out', type=str, default='tts_output.wav')
    parser.add_argument('--model-dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--prompt-wav', type=str, default='asset/zero_shot_prompt.wav',
                        help='voice prompt wav (3-10s), reused as the speaking voice')
    parser.add_argument('--prompt-text', type=str,
                        default='希望你以后能够做的比我还好呦。', help='transcript of the prompt wav')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        sys.exit(f'model dir {args.model_dir} missing - run: pixi run get-model')

    # fp16 autocast keeps the 0.5B LLM within 4GB VRAM on a GTX 1650
    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=True)
    set_all_random_seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for i, chunk in enumerate(cosyvoice.inference_zero_shot(
            args.text, args.prompt_text, args.prompt_wav, stream=False)):
        out_i = args.out if i == 0 else args.out.replace('.wav', f'_{i}.wav')
        torchaudio.save(out_i, chunk['tts_speech'], cosyvoice.sample_rate)
        print(f'saved {out_i} ({chunk["tts_speech"].shape[1] / cosyvoice.sample_rate:.1f}s @ {cosyvoice.sample_rate}Hz)')


if __name__ == '__main__':
    main()
