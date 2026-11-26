# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Built-in voices', 'Quick voice clone', 'Cross-lingual clone', 'Natural language control']
instruct_dict = {
                 'Built-in voices': '1. Pick a built-in voice\n2. Click Generate audio',
                 'Quick voice clone': '1. Choose a prompt audio file, or record your own (max 30s; if both are provided, the uploaded file wins)\n2. Type the prompt transcript (must match what is said in the audio)\n3. Click Generate audio',
                 'Cross-lingual clone': '1. Choose a prompt audio file, or record your own (max 30s)\n2. Click Generate audio',
                 'Natural language control': '1. Pick a built-in voice\n2. Type instruct text\n3. Click Generate audio'}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['Natural language control']:
        if instruct_text == '':
            gr.Warning('Natural language control mode: please enter instruct text')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('Natural language control mode: prompt audio/prompt text will be ignored')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['Cross-lingual clone']:
        if instruct_text != '':
            gr.Info('Cross-lingual clone mode: instruct text will be ignored')
        if prompt_wav is None:
            gr.Warning('Cross-lingual clone mode: please provide a prompt audio file')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('Cross-lingual clone mode: make sure the synthesis text and prompt text are in different languages')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['Quick voice clone', 'Cross-lingual clone']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty - did you forget to provide one?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate {} is lower than the required {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['Built-in voices']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('Built-in voices mode: prompt text/prompt audio/instruct text will be ignored!')
        if sft_dropdown == '':
            gr.Warning('No built-in voices available!')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['Quick voice clone']:
        if prompt_text == '':
            gr.Warning('Prompt text is empty - did you forget to type the transcript?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('Quick voice clone mode: built-in voices/instruct text will be ignored!')

    if mode_checkbox_group == 'Built-in voices':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Quick voice clone':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-lingual clone':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks(title="CosyVoice Text-to-Speech") as demo:
        gr.Markdown("### Codebase: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    Pretrained models: [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Enter the text to synthesize, choose an inference mode, and follow the step hints")

        tts_text = gr.Textbox(label="Text to synthesize", lines=1, value="你好，欢迎来学中文。今天天气很好，我们一起读课文吧。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Inference mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Built-in voice', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Stream inference', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed (non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload prompt audio (3-10s, sample rate >= 16khz)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio')
        prompt_text = gr.Textbox(label="Prompt transcript", lines=1, placeholder="Type what is said in the prompt audio (no auto transcription yet)...", value='')
        instruct_text = gr.Textbox(label="Instruct text", lines=1, placeholder="e.g. 用开心的语气说 This is a happy voice.", value='')

        generate_button = gr.Button("Generate audio")

        audio_output = gr.Audio(label="Synthesized audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
