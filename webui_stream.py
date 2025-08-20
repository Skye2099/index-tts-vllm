import os
import sys
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
import numpy as np
from io import BytesIO
import tempfile

from indextts.infer_vllm_stream import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")

model_dir = "/data/wts/index-tts-vllm/pretrain/IndexTeam/IndexTTS-1.5"
gpu_memory_utilization = 0.25

cfg_path = os.path.join(model_dir, "config.yaml")
tts = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, gpu_memory_utilization=gpu_memory_utilization)

async def gen_single_direct(prompts, text, progress=gr.Progress()):
    if isinstance(prompts, list):
        prompt_paths = [prompt.name for prompt in prompts if prompt is not None]
    else:
        prompt_paths = [prompts.name] if prompts is not None else []
    
    if not prompt_paths or not text:
        yield None
        return
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_path = temp_audio.name
    
    try:
        audio_chunks = []
        
        # 直接使用IndexTTS的流式接口
        async for sr, pcm_data in tts.stream_infer(prompt_paths, text):
            # 收集音频数据
            audio_chunks.append(pcm_data)
            
            # 定期更新（模拟流式效果）
            if len(audio_chunks) > 3:  # 每3个块更新一次
                combined_audio = np.concatenate(audio_chunks)
                # 保存为WAV文件
                import scipy.io.wavfile as wavfile
                wavfile.write(temp_path, sr, combined_audio)
                yield temp_path
        
        # 最终结果
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, sr, combined_audio)
            yield temp_path
            
    except Exception as e:
        print(f"生成错误: {e}")
        yield None
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

def update_prompt_audio():
    return gr.update(interactive=True)

# 创建自定义HTML组件用于更好的流式播放体验
streaming_js = """
<script>
function streamAudio(audioUrl) {
    const audioElement = document.getElementById('streaming-audio');
    audioElement.src = audioUrl;
    audioElement.play();
}
</script>
<audio id="streaming-audio" controls autoplay></audio>
"""

with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
    ''')
    
    # 添加自定义JS
    gr.HTML(streaming_js)
    
    with gr.Tab("音频生成"):
        with gr.Row():
            prompt_audio = gr.File(
                label="请上传参考音频（可上传多个）",
                file_count="multiple",
                file_types=["audio"]
            )
            with gr.Column():
                input_text_single = gr.TextArea(label="请输入目标文本", key="input_text_single")
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True, key="output_audio")

    prompt_audio.upload(
        update_prompt_audio,
        inputs=[],
        outputs=[gen_button]
    )

    gen_button.click(
        gen_single,
        inputs=[prompt_audio, input_text_single],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="0.0.0.0")