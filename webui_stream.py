import os
import sys
import threading
import time
import warnings
import numpy as np
import tempfile
import json
import aiohttp
from io import BytesIO
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr

# 假设您有一个FastAPI服务器运行在某个端口
FASTAPI_SERVER = "http://119.45.179.67:9880"  # 修改为您的FastAPI服务器地址

async def download_audio(audio_path):
    """处理本地音频文件路径"""
    return audio_path  # 如果是本地文件，直接返回路径

async def gen_single(prompts, text, progress=gr.Progress()):
    if isinstance(prompts, list):
        prompt_paths = [prompt.name for prompt in prompts if prompt is not None]
    else:
        prompt_paths = [prompts.name] if prompts is not None else []
    
    if not prompt_paths or not text:
        yield None
        return
    
    # 连接到FastAPI服务器进行流式TTS
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": text,
                "audio_paths": prompt_paths
            }
            
            async with session.post(
                f"{FASTAPI_SERVER}/tts_live_stream",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"API错误: {error_text}")
                    yield None
                    return
                
                # 创建临时文件来存储流式音频
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_path = temp_audio.name
                
                try:
                    # 收集所有音频数据
                    audio_data = bytearray()
                    
                    async for chunk in response.content.iter_any():
                        if chunk:
                            audio_data.extend(chunk)
                            
                            # 定期更新音频文件（模拟流式效果）
                            if len(audio_data) > 44100 * 2:  # 每约0.5秒更新一次
                                try:
                                    # 将RAW音频数据转换为WAV格式
                                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                                    sf.write(temp_path, audio_array, 24000, format='WAV')
                                    yield temp_path
                                except Exception as e:
                                    print(f"音频处理错误: {e}")
                                    continue
                    
                    # 最终写入完整的音频
                    if audio_data:
                        audio_array = np.frombuffer(audio_data, dtype=np.float32)
                        sf.write(temp_path, audio_array, 24000, format='WAV')
                        yield temp_path
                        
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
    except Exception as e:
        print(f"请求错误: {e}")
        yield None

def update_prompt_audio():
    return gr.update(interactive=True)

# 创建自定义HTML组件用于更好的流式播放体验
streaming_js = """
<script>
function updateAudioStream(audioUrl) {
    const audioElement = document.getElementById('streaming-audio');
    if (audioElement.src !== audioUrl) {
        audioElement.src = audioUrl;
        audioElement.play().catch(e => console.log('Autoplay prevented:', e));
    }
}
</script>
<audio id="streaming-audio" controls autoplay style="width: 100%"></audio>
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
                input_text_single = gr.TextArea(
                    label="请输入目标文本", 
                    key="input_text_single",
                    placeholder="请输入要转换为语音的文本..."
                )
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            output_audio = gr.Audio(
                label="生成结果", 
                visible=True, 
                key="output_audio",
                elem_id="streaming-audio"
            )

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
    demo.queue(concurrency_count=5)
    demo.launch(server_name="0.0.0.0", server_port=7860)