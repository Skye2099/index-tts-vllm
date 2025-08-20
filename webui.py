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
import tempfile
import json

from indextts.infer_vllm import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")

model_dir = "/data/wts/index-tts-vllm/pretrain/IndexTeam/IndexTTS-1.5"
gpu_memory_utilization = 0.25

cfg_path = os.path.join(model_dir, "config.yaml")
tts = IndexTTS(model_dir=model_dir, cfg_path=cfg_path, gpu_memory_utilization=gpu_memory_utilization)

# 自定义JavaScript代码，用于自动播放音频
play_audio_js = """
function playAudio(audioPath) {
    console.log('Attempting to play audio:', audioPath);
    
    // 找到输出音频容器
    const outputContainer = document.getElementById('output-audio-container');
    
    // 创建新的音频元素
    const audioElement = document.createElement('audio');
    audioElement.src = audioPath;
    audioElement.controls = true;
    audioElement.autoplay = true;
    audioElement.style.width = '100%';
    
    // 清空容器并添加音频元素
    outputContainer.innerHTML = '';
    outputContainer.appendChild(audioElement);
    
    // 添加事件监听器
    audioElement.addEventListener('canplay', function() {
        console.log('Audio can play now');
        // 尝试播放
        audioElement.play().catch(function(error) {
            console.log('Autoplay prevented:', error);
            // 显示播放按钮让用户手动点击
            const playButton = document.createElement('button');
            playButton.textContent = '点击播放音频';
            playButton.style.marginTop = '10px';
            playButton.style.padding = '10px 20px';
            playButton.style.backgroundColor = '#4CAF50';
            playButton.style.color = 'white';
            playButton.style.border = 'none';
            playButton.style.borderRadius = '5px';
            playButton.style.cursor = 'pointer';
            playButton.onclick = function() {
                audioElement.play();
                playButton.style.display = 'none';
            };
            outputContainer.appendChild(playButton);
        });
    });
    
    audioElement.addEventListener('error', function(e) {
        console.error('音频加载错误:', e);
        outputContainer.innerHTML = '<p style="color: red;">音频加载失败，请重试或检查浏览器控制台</p>';
    });
    
    return audioElement;
}

// 全局函数，供Gradio调用
window.playGeneratedAudio = function(audioPath) {
    return playAudio(audioPath);
}
"""

async def gen_single(prompts, text, progress=gr.Progress()):
    output_path = None
    tts.gr_progress = progress
    
    if isinstance(prompts, list):
        prompt_paths = [prompt.name for prompt in prompts if prompt is not None]
    else:
        prompt_paths = [prompts.name] if prompts is not None else []
    
    # 创建临时文件用于存储生成的音频
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        output_path = tmp_file.name
    
    # 生成音频
    output = await tts.infer(prompt_paths, text, output_path, verbose=True)
    
    # 返回音频路径和JavaScript代码以触发自动播放
    js_code = f"""
    <script>
        if (window.playGeneratedAudio) {{
            window.playGeneratedAudio('{output_path}');
        }} else {{
            console.error('playGeneratedAudio function not found');
        }}
    </script>
    """
    
    return output_path, gr.update(visible=True), js_code

def update_prompt_audio():
    return gr.update(interactive=True)

with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
    <h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
    ''')
    
    # 添加自定义JavaScript
    gr.HTML(f"<script>{play_audio_js}</script>")
    
    with gr.Tab("音频生成"):
        with gr.Row():
            # 使用 gr.File 替代 gr.Audio 来支持多文件上传
            prompt_audio = gr.File(
                label="请上传参考音频（可上传多个）",
                file_count="multiple",
                file_types=["audio"]
            )
            with gr.Column():
                input_text_single = gr.TextArea(label="请输入目标文本", key="input_text_single")
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            
            # 创建一个容器用于放置音频元素
            output_audio_container = gr.HTML(
                value="<div id='output-audio-container' style='min-height: 100px; border: 1px dashed #ccc; padding: 10px; margin-top: 10px;'>音频将在这里播放</div>",
                elem_id="output-audio-container"
            )
            
            # 保留原有的音频组件用于显示和下载
            output_audio = gr.Audio(label="生成结果（用于下载）", visible=True, key="output_audio")
            
            # 隐藏的组件用于存储JavaScript代码
            js_output = gr.HTML(visible=False)

    prompt_audio.upload(
        update_prompt_audio,
        inputs=[],
        outputs=[gen_button]
    )

    gen_button.click(
        gen_single,
        inputs=[prompt_audio, input_text_single],
        outputs=[output_audio, output_audio_container, js_output]
    )

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="0.0.0.0")