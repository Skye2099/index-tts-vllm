import gradio as gr
import wave
import numpy as np
from time import sleep
import io

def stream_wav_audio(wav_file_path, chunk_duration=1.0):
    """
    读取WAV文件并实时流式播放
    
    参数:
    wav_file_path: WAV文件路径
    chunk_duration: 每个音频块的长度（秒）
    """
    try:
        # 打开WAV文件
        with wave.open(wav_file_path, 'rb') as wav_file:
            # 获取音频参数
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            
            # 计算每个块的帧数
            frames_per_chunk = int(frame_rate * chunk_duration)
            
            print(f"音频信息: 采样率={frame_rate}Hz, 声道数={n_channels}, "
                  f"采样宽度={sample_width}字节, 块大小={frames_per_chunk}帧")
            
            # 读取并流式传输音频数据
            while True:
                # 读取一个块的帧数据
                frames = wav_file.readframes(frames_per_chunk)
                
                # 如果没有更多数据，退出循环
                if not frames:
                    break
                
                # 将帧数据转换为numpy数组（用于验证）
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # 创建一个新的WAV文件在内存中（包含当前块的音频）
                with io.BytesIO() as buffer:
                    with wave.open(buffer, 'wb') as chunk_wav:
                        chunk_wav.setnchannels(n_channels)
                        chunk_wav.setsampwidth(sample_width)
                        chunk_wav.setframerate(frame_rate)
                        chunk_wav.writeframes(frames)
                    
                    # 获取字节数据并返回
                    buffer.seek(0)
                    yield buffer.read()
                
                # 模拟实时处理延迟
                sleep(chunk_duration * 0.9)  # 稍微快于块长度以确保连续播放
    
    except Exception as e:
        print(f"处理音频时出错: {e}")
        yield None

def process_and_stream(audio_file):
    """
    处理音频文件并流式播放
    """
    if audio_file is None:
        return
    
    # 检查文件类型
    if not audio_file.lower().endswith('.wav'):
        print("请上传WAV格式的音频文件")
        yield None
        return
    
    # 流式播放WAV文件
    for chunk in stream_wav_audio(audio_file, chunk_duration=1.0):
        if chunk is not None:
            yield chunk
        else:
            break

# 创建Gradio界面
demo = gr.Interface(
    fn=process_and_stream,
    inputs=gr.Audio(sources=["upload"], type="filepath", label="上传WAV文件"),
    outputs=gr.Audio(streaming=True, autoplay=True, label="实时播放"),
    title="WAV文件实时流式播放器",
    description="上传WAV文件并实时流式播放。支持大文件播放，无需等待完整加载。"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")