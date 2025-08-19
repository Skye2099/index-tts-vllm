import requests
import numpy as np
import librosa
import sounddevice as sd
from queue import Queue

url = "http://119.45.179.67:7860/tts_live_stream"

text_content = '''水瓶座像一颗悬在天际的孤星，总以疏离又热忱的姿态打量世界。他们的独立是刻在骨子里的，从不愿被世俗的框架束缚 —— 别人追逐潮流时，他们偏要在旧物里翻找新意；众人在人群中寻求认同，他们却能在独处时听见更清晰的声音。这种特立独行并非刻意标榜，而是源于对 “独特” 的本能向往。
理性是水瓶座的另一枚标签。面对纷争，他们总能跳出情绪的漩涡，用逻辑梳理脉络，仿佛自带 “理性滤镜”。但这份冷静之下，藏着滚烫的人道主义精神：他们会为素不相识者的困境揪心，为社会不公发声，只是表达善意的方式总带着点 “水瓶座式” 的特别 —— 可能是匿名捐赠，也可能是用代码搭建公益平台。
他们的思维像万花筒，能从落叶联想到宇宙膨胀，在日常琐事里提炼哲学命题。这种跳跃性常常让身边人跟不上节奏，却也让他们成为创意领域的 “破局者”。只是，过度沉浸在自己的精神世界时，他们偶尔会显得有些 “不近人情”，就像星群虽璀璨，却始终隔着光年的距离。
这就是水瓶座，一半是清醒的观察者，一半是热忱的理想主义者，在世俗与理想的间隙里，活成了独一无二的星轨。'''

data = {
    "text": text_content,
    "audio_paths": [  # 支持多参考音频
        "/data/wts/index-tts-vllm/tests/sample_prompt.wav"

    ]
}
# 参数
SAMPLE_RATE_TTS = 24000  # 服务端输出采样率
SAMPLE_RATE_PLAY = 48000  # 播放设备支持的
CHUNK_SIZE = 1024  # 每次读取的样本数（按 24k 计算）
audio_queue = Queue(maxsize=10)

def audio_callback(outdata, frames, time, status):
    """声音设备回调函数"""
    try:
        if not audio_queue.empty():
            # 从队列获取数据并确保形状正确 (frames, channels)
            data = audio_queue.get_nowait()
            if len(data) < frames:
                # 如果数据不足，补零
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
        else:
            # 队列为空时输出静音
            outdata.fill(0)
    except Exception as e:
        print(f"Audio callback error: {e}")
        outdata.fill(0)



# 打印设备信息
print("=== 可用音频设备 ===")
print(sd.query_devices())
print("===================")
# 打开流
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE_PLAY,
    blocksize=CHUNK_SIZE,
    channels=1,
    dtype='float32',
    callback=audio_callback,
    device=11
)

# 请求流
with stream:
    response = requests.post(url, json=data, stream=True)
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE * 4):  # 4096 字节
        if not chunk:
            continue
        try:
            # ✅ 修复 buffer size 错误
            valid_len = len(chunk) // 4 * 4
            if valid_len == 0:
                continue
            chunk = chunk[:valid_len]
            pcm_24k = np.frombuffer(chunk, dtype=np.float32)

            # ✅ 重采样到 48000 Hz
            pcm_48k = librosa.resample(pcm_24k, orig_sr=24000, target_sr=48000)

            # 确保数据是二维的 (frames, channels)
            if pcm_48k.ndim == 1:
                pcm_48k = pcm_48k.reshape(-1, 1)

            # 将数据分块放入队列
            for i in range(0, len(pcm_48k), CHUNK_SIZE):
                block = pcm_48k[i:i + CHUNK_SIZE]
                audio_queue.put(block)

        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            continue  # 跳过错误帧

    # 等待播放完成
    while not audio_queue.empty():
        sd.sleep(100)

print("--complete--")