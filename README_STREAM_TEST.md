# TTS流式输出测试指南

本文档介绍了如何使用`simple_test.py`脚本进行TTS服务的流式输出测试，包括首token时间(TTFT)和RTF等指标。

## 使用方法

### 1. 常规测试模式

```bash
python simple_test.py --urls http://localhost:9880/tts --concurrency 16 --requests 5
```

### 2. 流式测试模式

```bash
python simple_test.py --urls http://localhost:9880/tts --mode stream --concurrency 16 --requests 5
```

## 参数说明

- `--urls`: TTS服务地址列表（多个用空格分隔）
- `--text`: 需要合成的文本内容（默认：测试文本）
- `--character`: 合成角色名称（默认：lancy）
- `--concurrency`: 并发线程数（默认：16）
- `--requests`: 每个线程的请求数（默认：5）
- `--mode`: 测试模式，可选`regular`（常规）或`stream`（流式）

## 流式测试指标

流式测试模式下，除了常规的响应时间、吞吐量等指标外，还会统计以下特有指标：

1. **首token时间 (TTFT - Time To First Token)**: 从发送请求到接收到第一个音频数据块的时间
2. **RTF (Real-Time Factor)**: 音频生成时间与音频时长的比值

## 注意事项

1. 流式测试时，脚本会自动将URL中的`/tts`替换为`/tts_live_stream`
2. 流式测试需要提供参考音频路径，默认使用`assets/seed_tts/lancy_001.wav`
3. RTF计算基于简化模型，实际值可能因音频时长计算方式不同而有差异