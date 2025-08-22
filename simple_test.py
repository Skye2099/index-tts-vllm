import argparse
import threading
import time
import requests
from collections import defaultdict
import random
import json

class TTSStressTester:
    def __init__(self, urls, data, concurrency, requests_per_thread, test_mode='regular'):
        self.urls = urls
        self.data = data
        self.concurrency = concurrency
        self.requests_per_thread = requests_per_thread
        self.test_mode = test_mode  # 'regular' or 'stream'
        self.stats = {
            'total': 0,
            'success': 0,
            'fail': 0,
            'durations': [],
            'status_codes': defaultdict(int),
            'errors': defaultdict(int),
            'ttft_times': [],  # 首token时间
            'rtfs': []  # RTF值
        }
        self.lock = threading.Lock()
        self.current_url_index = 0
        self.url_lock = threading.Lock()  # 用于轮询URL的锁

    def _get_next_url(self):
        with self.url_lock:
            url = self.urls[self.current_url_index]
            self.current_url_index = (self.current_url_index + 1) % len(self.urls)
        return url

    def _send_request(self):
        start_time = time.time()
        try:
            # 生成随机数字符串，确保不触发 vllm 的 cache
            self.data["text"] = ",".join(["".join([str(random.randint(0, 9)) for _ in range(5)]) for _ in range(5)])
            target_url = self._get_next_url()  # 获取轮询后的URL
            
            if self.test_mode == 'stream':
                # 流式测试
                response = requests.post(target_url, json=self.data, timeout=10, stream=True)
                first_chunk_time = None
                audio_data_length = 0
                start_time_local = time.time()
                
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time_local
                        audio_data_length += len(chunk)
                
                elapsed = time.time() - start_time_local
                
                # 计算RTF (简化计算，实际应根据音频时长)
                # 这里假设音频采样率为24000Hz，16位深度
                audio_duration = audio_data_length / (2 * 24000)  # 2字节每样本
                rtf = elapsed / audio_duration if audio_duration > 0 else 0
                
                with self.lock:
                    self.stats['ttft_times'].append(first_chunk_time if first_chunk_time is not None else 0)
                    self.stats['rtfs'].append(rtf)
            else:
                # 常规测试
                response = requests.post(target_url, json=self.data, timeout=10)
                elapsed = time.time() - start_time
            
            with self.lock:
                self.stats['durations'].append(elapsed)
                self.stats['status_codes'][response.status_code] += 1
                self.stats['total'] += 1
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'audio' in content_type:
                        self.stats['success'] += 1
                    else:
                        self.stats['fail'] += 1
                        self.stats['errors']['invalid_content_type'] += 1
                else:
                    self.stats['fail'] += 1
                    
        except Exception as e:
            with self.lock:
                self.stats['fail'] += 1
                self.stats['errors'][str(type(e).__name__)] += 1
                self.stats['durations'].append(time.time() - start_time)

    def _worker(self):
        for _ in range(self.requests_per_thread):
            self._send_request()

    def run(self):
        threads = []
        start_time = time.time()
        
        for _ in range(self.concurrency):
            thread = threading.Thread(target=self._worker)
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time
        self._generate_report(total_time)

    def _generate_report(self, total_time):
        durations = self.stats['durations']
        total_requests = self.stats['total']
        
        print(f"\n{' 测试报告 ':=^40}")
        print(f"总请求时间: {total_time:.2f}秒")
        print(f"总请求量: {total_requests}")
        print(f"成功请求: {self.stats['success']}")
        print(f"失败请求: {self.stats['fail']}")
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            print(f"\n响应时间统计:")
            print(f"平均: {avg_duration:.3f}秒")
            print(f"最大: {max_duration:.3f}秒")
            print(f"最小: {min_duration:.3f}秒")
            
            sorted_durations = sorted(durations)
            for p in [50, 90, 95, 99]:
                index = int(p / 100 * len(sorted_durations))
                print(f"P{p}: {sorted_durations[index]:.3f}秒")

        # 流式测试特有指标
        if self.test_mode == 'stream':
            if self.stats['ttft_times']:
                ttft_times = self.stats['ttft_times']
                avg_ttft = sum(ttft_times) / len(ttft_times)
                max_ttft = max(ttft_times)
                min_ttft = min(ttft_times)
                print(f"\n首token时间 (TTFT) 统计:")
                print(f"平均: {avg_ttft:.3f}秒")
                print(f"最大: {max_ttft:.3f}秒")
                print(f"最小: {min_ttft:.3f}秒")
                
                sorted_ttft = sorted(ttft_times)
                for p in [50, 90, 95, 99]:
                    index = int(p / 100 * len(sorted_ttft))
                    print(f"P{p}: {sorted_ttft[index]:.3f}秒")
            
            if self.stats['rtfs']:
                rtfs = self.stats['rtfs']
                avg_rtf = sum(rtfs) / len(rtfs)
                max_rtf = max(rtfs)
                min_rtf = min(rtfs)
                print(f"\nRTF (Real-Time Factor) 统计:")
                print(f"平均: {avg_rtf:.4f}")
                print(f"最大: {max_rtf:.4f}")
                print(f"最小: {min_rtf:.4f}")
                
                sorted_rtfs = sorted(rtfs)
                for p in [50, 90, 95, 99]:
                    index = int(p / 100 * len(sorted_rtfs))
                    print(f"P{p}: {sorted_rtfs[index]:.4f}")

        print("\n状态码分布:")
        for code, count in self.stats['status_codes'].items():
            print(f"HTTP {code}: {count}次")

        if self.stats['errors']:
            print("\n错误统计:")
            for error, count in self.stats['errors'].items():
                print(f"{error}: {count}次")

        print(f"\n吞吐量: {total_requests / total_time:.2f} 请求/秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTS服务压力测试脚本')
    parser.add_argument('--urls', nargs='+', 
                        default=['http://localhost:9880/tts'],  # , 'http://localhost:11997/tts'
                        help='TTS服务地址列表（多个用空格分隔）')
    parser.add_argument('--text', type=str, default='测试文本', help='需要合成的文本内容')
    parser.add_argument('--character', type=str, default='lancy', help='合成角色名称')
    parser.add_argument('--concurrency', type=int, default=16, help='并发线程数')
    parser.add_argument('--requests', type=int, default=5, help='每个线程的请求数')
    parser.add_argument('--mode', type=str, default='stream', choices=['regular', 'stream'], help='测试模式: regular(常规) 或 stream(流式)')
    
    args = parser.parse_args()
    
    test_data = {
        "text": args.text,
        "character": args.character
    }
    
    # 如果是流式测试，需要修改URL
    urls = args.urls
    if args.mode == 'stream':
        # 将/tts替换为/tts_live_stream
        urls = [url.replace('/tts', '/tts_live_stream') for url in args.urls]
        # 确保数据格式正确
        test_data = {
            "text": args.text,
            "audio_paths": ["assets/jay_promptvn.wav"]  # 默认参考音频
        }
    
    tester = TTSStressTester(
        urls=urls,
        data=test_data,
        concurrency=args.concurrency,
        requests_per_thread=args.requests,
        test_mode=args.mode
    )
    
    print(f"开始压力测试，配置参数：")
    print(f"目标服务: {', '.join(urls)}")
    print(f"测试模式: {args.mode}")
    print(f"并发线程: {args.concurrency}")
    print(f"单线程请求数: {args.requests}")
    print(f"总预计请求量: {args.concurrency * args.requests}")
    print(f"{' 测试启动 ':=^40}")
    
    try:
        tester.run()
    except KeyboardInterrupt:
        print("\n测试被用户中断")