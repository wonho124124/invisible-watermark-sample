import cv2
import numpy as np
import os
import pywt
import time
import argparse
from tqdm import tqdm

class VideoFrameExtractor:
    """비디오에서 프레임을 추출하는 클래스"""
    
    def __init__(self, video_path, output_dir=None):
        """
        Args:
            video_path (str): 처리할 비디오 파일 경로
            output_dir (str, optional): 프레임을 저장할 디렉토리 경로
        """
        self.video_path = video_path
        
        if output_dir is None:
            # 비디오 파일 이름을 기반으로 output 디렉토리 생성
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.output_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        else:
            self.output_dir = output_dir
            
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 비디오 속성 초기화
        self.cap = None
        self.fps = None
        self.frame_count = None
        self.width = None
        self.height = None
        self.duration = None
        
    def get_video_properties(self):
        """비디오 속성 정보 얻기"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        video_info = {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.duration
        }
        
        return video_info
        
    def extract_frames(self, every_n_frame=1):
        """
        비디오에서 프레임 추출하여 저장
        
        Args:
            every_n_frame (int): n번째 프레임마다 추출 (기본값: 1, 모든 프레임 추출)
            
        Returns:
            list: 추출된 프레임 파일 경로 리스트
        """
        if self.cap is None:
            self.get_video_properties()
        
        frame_paths = []
        frame_idx = 0
        
        # 진행 상황을 표시하기 위해 tqdm 사용
        print(f"비디오 프레임 추출 중: {self.video_path}")
        pbar = tqdm(total=self.frame_count)
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            if frame_idx % every_n_frame == 0:
                frame_path = os.path.join(self.output_dir, f"frame_{frame_idx:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        self.cap.release()
        
        print(f"총 {len(frame_paths)}개 프레임 추출 완료 (저장 경로: {self.output_dir})")
        return frame_paths
    
    def __del__(self):
        """소멸자: 열려있는 비디오 캡처 객체를 닫음"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class WatermarkEmbedder:
    """프레임에 워터마크를 삽입하는 클래스"""
    
    def __init__(self, output_dir=None):
        """
        Args:
            output_dir (str, optional): 워터마크가 삽입된 프레임을 저장할 디렉토리
        """
        if output_dir is None:
            # 현재 시간을 기반으로 출력 디렉토리 생성
            timestamp = int(time.time())
            self.output_dir = f"watermarked_frames_{timestamp}"
        else:
            self.output_dir = output_dir
            
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def _embed_dct_watermark(self, frame, watermark_text, alpha=0.1):
        """
        DCT(Discrete Cosine Transform) 기반 워터마크 삽입
        
        Args:
            frame (numpy.ndarray): 워터마크를 삽입할 프레임
            watermark_text (str): 워터마크로 사용할 텍스트
            alpha (float): 워터마크 강도(0.05-0.2 권장)
            
        Returns:
            numpy.ndarray: 워터마크가 삽입된 프레임
        """
        # 컬러 이미지의 경우 YUV 색상 공간으로 변환 (Y 채널에만 워터마크 삽입)
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
        else:
            y_channel = frame.astype(np.float32)
        
        # 워터마크 텍스트를 바이너리 시퀀스로 변환
        watermark_binary = ''.join(format(ord(char), '08b') for char in watermark_text)
        watermark_bits = [int(bit) for bit in watermark_binary]
        
        # 워터마크 길이가 너무 긴 경우 잘라내기
        max_bits = min(64, len(watermark_bits))
        watermark_bits = watermark_bits[:max_bits]
        
        # 8x8 블록으로 분할하여 DCT 적용
        height, width = y_channel.shape
        watermarked_y = y_channel.copy()
        
        # 워터마크를 중간 주파수 계수에 삽입할 위치 (지그재그 스캔 순서)
        zigzag_positions = [(1, 2), (2, 1), (0, 3), (3, 0), (2, 2)]
        
        bit_index = 0
        block_size = 8
        
        # 이미지 중앙 부분에 워터마크 삽입 (탐지 가능성 높이기)
        center_h, center_w = height // 2, width // 2
        start_h = max(0, center_h - 128)
        start_w = max(0, center_w - 128)
        
        for i in range(start_h, min(start_h + 256, height - block_size), block_size):
            for j in range(start_w, min(start_w + 256, width - block_size), block_size):
                if bit_index >= len(watermark_bits):
                    bit_index = 0  # 워터마크 반복
                
                # 8x8 블록 추출
                block = watermarked_y[i:i+block_size, j:j+block_size]
                
                # DCT 변환
                dct_block = cv2.dct(block)
                
                # 워터마크 삽입
                bit = watermark_bits[bit_index]
                pos = zigzag_positions[bit_index % len(zigzag_positions)]
                
                if bit == 1:
                    dct_block[pos] += alpha * abs(dct_block[pos]) + 0.1
                else:
                    dct_block[pos] -= alpha * abs(dct_block[pos]) + 0.1
                
                # 역 DCT 변환
                watermarked_block = cv2.idct(dct_block)
                watermarked_y[i:i+block_size, j:j+block_size] = watermarked_block
                
                bit_index += 1
        
        # 워터마크가 삽입된 Y 채널을 원래 이미지에 다시 합치기
        if len(frame.shape) == 3:
            yuv_frame[:, :, 0] = np.clip(watermarked_y, 0, 255).astype(np.uint8)
            watermarked_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
        else:
            watermarked_frame = np.clip(watermarked_y, 0, 255).astype(np.uint8)
            
        return watermarked_frame
        
    def _embed_dwt_watermark(self, frame, watermark_text, alpha=0.1):
        """
        DWT(Discrete Wavelet Transform) 기반 워터마크 삽입
        
        Args:
            frame (numpy.ndarray): 워터마크를 삽입할 프레임
            watermark_text (str): 워터마크로 사용할 텍스트
            alpha (float): 워터마크 강도(0.05-0.2 권장)
            
        Returns:
            numpy.ndarray: 워터마크가 삽입된 프레임
        """
        # 워터마크 텍스트를 바이너리 시퀀스로 변환
        watermark_binary = ''.join(format(ord(char), '08b') for char in watermark_text)
        watermark_bits = np.array([int(bit) for bit in watermark_binary])
        
        # BGR 이미지를 YUV로 변환하여 Y 채널에 워터마크 추가
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
        else:
            y_channel = frame.astype(np.float32)
            
        # 2D 이산 웨이블릿 변환 (DWT) 수행
        coeffs2 = pywt.dwt2(y_channel, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # 워터마크 비트 수가 너무 많으면 잘라내기
        max_bits = min(len(watermark_bits), HL.size // 100)
        watermark_bits = watermark_bits[:max_bits]
        
        # 워터마크를 HL 서브밴드에 삽입 (주로 중간 주파수 성분)
        height, width = HL.shape
        bit_index = 0
        
        # 이미지 중앙 부분에 워터마크 삽입
        center_h, center_w = height // 2, width // 2
        region_size = min(height, width) // 4
        
        for i in range(center_h - region_size, center_h + region_size, 4):
            for j in range(center_w - region_size, center_w + region_size, 4):
                if i < 0 or j < 0 or i >= height or j >= width:
                    continue
                    
                if bit_index >= len(watermark_bits):
                    break
                    
                bit = watermark_bits[bit_index]
                
                # 워터마크 비트에 따라 계수 수정
                if bit == 1:
                    HL[i, j] += alpha * abs(HL[i, j]) + 0.1
                else:
                    HL[i, j] -= alpha * abs(HL[i, j]) + 0.1
                    
                bit_index += 1
        
        # 역 DWT 수행하여 워터마크가 삽입된 이미지 얻기
        watermarked_y = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
        
        # 이미지 크기가 원본과 다를 수 있으므로 조정
        watermarked_y = watermarked_y[:y_channel.shape[0], :y_channel.shape[1]]
        
        # 워터마크가 삽입된 Y 채널을 원래 이미지에 다시 합치기
        if len(frame.shape) == 3:
            yuv_frame[:, :, 0] = np.clip(watermarked_y, 0, 255).astype(np.uint8)
            watermarked_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
        else:
            watermarked_frame = np.clip(watermarked_y, 0, 255).astype(np.uint8)
            
        return watermarked_frame

    def embed_watermark(self, frame_paths, watermark_text, method='dwt', every_n_frame=1, alpha=0.1):
        """
        프레임에 워터마크 삽입
        
        Args:
            frame_paths (list): 프레임 파일 경로 리스트
            watermark_text (str): 워터마크로 사용할 텍스트
            method (str): 워터마크 방식 ('dwt' 또는 'dct')
            every_n_frame (int): n번째 프레임마다 워터마크 삽입 (기본값: 1, 모든 프레임에 삽입)
            alpha (float): 워터마크 강도
            
        Returns:
            list: 워터마크가 삽입된 프레임 파일 경로 리스트
        """
        watermarked_frame_paths = []
        
        # 워터마크 삽입 방법 선택
        if method.lower() == 'dwt':
            watermark_func = self._embed_dwt_watermark
        elif method.lower() == 'dct':
            watermark_func = self._embed_dct_watermark
        else:
            raise ValueError("워터마크 방법은 'dwt' 또는 'dct'여야 합니다.")
        
        print(f"{method.upper()} 방식으로 워터마크 삽입 중...")
        # 진행 상황 표시
        for i, frame_path in enumerate(tqdm(frame_paths)):
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"경고: 프레임을 읽을 수 없습니다: {frame_path}")
                continue
                
            # n번째 프레임마다 워터마크 삽입
            if i % every_n_frame == 0:
                watermarked_frame = watermark_func(frame, watermark_text, alpha)
            else:
                watermarked_frame = frame
                
            # 워터마크가 삽입된 프레임 저장
            frame_name = os.path.basename(frame_path)
            output_path = os.path.join(self.output_dir, frame_name)
            cv2.imwrite(output_path, watermarked_frame)
            watermarked_frame_paths.append(output_path)
        
        print(f"총 {len(watermarked_frame_paths)}개 프레임에 워터마크 삽입 완료 (저장 경로: {self.output_dir})")
        return watermarked_frame_paths


class VideoFrameCombiner:
    """프레임을 비디오로 인코딩하는 클래스"""
    
    def __init__(self, output_path=None):
        """
        Args:
            output_path (str, optional): 출력 비디오 파일 경로
        """
        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"watermarked_video_{timestamp}.mp4"
        else:
            self.output_path = output_path
    
    def combine_frames(self, frame_paths, fps=30, codec='mp4v'):
        """
        프레임을 비디오로 인코딩
        
        Args:
            frame_paths (list): 인코딩할 프레임 파일 경로 리스트
            fps (float): 출력 비디오의 프레임 레이트
            codec (str): 비디오 코덱 (기본값: 'mp4v', 다른 옵션: 'avc1', 'h264')
            
        Returns:
            str: 인코딩된 비디오 파일 경로
        """
        if not frame_paths:
            raise ValueError("인코딩할 프레임이 없습니다.")
            
        # 첫 번째 프레임을 읽어서 비디오 해상도 결정
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"프레임을 읽을 수 없습니다: {frame_paths[0]}")
            
        height, width = first_frame.shape[:2]
        
        # 코덱 설정
        if codec == 'mp4v':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
        elif codec == 'avc1' or codec == 'h264':
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        else:
            raise ValueError(f"지원되지 않는 코덱: {codec}")
            
        # VideoWriter 객체 생성
        video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise ValueError(f"비디오 작성기를 초기화할 수 없습니다: {self.output_path}")
        
        print(f"비디오 인코딩 중: {self.output_path}")
        # 각 프레임을 비디오에 추가
        for frame_path in tqdm(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        print(f"비디오 인코딩 완료: {self.output_path}")
        return self.output_path


class WatermarkDetector:
    """비디오에서 워터마크를 감지하는 클래스"""
    
    def __init__(self):
        """워터마크 감지기 초기화"""
        
    def _detect_dct_watermark(self, frame, watermark_text, threshold=0.6):
        """
        DCT 기반 워터마크 검출
        
        Args:
            frame (numpy.ndarray): 워터마크를 검출할 프레임
            watermark_text (str): 원본 워터마크 텍스트
            threshold (float): 감지 임계값 (0-1 사이)
            
        Returns:
            bool: 워터마크 감지 여부
            float: 신뢰도 점수 (0-1 사이)
        """
        # 컬러 이미지의 경우 YUV 색상 공간으로 변환
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
        else:
            y_channel = frame.astype(np.float32)
            
        # 워터마크 텍스트를 바이너리 시퀀스로 변환
        watermark_binary = ''.join(format(ord(char), '08b') for char in watermark_text)
        original_bits = [int(bit) for bit in watermark_binary]
        
        # 워터마크 길이가 너무 긴 경우 잘라내기
        max_bits = min(64, len(original_bits))
        original_bits = original_bits[:max_bits]
        
        # 8x8 블록으로 분할하여 DCT 적용
        height, width = y_channel.shape
        
        # 워터마크를 중간 주파수 계수에 삽입했던 위치
        zigzag_positions = [(1, 2), (2, 1), (0, 3), (3, 0), (2, 2)]
        
        bit_index = 0
        block_size = 8
        extracted_bits = []
        
        # 이미지 중앙 부분에서 워터마크 추출
        center_h, center_w = height // 2, width // 2
        start_h = max(0, center_h - 128)
        start_w = max(0, center_w - 128)
        
        for i in range(start_h, min(start_h + 256, height - block_size), block_size):
            for j in range(start_w, min(start_w + 256, width - block_size), block_size):
                if bit_index >= len(original_bits):
                    break
                
                # 8x8 블록 추출
                block = y_channel[i:i+block_size, j:j+block_size]
                
                # DCT 변환
                dct_block = cv2.dct(block)
                
                # 워터마크 추출
                pos = zigzag_positions[bit_index % len(zigzag_positions)]
                extracted_bit = 1 if dct_block[pos] > 0 else 0
                extracted_bits.append(extracted_bit)
                
                bit_index += 1
        
        # 원래 워터마크 비트와 추출된 비트 비교
        correct_bits = sum(1 for a, b in zip(original_bits, extracted_bits) if a == b)
        similarity = correct_bits / len(original_bits) if len(original_bits) > 0 else 0
        
        # 임계값보다 높으면 워터마크 감지됨
        is_detected = similarity >= threshold
        
        return is_detected, similarity
    
    def _detect_dwt_watermark(self, frame, watermark_text, threshold=0.6):
        """
        DWT 기반 워터마크 검출
        
        Args:
            frame (numpy.ndarray): 워터마크를 검출할 프레임
            watermark_text (str): 원본 워터마크 텍스트
            threshold (float): 감지 임계값 (0-1 사이)
            
        Returns:
            bool: 워터마크 감지 여부
            float: 신뢰도 점수 (0-1 사이)
        """
        # 워터마크 텍스트를 바이너리 시퀀스로 변환
        watermark_binary = ''.join(format(ord(char), '08b') for char in watermark_text)
        original_bits = [int(bit) for bit in watermark_binary]
        
        # BGR 이미지를 YUV로 변환
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
        else:
            y_channel = frame.astype(np.float32)
            
        # 2D 이산 웨이블릿 변환 (DWT) 수행
        coeffs2 = pywt.dwt2(y_channel, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # 워터마크 비트 수가 너무 많으면 잘라내기
        max_bits = min(len(original_bits), HL.size // 100)
        original_bits = original_bits[:max_bits]
        
        # HL 서브밴드에서 워터마크 추출
        height, width = HL.shape
        extracted_bits = []
        bit_index = 0
        
        # 이미지 중앙 부분에서 워터마크 추출
        center_h, center_w = height // 2, width // 2
        region_size = min(height, width) // 4
        
        for i in range(center_h - region_size, center_h + region_size, 4):
            for j in range(center_w - region_size, center_w + region_size, 4):
                if i < 0 or j < 0 or i >= height or j >= width:
                    continue
                    
                if bit_index >= len(original_bits):
                    break
                    
                # 계수 값의 부호에 따라 비트 추출
                extracted_bit = 1 if HL[i, j] > 0 else 0
                extracted_bits.append(extracted_bit)
                bit_index += 1
        
        # 원래 워터마크 비트와 추출된 비트 비교
        correct_bits = sum(1 for a, b in zip(original_bits, extracted_bits) if a == b)
        similarity = correct_bits / len(original_bits) if original_bits else 0
        
        # 임계값보다 높으면 워터마크 감지됨
        is_detected = similarity >= threshold
        
        return is_detected, similarity
    
    def detect_watermark(self, video_path, watermark_text, method='dwt', sampling_rate=30, threshold=0.6):
        """
        비디오에서 워터마크 감지
        
        Args:
            video_path (str): 워터마크를 감지할 비디오 파일 경로
            watermark_text (str): 원본 워터마크 텍스트
            method (str): 워터마크 방식 ('dwt' 또는 'dct')
            sampling_rate (int): 몇 프레임마다 샘플링할지 (기본값: 30)
            threshold (float): 감지 임계값 (0-1 사이)
            
        Returns:
            dict: 워터마크 감지 결과 정보
        """
        # 워터마크 감지 방법 선택
        if method.lower() == 'dwt':
            detect_func = self._detect_dwt_watermark
        elif method.lower() == 'dct':
            detect_func = self._detect_dct_watermark
        else:
            raise ValueError("워터마크 방법은 'dwt' 또는 'dct'여야 합니다.")
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
            
        # 비디오 속성 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"워터마크 감지 중: {video_path}")
        print(f"비디오 정보: {frame_count}프레임, {fps}fps, {duration:.2f}초")
        
        # 워터마크 감지 결과
        detection_results = []
        frame_idx = 0
        
        # 진행 상황 표시
        pbar = tqdm(total=frame_count)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # 일정 프레임마다 샘플링
            if frame_idx % sampling_rate == 0:
                is_detected, similarity = detect_func(frame, watermark_text, threshold)
                
                detection_results.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'is_detected': is_detected,
                    'similarity': similarity
                })
                
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # 감지된 프레임 수와 비율 계산
        detected_frames = sum(1 for result in detection_results if result['is_detected'])
        detection_rate = detected_frames / len(detection_results) if detection_results else 0
        
        # 감지 결과 요약
        detection_summary = {
            'video_path': video_path,
            'watermark_method': method,
            'detected_frames': detected_frames,
            'total_sampled_frames': len(detection_results),
            'detection_rate': detection_rate,
            'avg_similarity': sum(r['similarity'] for r in detection_results) / len(detection_results) if detection_results else 0,
            'detection_results': detection_results
        }
        
        print(f"워터마크 감지 완료: {detected_frames}/{len(detection_results)} 프레임 감지됨 (감지율: {detection_rate:.2%})")
        return detection_summary