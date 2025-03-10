import os
import argparse
import time
from video_watermark import VideoFrameExtractor, WatermarkEmbedder, VideoFrameCombiner, WatermarkDetector

class VideoWatermarker:
    """비디오 워터마킹 주 클래스"""
    
    def __init__(self, watermark_text="Copyright", method="dwt", alpha=0.1, every_n_frame=1):
        """
        Args:
            watermark_text (str): 워터마크로 사용할 텍스트
            method (str): 워터마크 방식 ('dwt' 또는 'dct')
            alpha (float): 워터마크 강도(0.05-0.2 권장)
            every_n_frame (int): n번째 프레임마다 워터마크 삽입
        """
        self.watermark_text = watermark_text
        self.method = method
        self.alpha = alpha
        self.every_n_frame = every_n_frame
        
        # 작업 디렉토리 설정
        timestamp = int(time.time())
        self.work_dir = f"watermark_job_{timestamp}"
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        
        # 하위 디렉토리 설정
        self.frames_dir = os.path.join(self.work_dir, "frames")
        self.watermarked_frames_dir = os.path.join(self.work_dir, "watermarked_frames")
        
    def embed_watermark(self, input_video, output_video=None, clean_temp=True):
        """
        비디오에 워터마크 삽입
        
        Args:
            input_video (str): 입력 비디오 파일 경로
            output_video (str, optional): 출력 비디오 파일 경로
            clean_temp (bool): 임시 파일 정리 여부
            
        Returns:
            str: 워터마크가 삽입된 비디오 파일 경로
        """
        print(f"\n{'='*50}")
        print(f"비디오 워터마킹 시작: {input_video}")
        print(f"워터마크 텍스트: {self.watermark_text}")
        print(f"워터마크 방식: {self.method.upper()}")
        print(f"워터마크 강도: {self.alpha}")
        print(f"워터마크 주기: 매 {self.every_n_frame}번째 프레임")
        print(f"{'='*50}\n")
        
        # 1. 비디오 프레임 추출
        extractor = VideoFrameExtractor(input_video, self.frames_dir)
        video_info = extractor.get_video_properties()
        print(f"비디오 정보: {video_info['frame_count']}프레임, {video_info['fps']}fps, {video_info['width']}x{video_info['height']}, {video_info['duration']:.2f}초")
        
        frame_paths = extractor.extract_frames()
        
        # 2. 워터마크 삽입
        embedder = WatermarkEmbedder(self.watermarked_frames_dir)
        watermarked_frame_paths = embedder.embed_watermark(
            frame_paths, 
            self.watermark_text, 
            method=self.method, 
            every_n_frame=self.every_n_frame,
            alpha=self.alpha
        )
        
        # 3. 프레임 재결합 및 비디오 인코딩
        if output_video is None:
            video_name = os.path.splitext(os.path.basename(input_video))[0]
            output_video = f"{video_name}_watermarked.mp4"
            
        combiner = VideoFrameCombiner(output_video)
        output_path = combiner.combine_frames(watermarked_frame_paths, fps=video_info['fps'])
        
        print(f"\n워터마크 삽입 완료: {output_path}")
        
        # 임시 파일 정리
        if clean_temp:
            print("임시 파일 정리 중...")
            import shutil
            shutil.rmtree(self.work_dir)
            print("임시 파일 정리 완료")
        
        return output_path
        
    def verify_watermark(self, video_path, sampling_rate=30, threshold=0.6):
        """
        비디오에서 워터마크 감지
        
        Args:
            video_path (str): 워터마크를 감지할 비디오 파일 경로
            sampling_rate (int): 몇 프레임마다 샘플링할지
            threshold (float): 감지 임계값 (0-1 사이)
            
        Returns:
            dict: 워터마크 감지 결과 정보
        """
        print(f"\n{'='*50}")
        print(f"워터마크 검증 시작: {video_path}")
        print(f"워터마크 텍스트: {self.watermark_text}")
        print(f"워터마크 방식: {self.method.upper()}")
        print(f"{'='*50}\n")
        
        detector = WatermarkDetector()
        detection_result = detector.detect_watermark(
            video_path,
            self.watermark_text,
            method=self.method,
            sampling_rate=sampling_rate,
            threshold=threshold
        )
        
        # 결과 출력
        print(f"\n워터마크 검증 결과:")
        print(f"감지된 프레임: {detection_result['detected_frames']}/{detection_result['total_sampled_frames']} ({detection_result['detection_rate']:.2%})")
        print(f"평균 유사도: {detection_result['avg_similarity']:.4f}")
        
        if detection_result['detection_rate'] > 0.5:
            print("결론: 워터마크가 확인되었습니다.")
        else:
            print("결론: 워터마크가 감지되지 않았거나 신뢰도가 낮습니다.")
            
        return detection_result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='비디오 인비저블 워터마킹 도구')
    
    parser.add_argument('action', choices=['embed', 'verify'], help='수행할 작업: embed(삽입) 또는 verify(검증)')
    parser.add_argument('input', help='입력 비디오 파일 경로')
    parser.add_argument('-o', '--output', help='출력 비디오 파일 경로 (embed 모드에서만 사용)')
    parser.add_argument('-t', '--text', default='Copyright', help='워터마크 텍스트 (기본값: "Copyright")')
    parser.add_argument('-m', '--method', choices=['dwt', 'dct'], default='dwt', help='워터마크 방식 (기본값: dwt)')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='워터마크 강도 (기본값: 0.1)')
    parser.add_argument('-n', '--every-n-frame', type=int, default=1, help='n번째 프레임마다 워터마크 삽입 (기본값: 1, 모든 프레임)')
    parser.add_argument('-s', '--sampling', type=int, default=30, help='검증 시 몇 프레임마다 샘플링할지 (기본값: 30)')
    parser.add_argument('-th', '--threshold', type=float, default=0.6, help='워터마크 감지 임계값 (기본값: 0.6)')
    parser.add_argument('-k', '--keep-temp', action='store_true', help='임시 파일 유지')
    
    args = parser.parse_args()
    
    watermarker = VideoWatermarker(
        watermark_text=args.text,
        method=args.method,
        alpha=args.alpha,
        every_n_frame=args.every_n_frame
    )
    
    if args.action == 'embed':
        watermarker.embed_watermark(args.input, args.output, clean_temp=not args.keep_temp)
    elif args.action == 'verify':
        watermarker.verify_watermark(args.input, args.sampling, args.threshold)


if __name__ == "__main__":
    main()


# 사용 예시:
def usage_example():
    """사용 예시 코드"""
    # 1. 비디오에 워터마크 삽입
    watermarker = VideoWatermarker(
        watermark_text="COMPANY_COPYRIGHT_2023",  # 워터마크 텍스트
        method="dwt",                            # 워터마크 방식 ('dwt' 또는 'dct')
        alpha=0.1,                               # 워터마크 강도
        every_n_frame=1                          # 모든 프레임에 워터마크 삽입
    )
    
    input_video = "sample_video.mp4"  # 워터마크를 삽입할 비디오
    output_video = "sample_video_watermarked.mp4"  # 워터마크가 삽입된 출력 비디오
    
    watermarked_video = watermarker.embed_watermark(input_video, output_video)
    
    # 2. 워터마크 검증
    verification_result = watermarker.verify_watermark(watermarked_video)
    
    # 결과 확인
    if verification_result['detection_rate'] > 0.5:
        print("워터마크가 성공적으로 삽입되었습니다.")
    else:
        print("워터마크 삽입에 문제가 있습니다. 파라미터를 조정해보세요.")


# 명령줄에서 사용 예시:
"""
# 워터마크 삽입:
python main.py embed input_video.mp4 -o watermarked_output.mp4 -t "COMPANY_COPYRIGHT" -m dwt -a 0.3 

# 워터마크 검증:
python main.py verify watermarked_output.mp4 -t "COMPANY_COPYRIGHT" -m dwt -s 15
"""