#!/usr/bin/env python3
"""
Demo Script - API-based Recording System
모듈화된 구조를 사용한 간소화된 메인 스크립트
"""

import sys
import time
import warnings
from pathlib import Path
from threading import Thread
from datetime import datetime
from typing import List, Dict

# 모듈 import를 위한 경로 추가
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from camera_manager import CameraManager
from dam_analyzer import DAMAnalyzer
from log_manager import LogManager
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DURATION_SEC = 5
FPS = 5
API_HOST = 'localhost'
API_PORT = 5000

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAM_SCRIPT = PROJECT_ROOT / "src" / "dam_video_with_sam2.py"
CAPTURE_DIR = PROJECT_ROOT / "captures"
LOG_FILE = PROJECT_ROOT / "action_log.txt"

# -----------------------------------------------------------------------------
# Global Variables & Flask App
# -----------------------------------------------------------------------------
recording_active = False
signal_queue = []
app = Flask(__name__)

# 모듈 인스턴스
camera_manager = None
dam_analyzer = None
log_manager = None

# -----------------------------------------------------------------------------
# External Signal System
# -----------------------------------------------------------------------------
class ExternalSignal:
    """외부 신호 데이터 구조"""
    def __init__(self, signal_type: str, bbox_normalized: List[float], metadata: Dict = None):
        self.signal_type = signal_type
        self.bbox_normalized = bbox_normalized
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

# -----------------------------------------------------------------------------
# Flask API Routes
# -----------------------------------------------------------------------------
@app.route('/trigger_recording', methods=['POST'])
def trigger_recording():
    """녹화 트리거 API"""
    try:
        data = request.get_json()
        bbox_normalized = data.get('bbox_normalized', [0.25, 0.25, 0.75, 0.75])
        signal_type = data.get('signal_type', 'api_trigger')
        metadata = data.get('metadata', {})
        
        signal = ExternalSignal(signal_type, bbox_normalized, metadata)
        signal_queue.append(signal)
        
        print(f"[API] 녹화 트리거: {signal_type} with bbox {bbox_normalized}")
        
        return jsonify({
            'status': 'success',
            'message': 'Recording triggered successfully',
            'signal_type': signal_type,
            'bbox_normalized': bbox_normalized
        }), 200
        
    except Exception as e:
        print(f"[API ERROR] {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/status', methods=['GET'])
def get_status():
    """시스템 상태 조회 API"""
    return jsonify({
        'status': 'success',
        'recording_active': recording_active,
        'signals_in_queue': len(signal_queue),
        'timestamp': datetime.now().isoformat()
    }), 200

def run_flask_api():
    """Flask API 서버 실행"""
    print(f"[API] API 서버 시작: {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False)

# -----------------------------------------------------------------------------
# Signal Processing
# -----------------------------------------------------------------------------
def process_external_signals():
    """외부 신호 처리"""
    global recording_active
    
    while True:
        if signal_queue and not recording_active:
            signal = signal_queue.pop(0)
            
            print(f"[TRIGGER] 신호 처리: {signal.signal_type}")
            print(f"[TRIGGER] BBox: {signal.bbox_normalized}")
            
            def record_and_analyze():
                global recording_active
                recording_active = True
                try:
                    # 녹화
                    video_path = camera_manager.record_video(DURATION_SEC, FPS)
                    if video_path:
                        # DAM 분석
                        description = dam_analyzer.analyze_video(video_path, signal.bbox_normalized, use_sam2=False)
                        if description:
                            # 로그 저장
                            log_manager.log_analysis_result(
                                video_path, signal.bbox_normalized, description, "bbox_based", DURATION_SEC
                            )
                            print(f"[완료] 분석 결과: {description}")
                        
                except Exception as e:
                    print(f"[오류] 녹화/분석 실패: {e}")
                finally:
                    recording_active = False
            
            Thread(target=record_and_analyze, daemon=True).start()
        
        time.sleep(0.1)

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    global camera_manager, dam_analyzer, log_manager, recording_active
    
    # 환경 체크
    try:
        import sam2, torch
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available – inference will run on CPU (slow)")
    except ImportError:
        warnings.warn("sam2 or torch not importable – please check installation")
    
    # 모듈 초기화
    print(" 모듈 초기화 중...")
    try:
        camera_manager = CameraManager(CAPTURE_DIR, width=1280, height=720, fps=10)
        dam_analyzer = DAMAnalyzer(DAM_SCRIPT, temperature=0.1, top_p=0.15)
        log_manager = LogManager(LOG_FILE)
        print("모듈 초기화 완료")
    except Exception as e:
        print(f"모듈 초기화 실패: {e}")
        return
    
    # 카메라 초기화
    if not camera_manager.initialize_camera():
        print(" 카메라 초기화 실패")
        return
    
    print("=== API 기반 녹화 시스템 ===")
    print("외부 API 신호 대기 중...")
    print(f"API 서버: http://{API_HOST}:{API_PORT}")
    print("명령어:")
    print("  'q' - 종료")
    print("외부 스크립트로 API 요청을 보내세요")
    
    # 백그라운드 스레드 시작
    Thread(target=run_flask_api, daemon=True).start()
    Thread(target=process_external_signals, daemon=True).start()
    
    # API 서버 시작 대기
    time.sleep(2)
    
    # 메인 루프
    try:
        while True:
            ret, frame = camera_manager.read_frame()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 상태 오버레이 추가
            frame = camera_manager.add_status_overlay(frame, recording_active, len(signal_queue))
            
            # 화면 표시
            import cv2
            cv2.imshow("API-based Recording System", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    
    finally:
        # 리소스 정리
        camera_manager.release()
        print(" 시스템 종료")

# -----------------------------------------------------------------------------
# External API Functions (for integration)
# -----------------------------------------------------------------------------
def send_external_signal(bbox_normalized: List[float], signal_type: str = "intrusion_detected", metadata: Dict = None):
    """외부 API 함수 - 신호 전송"""
    signal = ExternalSignal(signal_type, bbox_normalized, metadata)
    signal_queue.append(signal)
    print(f"[API] 신호 전송: {signal_type} with bbox {bbox_normalized}")

def analyze_video_with_external_bbox(video_path: str, bbox_normalized: List[float], use_sam2: bool = False) -> str:
    """외부 API 함수 - 비디오 분석"""
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # DAM 분석기가 없으면 임시 생성
        if dam_analyzer is None:
            temp_analyzer = DAMAnalyzer(DAM_SCRIPT)
            description = temp_analyzer.analyze_video(video_path, bbox_normalized, use_sam2)
        else:
            description = dam_analyzer.analyze_video(video_path, bbox_normalized, use_sam2)
        
        # 로그 저장
        if log_manager and description:
            log_manager.log_analysis_result(video_path, bbox_normalized, description, 
                                          "sam2_segmentation" if use_sam2 else "bbox_based")
        
        return description
    except Exception as e:
        print(f"외부 분석 실패: {e}")
        raise

if __name__ == "__main__":
    main()
