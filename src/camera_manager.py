#!/usr/bin/env python3
"""
Camera Manager Module
카메라 초기화, 설정, 녹화 기능을 담당
"""

import cv2
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

class CameraManager:
    """카메라 관리 클래스"""
    
    def __init__(self, capture_dir: Path, width: int = 1280, height: int = 720, fps: int = 10):
        self.capture_dir = capture_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        # 캡처 디렉토리 생성
        self.capture_dir.mkdir(exist_ok=True)
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """카메라 초기화"""
        try:
            # 카메라 0 시도
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"카메라 {camera_id}을 열 수 없습니다. 카메라 1을 시도합니다...")
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    print("카메라를 열 수 없습니다.")
                    return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 줄여서 지연 감소
            
            print(f"✅ 카메라 초기화 완료: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 초기화 실패: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """프레임 읽기"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def record_video(self, duration: int = 5, recording_fps: int = 5) -> Optional[Path]:
        """비디오 녹화"""
        if self.cap is None:
            print("❌ 카메라가 초기화되지 않았습니다.")
            return None
        
        try:
            # 비디오 파일 경로 생성
            start_dt = datetime.now()
            vid_path = self.capture_dir / f"video_{start_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(vid_path), fourcc, recording_fps, (self.width, self.height))
            
            if not vw.isOpened():
                raise RuntimeError("VideoWriter 초기화 실패")
            
            print(f"🎬 [RECORDING] {duration}초 비디오 녹화 시작...")
            
            # 녹화 시작
            t0 = time.time()
            frame_count = 0
            
            while time.time() - t0 < duration:
                ret, frame = self.read_frame()
                if not ret:
                    break
                vw.write(frame)
                frame_count += 1
            
            vw.release()
            print(f"✅ [RECORDING] 저장 완료: {vid_path} ({frame_count} 프레임)")
            return vid_path
            
        except Exception as e:
            print(f"❌ 녹화 실패: {e}")
            return None
    
    def add_status_overlay(self, frame: cv2.Mat, recording_active: bool, queue_size: int) -> cv2.Mat:
        """프레임에 상태 오버레이 추가"""
        # 녹화 상태 표시
        status_text = "RECORDING" if recording_active else "MONITORING"
        status_color = (0, 0, 255) if recording_active else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 신호 큐 상태 표시
        queue_text = f"Signals in queue: {queue_size}"
        cv2.putText(frame, queue_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def release(self):
        """카메라 리소스 해제"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        print("📷 카메라 리소스 해제 완료")
    
    def __del__(self):
        """소멸자"""
        self.release() 