#!/usr/bin/env python3
"""
DAM Analyzer Module
DAM 비디오 분석 기능을 담당 (TensorRT 최적화 지원)
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import List, Optional

class DAMAnalyzer:
    """DAM 분석 클래스 (TensorRT 최적화 지원)"""
    
    def __init__(self, dam_script_path: Path, temperature: float = 0.1, top_p: float = 0.15, 
                 use_tensorrt: bool = True, tensorrt_cache_dir: str = "tensorrt_cache"):
        self.dam_script_path = dam_script_path
        self.temperature = temperature
        self.top_p = top_p
        self.use_tensorrt = use_tensorrt
        self.tensorrt_cache_dir = tensorrt_cache_dir
        
        # TensorRT 최적화기
        self.tensorrt_optimizer = None
        
        # DAM 스크립트 존재 확인
        if not self.dam_script_path.exists():
            raise FileNotFoundError(f"DAM script not found at: {self.dam_script_path}")
        
        # 기본 프롬프트
        self.prompt = (
            "Video: <image><image><image><image><image><image><image><image>\n"
            "Return **one concise English sentence** that describes ONLY the subject's action or state change. "
            "Do NOT mention appearance, colour, clothing, background, objects, or physical attributes."
        )
        
        # TensorRT 초기화 시도
        if self.use_tensorrt:
            self._initialize_tensorrt()
    
    def _initialize_tensorrt(self):
        """TensorRT 최적화기 초기화"""
        try:
            from dam_tensorrt_optimizer import create_optimized_dam_analyzer
            
            print("🚀 TensorRT 최적화 초기화 중...")
            self.tensorrt_optimizer = create_optimized_dam_analyzer(
                model_path="nvidia/DAM-3B-Video",
                cache_dir=self.tensorrt_cache_dir,
                force_rebuild=False  # 기존 엔진이 있으면 재사용
            )
            print("✅ TensorRT 최적화 완료 - 고속 추론 모드 활성화")
            
        except Exception as e:
            print(f"⚠️ TensorRT 초기화 실패, 기본 모드로 전환: {e}")
            self.use_tensorrt = False
            self.tensorrt_optimizer = None
    
    def _extract_description(self, raw_output: str) -> str:
        """DAM 출력에서 설명 추출"""
        desc = ""
        for line in raw_output.splitlines():
            if line.startswith("Description:"):
                desc = line.split("Description:", 1)[1].strip()
        
        if desc:
            return desc
        
        # fallback - 진행률 표시줄이나 경고가 아닌 마지막 깨끗한 줄 선택
        clean_lines = [
            l for l in raw_output.splitlines() 
            if l.strip() and not re.search(r"frame loading|propagate in video|Loading checkpoint|UserWarning", l)
        ]
        return clean_lines[-1].strip() if clean_lines else raw_output.strip()
    
    def _analyze_with_tensorrt(self, video_path: Path, bbox_normalized: List[float], use_sam2: bool = False) -> Optional[str]:
        """TensorRT를 사용한 고속 분석"""
        if not self.tensorrt_optimizer:
            return None
        
        try:
            import cv2
            from PIL import Image
            import numpy as np
            
            # 비디오에서 8개 프레임 추출
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 균등하게 8개 프레임 선택
            indices = np.linspace(0, frame_count-1, 8, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            
            cap.release()
            
            if len(frames) != 8:
                print(f"⚠️ 프레임 추출 실패: {len(frames)}/8")
                return None
            
            # 마스크 생성
            masks = []
            for frame in frames:
                width, height = frame.size
                
                # 정규화된 좌표를 절대 좌표로 변환
                x1 = int(bbox_normalized[0] * width)
                y1 = int(bbox_normalized[1] * height)
                x2 = int(bbox_normalized[2] * width)
                y2 = int(bbox_normalized[3] * height)
                
                # 마스크 생성 (bbox 영역은 255, 나머지는 0)
                mask_array = np.zeros((height, width), dtype=np.uint8)
                mask_array[y1:y2, x1:x2] = 255
                masks.append(Image.fromarray(mask_array))
            
            # TensorRT 추론
            print("🔥 TensorRT 고속 추론 실행...")
            description = self.tensorrt_optimizer.infer(frames, masks)
            
            if description:
                print(f"✅ TensorRT 분석 완료: {description}")
                return description
            else:
                print("⚠️ TensorRT 추론 실패, 기본 모드로 전환")
                return None
                
        except Exception as e:
            print(f"❌ TensorRT 분석 실패: {e}")
            return None
    
    def analyze_with_bbox(self, video_path: Path, bbox_normalized: List[float]) -> Optional[str]:
        """bbox 기반 마스크로 DAM 분석 (기본 모드 - 빠름)"""
        # TensorRT 우선 시도
        if self.use_tensorrt and self.tensorrt_optimizer:
            result = self._analyze_with_tensorrt(video_path, bbox_normalized, use_sam2=False)
            if result:
                return result
            print("🔄 기본 모드로 전환...")
        
        # 기본 subprocess 방식
        try:
            cmd = [
                sys.executable, str(self.dam_script_path),
                "--video_file", str(video_path),
                "--box", str(bbox_normalized),
                "--normalized_coords",
                "--use_box",
                "--no_stream",
                "--temperature", str(self.temperature),
                "--top_p", str(self.top_p),
                "--query", self.prompt,
            ]
            # --use_sam2 플래그 없음 = bbox 기반 마스크 사용
            
            print("🔍 DAM 분석 시작 (bbox 기반 마스크)...")
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print("[DAM stderr] ↓↓↓")
                print(result.stderr)
                raise RuntimeError(f"DAM exited {result.returncode}")
            
            description = self._extract_description(result.stdout or result.stderr)
            print(f"✅ DAM 분석 완료: {description}")
            return description
            
        except Exception as e:
            print(f"❌ DAM 분석 실패: {e}")
            return None
    
    def analyze_with_sam2(self, video_path: Path, bbox_normalized: List[float]) -> Optional[str]:
        """SAM2 세그멘테이션으로 DAM 분석 (선택적 모드 - 정확하지만 느림)"""
        # TensorRT 우선 시도
        if self.use_tensorrt and self.tensorrt_optimizer:
            result = self._analyze_with_tensorrt(video_path, bbox_normalized, use_sam2=True)
            if result:
                return result
            print("🔄 기본 모드로 전환...")
        
        # 기본 subprocess 방식
        try:
            cmd = [
                sys.executable, str(self.dam_script_path),
                "--video_file", str(video_path),
                "--box", str(bbox_normalized),
                "--normalized_coords",
                "--use_box",
                "--use_sam2",  # SAM2 처리 명시적 요청
                "--no_stream",
                "--temperature", str(self.temperature),
                "--top_p", str(self.top_p),
                "--query", self.prompt,
            ]
            
            print("🔍 DAM 분석 시작 (SAM2 세그멘테이션)...")
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print("[DAM stderr] ↓↓↓")
                print(result.stderr)
                raise RuntimeError(f"DAM exited {result.returncode}")
            
            description = self._extract_description(result.stdout or result.stderr)
            print(f"✅ DAM 분석 완료 (SAM2): {description}")
            return description
            
        except Exception as e:
            print(f"❌ DAM 분석 실패 (SAM2): {e}")
            return None
    
    def analyze_video(self, video_path: Path, bbox_normalized: List[float], use_sam2: bool = False) -> Optional[str]:
        """비디오 분석 (통합 메서드)"""
        if use_sam2:
            return self.analyze_with_sam2(video_path, bbox_normalized)
        else:
            return self.analyze_with_bbox(video_path, bbox_normalized)
    
    def set_prompt(self, new_prompt: str):
        """프롬프트 변경"""
        self.prompt = new_prompt
        print(f"📝 프롬프트 변경됨: {new_prompt[:50]}...")
    
    def set_parameters(self, temperature: float = None, top_p: float = None):
        """분석 파라미터 변경"""
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        print(f"⚙️ 파라미터 변경: temperature={self.temperature}, top_p={self.top_p}")
    
    def enable_tensorrt(self, force_rebuild: bool = False):
        """TensorRT 최적화 활성화"""
        if not self.use_tensorrt:
            self.use_tensorrt = True
            self._initialize_tensorrt()
        elif force_rebuild and self.tensorrt_optimizer:
            try:
                from dam_tensorrt_optimizer import create_optimized_dam_analyzer
                self.tensorrt_optimizer = create_optimized_dam_analyzer(
                    model_path="nvidia/DAM-3B-Video",
                    cache_dir=self.tensorrt_cache_dir,
                    force_rebuild=True
                )
                print("🔄 TensorRT 엔진 재빌드 완료")
            except Exception as e:
                print(f"❌ TensorRT 재빌드 실패: {e}")
    
    def disable_tensorrt(self):
        """TensorRT 최적화 비활성화"""
        self.use_tensorrt = False
        self.tensorrt_optimizer = None
        print("⚠️ TensorRT 최적화 비활성화 - 기본 모드 사용")
    
    def get_info(self) -> dict:
        """분석기 정보 반환"""
        info = {
            "dam_script_path": str(self.dam_script_path),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt": self.prompt,
            "use_tensorrt": self.use_tensorrt,
            "tensorrt_available": self.tensorrt_optimizer is not None
        }
        
        # TensorRT 성능 정보 추가
        if self.tensorrt_optimizer:
            info["tensorrt_info"] = self.tensorrt_optimizer.get_performance_info()
        
        return info 