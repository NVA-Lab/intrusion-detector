#!/usr/bin/env python3
"""
DAM Analyzer Module
DAM 비디오 분석 기능을 담당
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import List, Optional

class DAMAnalyzer:
    """DAM 분석 클래스"""
    
    def __init__(self, dam_script_path: Path, temperature: float = 0.1, top_p: float = 0.15):
        self.dam_script_path = dam_script_path
        self.temperature = temperature
        self.top_p = top_p
        
        # DAM 스크립트 존재 확인
        if not self.dam_script_path.exists():
            raise FileNotFoundError(f"DAM script not found at: {self.dam_script_path}")
        
        # 기본 프롬프트
        self.prompt = (
            "Video: <image><image><image><image><image><image><image><image>\n"
            "Return **one concise English sentence** that describes ONLY the subject's action or state change. "
            "Do NOT mention appearance, colour, clothing, background, objects, or physical attributes."
        )
    
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
    
    def analyze_with_bbox(self, video_path: Path, bbox_normalized: List[float]) -> Optional[str]:
        """bbox 기반 마스크로 DAM 분석 (기본 모드 - 빠름)"""
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
    
    def get_info(self) -> dict:
        """분석기 정보 반환"""
        return {
            "dam_script_path": str(self.dam_script_path),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt": self.prompt
        } 