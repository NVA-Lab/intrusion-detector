#!/usr/bin/env python3
"""
DAM TensorRT 최적화 모듈
DAM 모델의 추론 속도를 최적화 (단순화된 접근법)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

class DAMTensorRTOptimizer:
    """DAM 모델 최적화기 (GPU 메모리 + Torch Compile)"""
    
    def __init__(self, model_path: str = "nvidia/DAM-3B-Video", 
                 cache_dir: str = "tensorrt_cache",
                 force_rebuild: bool = False):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.force_rebuild = force_rebuild
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # DAM 모델 관련
        self.dam_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 성능 정보
        self.performance_info = {
            "optimization_enabled": False,
            "inference_time": 0.0,
            "memory_usage": 0,
            "batch_size": 1,
        }
    
    def optimize_model(self, force_rebuild: bool = False) -> bool:
        """모델 최적화 (단순화된 버전)"""
        try:
            # DAM 모델 로드
            if not self._load_dam_model():
                return False
            
            # GPU 메모리 최적화
            self._optimize_gpu_memory()
            
            # 컴파일 최적화 (PyTorch 2.0+)
            self._apply_torch_compile()
            
            self.performance_info["optimization_enabled"] = True
            self.logger.info("DAM 모델 최적화 완료 (GPU 메모리 + Torch Compile)")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 최적화 실패: {e}")
            return False
    
    def _load_dam_model(self) -> bool:
        """DAM 모델 로드"""
        try:
            from dam import DescribeAnythingModel, disable_torch_init
            
            disable_torch_init()
            
            self.dam_model = DescribeAnythingModel(
                model_path=self.model_path,
                conv_mode="v1",
                prompt_mode="full+focal_crop"
            )
            
            # GPU로 이동
            self.dam_model = self.dam_model.to(self.device)
            self.dam_model.eval()
            
            self.logger.info(f"DAM 모델 로드 완료: {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"DAM 모델 로드 실패: {e}")
            return False
    
    def _optimize_gpu_memory(self):
        """GPU 메모리 최적화"""
        if torch.cuda.is_available():
            # 메모리 정리
            torch.cuda.empty_cache()
            
            # 혼합 정밀도 사용
            if hasattr(self.dam_model, 'model'):
                self.dam_model.model = self.dam_model.model.half()
            
            # CUDA 그래프 최적화 준비
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _apply_torch_compile(self):
        """PyTorch 2.0 컴파일 최적화 적용"""
        try:
            import platform
            
            # Windows에서는 Triton이 지원되지 않으므로 torch.compile 건너뜀
            if platform.system() == "Windows":
                self.logger.info("Windows 환경 - torch.compile 건너뜀 (Triton 미지원)")
                self._apply_windows_optimizations()
                return
            
            # Triton 설치 확인 (Linux/macOS)
            try:
                import triton
                self.logger.info(f"Triton 버전: {triton.__version__}")
            except ImportError:
                self.logger.warning("Triton이 설치되지 않음 - torch.compile 건너뜀")
                self._apply_fallback_optimizations()
                return
            
            # torch.compile 사용 가능 확인
            if not hasattr(torch, 'compile'):
                self.logger.warning("PyTorch 2.0+ 필요 - torch.compile 건너뜀")
                self._apply_fallback_optimizations()
                return
            
            if not hasattr(self.dam_model, 'model'):
                self.logger.warning("DAM 모델 구조 확인 실패 - torch.compile 건너뜀")
                return
            
            # Dynamo 오류 억제 설정
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            # 핵심 모델 부분만 컴파일 (안전한 모드)
            try:
                self.dam_model.model.llm = torch.compile(
                    self.dam_model.model.llm,
                    mode="default",  # reduce-overhead 대신 default 사용
                    fullgraph=False,
                    dynamic=True
                )
                self.logger.info("LLM 모듈 컴파일 완료")
            except Exception as e:
                self.logger.warning(f"LLM 컴파일 실패 (무시됨): {e}")
            
            # 비전 타워 컴파일 (선택적)
            try:
                if hasattr(self.dam_model.model, 'vision_tower'):
                    self.dam_model.model.vision_tower = torch.compile(
                        self.dam_model.model.vision_tower,
                        mode="default",
                        fullgraph=False,
                        dynamic=True
                    )
                    self.logger.info("Vision Tower 컴파일 완료")
            except Exception as e:
                self.logger.warning(f"Vision Tower 컴파일 실패 (무시됨): {e}")
            
            self.logger.info("Torch Compile 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"Torch Compile 최적화 전체 실패 (무시됨): {e}")
            self._apply_fallback_optimizations()
    
    def _apply_windows_optimizations(self):
        """Windows 환경용 최적화"""
        try:
            # JIT 스크립팅 (torch.compile 대안)
            if hasattr(self.dam_model, 'model') and hasattr(self.dam_model.model, 'llm'):
                # 일부 모듈을 JIT으로 최적화
                if hasattr(self.dam_model.model.llm, 'model'):
                    # 임베딩 레이어 최적화
                    if hasattr(self.dam_model.model.llm.model, 'embed_tokens'):
                        try:
                            self.dam_model.model.llm.model.embed_tokens = torch.jit.script(
                                self.dam_model.model.llm.model.embed_tokens
                            )
                            self.logger.info("임베딩 레이어 JIT 최적화 완료")
                        except Exception as e:
                            self.logger.warning(f"임베딩 JIT 최적화 실패: {e}")
            
            # 추가 Windows 최적화
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            # 메모리 할당 최적화
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.9)
            
            self.logger.info("Windows 환경 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"Windows 최적화 실패: {e}")
    
    def _apply_fallback_optimizations(self):
        """Fallback 최적화 (torch.compile 사용 불가 시)"""
        try:
            # 기본 최적화만 적용
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 그래디언트 체크포인팅 비활성화 (추론 시)
            if hasattr(self.dam_model, 'model'):
                for module in self.dam_model.model.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = False
            
            self.logger.info("Fallback 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"Fallback 최적화 실패: {e}")
    
    def infer(self, images: List[Image.Image], masks: List[Image.Image]) -> Optional[str]:
        """최적화된 추론"""
        if self.dam_model is None:
            self.logger.error("모델이 로드되지 않았습니다")
            return None
        
        try:
            start_time = time.time()
            
            # 기본 프롬프트
            query = (
                "Video: <image><image><image><image><image><image><image><image>\n"
                "Return **one concise English sentence** that describes ONLY the subject's action or state change. "
                "Do NOT mention appearance, colour, clothing, background, objects, or physical attributes."
            )
            
            # GPU 메모리 사용량 측정 시작
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # DAM 추론 실행
            with torch.inference_mode():
                description = self.dam_model.get_description(
                    images, masks, query,
                    streaming=False,
                    temperature=0.1,
                    top_p=0.15,
                    num_beams=1,
                    max_new_tokens=512
                )
            
            # 성능 정보 업데이트
            inference_time = time.time() - start_time
            self.performance_info["inference_time"] = inference_time
            
            if torch.cuda.is_available():
                self.performance_info["memory_usage"] = torch.cuda.max_memory_allocated()
            
            self.logger.info(f"추론 완료: {inference_time:.3f}초")
            return description
            
        except Exception as e:
            self.logger.error(f"추론 실패: {e}")
            return None
    
    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        info = self.performance_info.copy()
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved(),
            })
        
        return info
    
    def warmup(self, num_iterations: int = 3):
        """모델 워밍업"""
        if self.dam_model is None:
            return
        
        try:
            # 더미 데이터 생성
            dummy_images = [Image.new('RGB', (224, 224), color='white') for _ in range(8)]
            dummy_masks = [Image.new('L', (224, 224), color=255) for _ in range(8)]
            
            self.logger.info(f"모델 워밍업 시작 ({num_iterations}회)")
            
            for i in range(num_iterations):
                self.infer(dummy_images, dummy_masks)
                self.logger.info(f"워밍업 {i+1}/{num_iterations} 완료")
            
            self.logger.info("모델 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"워밍업 실패: {e}")


def create_optimized_dam_analyzer(model_path: str = "nvidia/DAM-3B-Video",
                                cache_dir: str = "tensorrt_cache",
                                force_rebuild: bool = False) -> DAMTensorRTOptimizer:
    """최적화된 DAM 분석기 생성"""
    optimizer = DAMTensorRTOptimizer(
        model_path=model_path,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild
    )
    
    # 모델 최적화
    if optimizer.optimize_model(force_rebuild=force_rebuild):
        # 워밍업 실행
        optimizer.warmup()
        return optimizer
    else:
        raise RuntimeError("DAM 모델 최적화 실패")


if __name__ == "__main__":
    # 테스트 코드
    optimizer = create_optimized_dam_analyzer()
    
    # 더미 테스트
    dummy_images = [Image.new('RGB', (224, 224), color='white') for _ in range(8)]
    dummy_masks = [Image.new('L', (224, 224), color=255) for _ in range(8)]
    
    result = optimizer.infer(dummy_images, dummy_masks)
    print(f"테스트 결과: {result}")
    
    # 성능 정보 출력
    perf_info = optimizer.get_performance_info()
    for key, value in perf_info.items():
        print(f"{key}: {value}") 