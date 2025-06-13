#!/usr/bin/env python3
"""
TensorRT 설치 스크립트
다른 컴퓨터에서 TensorRT를 설치하기 위한 스크립트
"""

import subprocess
import sys
import platform

def install_tensorrt():
    """TensorRT 설치"""
    print("🔧 TensorRT 설치 시작...")
    
    # 시스템 확인
    system = platform.system()
    print(f"운영체제: {system}")
    
    try:
        # PyTorch 확인
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA GPU를 찾을 수 없습니다")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다")
        return False
    
    # TensorRT 설치 시도
    commands = [
        "pip install tensorrt",
        "pip install nvidia-tensorrt", 
        "pip install onnx onnxruntime-gpu"
    ]
    
    for cmd in commands:
        print(f"실행: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✅ 성공: {cmd}")
        except subprocess.CalledProcessError:
            print(f"⚠️ 실패: {cmd}")
    
    # 설치 확인
    try:
        import tensorrt
        import onnx
        print(f"✅ TensorRT 버전: {tensorrt.__version__}")
        print(f"✅ ONNX 버전: {onnx.__version__}")
        print("🎉 TensorRT 설치 완료!")
        return True
    except ImportError:
        print("⚠️ TensorRT 설치 실패 - PyTorch 최적화만 사용됩니다")
        return False

if __name__ == "__main__":
    install_tensorrt() 