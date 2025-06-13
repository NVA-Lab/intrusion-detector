#!/usr/bin/env python3
"""
API-based External Controller
's' 키를 눌러 demo.py API에 녹화 신호를 보내는 스크립트
"""

import requests
import json
import time
import random
from datetime import datetime
import keyboard  # pip install keyboard

# API 설정
API_BASE_URL = "http://localhost:5000"

def send_recording_signal(bbox_normalized=None, signal_type="manual_trigger", metadata=None):
    """API를 통해 demo.py에 녹화 신호를 보냅니다"""
    if bbox_normalized is None:
        # 기본 bbox (화면 중앙 영역)
        bbox_normalized = [0.25, 0.25, 0.75, 0.75]
    
    if metadata is None:
        metadata = {}
    
    # API 요청 데이터
    payload = {
        "signal_type": signal_type,
        "bbox_normalized": bbox_normalized,
        "metadata": metadata
    }
    
    try:
        # POST 요청 전송
        response = requests.post(
            f"{API_BASE_URL}/trigger_recording",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ [SUCCESS] {result['message']}")
            print(f"   Signal Type: {result['signal_type']}")
            print(f"   BBox: {result['bbox_normalized']}")
            return True
        else:
            error_data = response.json()
            print(f"❌ [ERROR] {error_data.get('message', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ [ERROR] demo.py API 서버에 연결할 수 없습니다.")
        print("   demo.py가 실행 중인지 확인해주세요.")
        return False
    except requests.exceptions.Timeout:
        print("❌ [ERROR] API 요청 시간 초과")
        return False
    except Exception as e:
        print(f"❌ [ERROR] API 요청 실패: {e}")
        return False

def get_system_status():
    """시스템 상태를 확인합니다"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        
        if response.status_code == 200:
            status = response.json()
            print("📊 [STATUS]")
            print(f"   Recording Active: {status['recording_active']}")
            print(f"   Signals in Queue: {status['signals_in_queue']}")
            print(f"   Timestamp: {status['timestamp']}")
            return True
        else:
            print("❌ [ERROR] 상태 조회 실패")
            return False
            
    except Exception as e:
        print(f"❌ [ERROR] 상태 조회 실패: {e}")
        return False

def on_s_key_pressed():
    """'s' 키가 눌렸을 때 실행되는 함수"""
    print("\n🎬 's' 키가 눌렸습니다! API로 녹화 신호를 보냅니다...")
    
    # 랜덤한 bbox 생성 (실제 환경에서는 추적 시스템에서 제공)
    x1 = random.uniform(0.1, 0.4)
    y1 = random.uniform(0.1, 0.4)
    x2 = random.uniform(0.6, 0.9)
    y2 = random.uniform(0.6, 0.9)
    bbox = [round(x, 3) for x in [x1, y1, x2, y2]]
    
    metadata = {
        "source": "api_controller",
        "confidence": round(random.uniform(0.8, 0.99), 2),
        "trigger_method": "manual_key_press",
        "timestamp": datetime.now().isoformat()
    }
    
    send_recording_signal(bbox, "manual_api_trigger", metadata)

def on_t_key_pressed():
    """'t' 키가 눌렸을 때 상태 확인"""
    print("\n📊 시스템 상태를 확인합니다...")
    get_system_status()

def test_connection():
    """연결 테스트 - 상태 확인만 수행 (녹화 신호 없음)"""
    print("🔗 demo.py API 서버와의 연결을 테스트합니다...")
    if get_system_status():
        print("✅ 연결 성공!")
        return True
    else:
        print("❌ 연결 실패!")
        return False

def main():
    """메인 함수"""
    print("=" * 60)
    print("🎮 API 기반 외부 컨트롤러가 시작되었습니다!")
    print(f"🌐 API 서버: {API_BASE_URL}")
    print()
    print("📋 사용법:")
    print("  's' 키: 녹화 신호 전송")
    print("  't' 키: 시스템 상태 확인")
    print("  'c' 키: 연결 테스트")
    print("  'q' 키: 종료")
    print("=" * 60)
    
    # 초기 연결 테스트
    print("\n🔄 초기 연결 테스트를 수행합니다...")
    time.sleep(1)
    test_connection()
    
    # 키보드 이벤트 등록
    keyboard.add_hotkey('s', on_s_key_pressed)
    keyboard.add_hotkey('t', on_t_key_pressed)
    keyboard.add_hotkey('c', test_connection)
    
    print("\n✅ 컨트롤러가 준비되었습니다. 키를 눌러보세요!")
    
    try:
        # 프로그램 실행 유지
        keyboard.wait('q')  # 'q' 키를 누를 때까지 대기
    except KeyboardInterrupt:
        pass
    
    print("\n👋 API 컨트롤러를 종료합니다.")

if __name__ == "__main__":
    main() 