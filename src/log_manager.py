#!/usr/bin/env python3
"""
Log Manager Module
로그 저장 및 관리 기능을 담당
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

class LogManager:
    """로그 관리 클래스"""
    
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        
        # 로그 파일 디렉토리 생성
        self.log_file_path.parent.mkdir(exist_ok=True)
        
        # 로그 파일이 없으면 헤더 생성
        if not self.log_file_path.exists():
            self._create_log_file()
    
    def _create_log_file(self):
        """로그 파일 생성 및 헤더 추가"""
        try:
            with self.log_file_path.open("w", encoding="utf8") as f:
                f.write("# Action Log File\n")
                f.write("# Format: TIME_RANGE\\tDESCRIPTION\\tBBOX_INFO\n")
                f.write("# Created: " + datetime.now().isoformat() + "\n")
                f.write("\n")
            print(f"📝 로그 파일 생성: {self.log_file_path}")
        except Exception as e:
            print(f"❌ 로그 파일 생성 실패: {e}")
    
    def append_log(self, start_dt: datetime, end_dt: datetime, description: str, bbox_info: Optional[Dict] = None) -> bool:
        """로그 항목 추가"""
        try:
            # 시간 범위 포맷
            time_range = f"{start_dt.strftime('%Y-%m-%d-%H%M%S')}~{end_dt.strftime('%H%M%S')}"
            
            # 로그 데이터 구성
            log_data = {
                'time_range': time_range,
                'description': description,
                'bbox_info': bbox_info,
                'timestamp': datetime.now().isoformat()
            }
            
            # 파일에 추가
            with self.log_file_path.open("a", encoding="utf8") as f:
                if bbox_info:
                    f.write(f"{time_range}\t{description}\t{json.dumps(bbox_info, ensure_ascii=False)}\n")
                else:
                    f.write(f"{time_range}\t{description}\n")
            
            print(f"📝 로그 저장 완료: {description}")
            return True
            
        except Exception as e:
            print(f"❌ 로그 저장 실패: {e}")
            return False
    
    def log_analysis_result(self, video_path: Path, bbox_normalized: List[float], 
                          description: str, analysis_mode: str = "bbox_based", 
                          duration_sec: int = 5) -> bool:
        """분석 결과 로그"""
        start_dt = datetime.now()
        end_dt = start_dt + timedelta(seconds=duration_sec)
        
        bbox_info = {
            'bbox_normalized': bbox_normalized,
            'analysis_mode': analysis_mode,
            'video_path': str(video_path),
            'duration_sec': duration_sec
        }
        
        return self.append_log(start_dt, end_dt, description, bbox_info)
    
    def log_api_trigger(self, signal_type: str, bbox_normalized: List[float], 
                       metadata: Dict, description: str = None) -> bool:
        """API 트리거 로그"""
        start_dt = datetime.now()
        end_dt = start_dt + timedelta(seconds=1)
        
        log_description = description or f"API trigger: {signal_type}"
        
        bbox_info = {
            'signal_type': signal_type,
            'bbox_normalized': bbox_normalized,
            'metadata': metadata,
            'source': 'api_trigger'
        }
        
        return self.append_log(start_dt, end_dt, log_description, bbox_info)
    
    def read_recent_logs(self, count: int = 10) -> List[str]:
        """최근 로그 읽기"""
        try:
            if not self.log_file_path.exists():
                return []
            
            with self.log_file_path.open("r", encoding="utf8") as f:
                lines = f.readlines()
            
            # 주석이 아닌 라인만 필터링
            log_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            
            # 최근 count개 반환
            return log_lines[-count:] if log_lines else []
            
        except Exception as e:
            print(f"❌ 로그 읽기 실패: {e}")
            return []
    
    def get_log_stats(self) -> Dict:
        """로그 통계 정보"""
        try:
            if not self.log_file_path.exists():
                return {"total_entries": 0, "file_size": 0}
            
            # 파일 크기
            file_size = self.log_file_path.stat().st_size
            
            # 로그 항목 수 계산
            with self.log_file_path.open("r", encoding="utf8") as f:
                lines = f.readlines()
            
            log_entries = len([line for line in lines if line.strip() and not line.startswith('#')])
            
            return {
                "total_entries": log_entries,
                "file_size": file_size,
                "file_path": str(self.log_file_path),
                "last_modified": datetime.fromtimestamp(self.log_file_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            print(f"❌ 로그 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def clear_logs(self) -> bool:
        """로그 파일 초기화"""
        try:
            self._create_log_file()
            print("🗑️ 로그 파일 초기화 완료")
            return True
        except Exception as e:
            print(f"❌ 로그 파일 초기화 실패: {e}")
            return False
    
    def backup_logs(self, backup_suffix: str = None) -> Optional[Path]:
        """로그 파일 백업"""
        try:
            if not self.log_file_path.exists():
                print("❌ 백업할 로그 파일이 없습니다.")
                return None
            
            # 백업 파일명 생성
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            backup_path = self.log_file_path.with_suffix(f".{backup_suffix}.txt")
            
            # 파일 복사
            import shutil
            shutil.copy2(self.log_file_path, backup_path)
            
            print(f"💾 로그 백업 완료: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"❌ 로그 백업 실패: {e}")
            return None 