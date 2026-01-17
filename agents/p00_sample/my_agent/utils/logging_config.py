"""
ロギング設定モジュール

p31_streamingのロギング実装を参考にした、シンプルなロギング設定
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
import time
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合はbackports.zoneinfoを使用
    from backports.zoneinfo import ZoneInfo


def jst_time(*args):
    """JST（日本時間）を返すconverter関数"""
    # UTC時間を取得してJSTに変換
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_jst = now_utc.astimezone(ZoneInfo('Asia/Tokyo'))
    # time.struct_time形式に変換
    return now_jst.timetuple()


class AlignedFormatter(logging.Formatter):
    """カスタムフォーマッター：ロガー名とログレベルを整列"""
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.converter = jst_time
        self.logger_name_width = 30
        self.level_name_width = 5
    
    def format(self, record):
        # ロガー名を整列
        logger_name = record.name
        if len(logger_name) > self.logger_name_width:
            logger_name = logger_name[:self.logger_name_width]
        else:
            logger_name = logger_name.ljust(self.logger_name_width)
        
        # ログレベルを整列
        level_name = record.levelname.ljust(self.level_name_width)
        
        # 整列されたフォーマットを作成
        aligned_format = f'%(asctime)s - {logger_name} - {level_name} - %(message)s'
        
        # 一時的なフォーマッターを作成
        temp_formatter = logging.Formatter(aligned_format, self.datefmt)
        temp_formatter.converter = jst_time
        return temp_formatter.format(record)


class LoggingConfig:
    """集中型ロギング設定"""
    
    def __init__(self):
        # 環境変数からログファイルパスを取得
        log_file = os.getenv('LOG_FILE', 'p00_sample.log')
        log_dir = os.getenv('LOG_DIR', '.')
        
        # ディレクトリが指定されている場合は結合
        if log_dir != '.':
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, os.path.basename(log_file))
        else:
            self.log_file = log_file
        
        # エラーログファイルパスを生成
        log_file_base = os.path.basename(log_file)
        if log_file_base.endswith('.log'):
            error_log_file = log_file_base.replace('.log', '_error.log')
        else:
            error_log_file = f"{log_file_base}_error.log"
        
        if log_dir != '.':
            self.error_log_file = os.path.join(log_dir, error_log_file)
        else:
            self.error_log_file = error_log_file
        
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
    def setup_logging(self, log_level: str = "INFO", initialize: bool = True) -> logging.Logger:
        """
        ロギング設定をセットアップ
        
        Args:
            log_level: ログレベル (DEBUG, INFO, WARNING, ERROR)
            initialize: 初期化するかどうか
            
        Returns:
            設定済みのルートロガー
        """
        # ルートロガーを作成
        root_logger = logging.getLogger('p00_sample')
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 既存のハンドラーをクリア（重複を避けるため）
        root_logger.handlers.clear()
        
        # ファイルハンドラーをセットアップ
        self._setup_file_handler(root_logger, log_level, initialize)
        
        # エラーファイルハンドラーをセットアップ
        self._setup_error_file_handler(root_logger, initialize)
        
        # コンソールハンドラーをセットアップ
        self._setup_console_handler(root_logger, log_level)
        
        # 伝播を防止（重複ログを避けるため）
        root_logger.propagate = False
        
        root_logger.debug(f"🔧 [LOGGING] ロギング設定が完了しました (ログレベル: {log_level})")
        return root_logger
    
    def _setup_file_handler(self, logger: logging.Logger, log_level: str, initialize: bool = True) -> None:
        """ファイルハンドラーをセットアップ（ローテーション付き）"""
        try:
            # ローテーションを使用するかどうか（環境変数で制御）
            use_python_rotation = os.getenv('LOG_USE_PYTHON_ROTATION', 'true').lower() == 'true'
            
            if use_python_rotation:
                # ローテーティングファイルハンドラーを作成
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                logger.debug(f"📁 [LOGGING] Pythonローテーション有効: {self.log_file}")
            else:
                # シンプルなファイルハンドラーを作成
                file_handler = logging.FileHandler(
                    filename=self.log_file,
                    encoding='utf-8'
                )
                logger.debug(f"📁 [LOGGING] logrotate使用（Pythonローテーション無効）: {self.log_file}")
            
            file_handler.setLevel(getattr(logging, log_level.upper()))
            
            # 整列フォーマッターを設定
            formatter = AlignedFormatter(
                fmt=self.log_format,
                datefmt=self.date_format
            )
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.debug(f"📁 [LOGGING] ファイルハンドラー設定完了: {self.log_file}")
            
        except Exception as e:
            logger.error(f"❌ [LOGGING] ファイルハンドラー設定エラー: {e}")
    
    def _setup_console_handler(self, logger: logging.Logger, log_level: str) -> None:
        """コンソールハンドラーをセットアップ"""
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            # 整列フォーマッターを設定
            formatter = AlignedFormatter(
                fmt=self.log_format,
                datefmt=self.date_format
            )
            console_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.debug("🖥️ [LOGGING] コンソールハンドラー設定完了")
            
        except Exception as e:
            logger.error(f"❌ [LOGGING] コンソールハンドラー設定エラー: {e}")
    
    def _setup_error_file_handler(self, logger: logging.Logger, initialize: bool = True) -> None:
        """エラーファイルハンドラーをセットアップ（ERROR/CRITICALログ用）"""
        try:
            # ローテーションを使用するかどうか（環境変数で制御）
            use_python_rotation = os.getenv('LOG_USE_PYTHON_ROTATION', 'true').lower() == 'true'
            
            if use_python_rotation:
                # エラー用のローテーティングファイルハンドラーを作成
                error_file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.error_log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
                logger.debug(f"📁 [LOGGING] エラーログ: Pythonローテーション有効: {self.error_log_file}")
            else:
                # エラー用のシンプルなファイルハンドラーを作成
                error_file_handler = logging.FileHandler(
                    filename=self.error_log_file,
                    encoding='utf-8'
                )
                logger.debug(f"📁 [LOGGING] エラーログ: logrotate使用（Pythonローテーション無効）: {self.error_log_file}")
            
            # ERRORとCRITICALログのみをキャプチャ
            error_file_handler.setLevel(logging.ERROR)
            
            # 整列フォーマッターを設定
            formatter = AlignedFormatter(
                fmt=self.log_format,
                datefmt=self.date_format
            )
            error_file_handler.setFormatter(formatter)
            
            logger.addHandler(error_file_handler)
            logger.debug(f"📁 [LOGGING] エラーログファイルハンドラー設定完了: {self.error_log_file}")
            
        except Exception as e:
            logger.error(f"❌ [LOGGING] エラーログファイルハンドラー設定エラー: {e}")


def setup_logging(log_level: str = "INFO", initialize: bool = True) -> logging.Logger:
    """
    ロギング設定をセットアップする便利関数
    
    Args:
        log_level: ログレベル (DEBUG, INFO, WARNING, ERROR)
        initialize: 初期化するかどうか
        
    Returns:
        設定済みのルートロガー
    """
    config = LoggingConfig()
    return config.setup_logging(log_level, initialize)


def get_logger(name: str) -> logging.Logger:
    """
    特定のモジュール用のロガーインスタンスを取得
    
    Args:
        name: ロガー名（通常は __name__）
        
    Returns:
        ロガーインスタンス
    """
    return logging.getLogger(f'p00_sample.{name}')


def get_log_level() -> str:
    """
    環境変数からログレベルを取得
    
    Returns:
        ログレベル文字列
    """
    environment = os.getenv('ENVIRONMENT', 'development').lower()
    log_level = os.getenv('LOG_LEVEL', '').upper()
    
    # 環境変数LOG_LEVELが明示的に設定されている場合はそれを使用
    if log_level:
        return log_level
    
    # 環境に基づくデフォルト値
    environment_defaults = {
        'production': 'INFO',
        'development': 'DEBUG',
        'staging': 'WARNING'
    }
    
    return environment_defaults.get(environment, 'INFO')


if __name__ == "__main__":
    # クイック検証
    print("✅ ロギング設定が利用可能です")
