"""
ロギング設定モジュール
"""
import logging
import logging.handlers
import os
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# 一時的なロガー（設定前のログ用）
_temp_logger = logging.getLogger(__name__)


def jst_time(*args):
    """JST（日本時間）を返すconverter関数"""
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_jst = now_utc.astimezone(ZoneInfo('Asia/Tokyo'))
    return now_jst.timetuple()


class AlignedFormatter(logging.Formatter):
    """カスタムフォーマッター：ロガー名とログレベルを整列"""
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.converter = jst_time
        self.logger_name_width = 30
        self.level_name_width = 5
    
    def format(self, record):
        logger_name = record.name
        if len(logger_name) > self.logger_name_width:
            logger_name = logger_name[:self.logger_name_width]
        else:
            logger_name = logger_name.ljust(self.logger_name_width)
        
        level_name = record.levelname.ljust(self.level_name_width)
        aligned_format = f'%(asctime)s - {logger_name} - {level_name} - %(message)s'
        
        temp_formatter = logging.Formatter(aligned_format, self.datefmt)
        temp_formatter.converter = jst_time
        return temp_formatter.format(record)


class LoggingConfig:
    """集中型ロギング設定"""
    
    def __init__(self):
        # デフォルトのログファイル名を明示的に設定（環境変数が設定されていない場合）
        default_log_file = 'p02_ex_arm_react.log'
        log_file = os.getenv('LOG_FILE', default_log_file)
        
        # ログファイル名がp00_sample.logになっている場合は強制的に修正
        if 'p00_sample' in log_file:
            _temp_logger.warning(f"⚠️ [LOGGING] ログファイル名がp00_sampleになっています。{default_log_file}に変更します。")
            log_file = default_log_file
        
        log_dir = os.getenv('LOG_DIR', '.')
        
        if log_dir != '.':
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, os.path.basename(log_file))
        else:
            self.log_file = log_file
        
        log_file_base = os.path.basename(log_file)
        if log_file_base.endswith('.log'):
            error_log_file = log_file_base.replace('.log', '_error.log')
        else:
            error_log_file = f"{log_file_base}_error.log"
        
        if log_dir != '.':
            self.error_log_file = os.path.join(log_dir, error_log_file)
        else:
            self.error_log_file = error_log_file
        
        self.max_file_size = 10 * 1024 * 1024
        self.backup_count = 5
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
    def setup_logging(self, log_level: str = "INFO", initialize: bool = True) -> logging.Logger:
        # 他のロガー（p00_sampleなど）の影響を避けるため、先にクリア
        # p00_sampleのロガーをクリア
        p00_logger = logging.getLogger('p00_sample')
        p00_logger.handlers.clear()
        p00_logger.propagate = False
        p00_logger.setLevel(logging.WARNING)  # 警告レベルに設定して無効化
        
        # ルートロガー名を明示的に設定
        root_logger_name = 'p02_ex_arm_react'
        root_logger = logging.getLogger(root_logger_name)
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 既存のハンドラーを完全にクリア
        root_logger.handlers.clear()
        root_logger.propagate = False
        
        # 他のロガー（p00_sampleなど）の影響を避けるため、明示的にファイル名を設定
        # 注意: この時点ではloggerはまだ設定されていないので、printを使用
        print(f"📝 [LOGGING] ロガー名: {root_logger_name}, ログファイル: {self.log_file}")
        
        self._setup_file_handler(root_logger, log_level, initialize)
        self._setup_error_file_handler(root_logger, initialize)
        self._setup_console_handler(root_logger, log_level)
        
        root_logger.debug(f"🔧 [LOGGING] ロギング設定が完了しました (ログレベル: {log_level}, ログファイル: {self.log_file})")
        return root_logger
    
    def _setup_file_handler(self, logger: logging.Logger, log_level: str, initialize: bool = True) -> None:
        try:
            use_python_rotation = os.getenv('LOG_USE_PYTHON_ROTATION', 'true').lower() == 'true'
            
            if use_python_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(
                    filename=self.log_file,
                    encoding='utf-8'
                )
            
            file_handler.setLevel(getattr(logging, log_level.upper()))
            formatter = AlignedFormatter(fmt=self.log_format, datefmt=self.date_format)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.error(f"❌ [LOGGING] ファイルハンドラー設定エラー: {e}")
    
    def _setup_console_handler(self, logger: logging.Logger, log_level: str) -> None:
        try:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            formatter = AlignedFormatter(fmt=self.log_format, datefmt=self.date_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            logger.error(f"❌ [LOGGING] コンソールハンドラー設定エラー: {e}")
    
    def _setup_error_file_handler(self, logger: logging.Logger, initialize: bool = True) -> None:
        try:
            use_python_rotation = os.getenv('LOG_USE_PYTHON_ROTATION', 'true').lower() == 'true'
            
            if use_python_rotation:
                error_file_handler = logging.handlers.RotatingFileHandler(
                    filename=self.error_log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding='utf-8'
                )
            else:
                error_file_handler = logging.FileHandler(
                    filename=self.error_log_file,
                    encoding='utf-8'
                )
            
            error_file_handler.setLevel(logging.ERROR)
            formatter = AlignedFormatter(fmt=self.log_format, datefmt=self.date_format)
            error_file_handler.setFormatter(formatter)
            logger.addHandler(error_file_handler)
            
        except Exception as e:
            logger.error(f"❌ [LOGGING] エラーログファイルハンドラー設定エラー: {e}")


def setup_logging(log_level: str = "INFO", initialize: bool = True) -> logging.Logger:
    config = LoggingConfig()
    return config.setup_logging(log_level, initialize)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f'p02_ex_arm_react.{name}')


def get_log_level() -> str:
    environment = os.getenv('ENVIRONMENT', 'development').lower()
    log_level = os.getenv('LOG_LEVEL', '').upper()
    
    if log_level:
        return log_level
    
    environment_defaults = {
        'production': 'INFO',
        'development': 'DEBUG',
        'staging': 'WARNING'
    }
    
    return environment_defaults.get(environment, 'INFO')
