"""
Utility functions for LightGCN
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logger(log_file: str = None, level: int = logging.INFO):
    """
    Setup logger with console and optional file handler.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to JSON file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Saved config to {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Loaded config from {config_path}")
    return config


def print_config(config: Dict[str, Any]):
    """Pretty print configuration"""
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    for key, value in config.items():
        print(f"  {key:.<30} {value}")
    print("="*50 + "\n")

