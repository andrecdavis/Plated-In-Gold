"""Plated in Gold: analysis package for Apple Watch resale listings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / 'data' / 'raw' / 'Global_Smartwatch_Marketplace.csv'
IMAGES_DIR = PROJECT_ROOT / 'images'
