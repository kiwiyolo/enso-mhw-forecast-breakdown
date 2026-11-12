"""Paths only; retain the established data sources without rerunning circulation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PAPER=Path('/data3/luoq/FEIS/research/papers/mhw-predictability')
CONFIG={'roots':{'data':'/data3/luoq/FEIS-data'}}
