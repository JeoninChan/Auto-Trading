#!/bin/bash
cd /Users/changpt/Downloads/stock_trading
python3 day_trading/local_trainer.py --skip-download --replay --cases mid_large --replay-start 2024-06-12 --threshold 0.5
