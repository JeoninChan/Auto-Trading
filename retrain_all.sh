#!/bin/bash
cd /Users/changpt/Downloads/stock_trading
python3 day_trading/local_trainer.py --skip-download --cases leverage_mid_large mid_large smallcap small_mid all --train-cutoff 2025-06-01 --no-backtest
