# Telegram Strategy Simulator (TSS)  
📊 **Instant Backtesting for Solana Trading Strategies via a TG-Style Dashboard**  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telegram-strategy-simulator-csyhcxe9xbkbgfmtr6qgzz.streamlit.app/)

---

## 🚀 Overview
**Telegram Strategy Simulator** is an interactive Streamlit app that simulates trading strategies on synthetic Solana token data.  
It mimics a Telegram bot experience with a **summary card**, equity curve, and dynamic metrics — all tunable in real-time.

This tool:
- Generates synthetic Solana-style token data (hourly)
- Engineers trading features from activity spikes, whale inflows, liquidity shifts
- Trains a **GradientBoostingClassifier** to predict short-term price movement
- Tunes the threshold to maximize *PnL* on a validation set
- Runs a **TP/SL/timed-exit backtest** on the test set
- Displays results TG-style with emoji indicators

---

## 📷 Screenshot
![App Screenshot](https://via.placeholder.com/1000x500.png?text=Telegram+Strategy+Simulator+Screenshot)

---

## ✨ Features
- **PnL-Optimized Threshold** selection  
- **Dynamic TP / SL / Timed Exit** controls  
- **Token Price Charts** with signal markers  
- **Equity Curve** from simulated trades  
- **14-day Trend Clustering** (Emerging / Stable / Declining)  

---

## 🛠 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/telegram-strategy-simulator.git
cd telegram-strategy-simulator

