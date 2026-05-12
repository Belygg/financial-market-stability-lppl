# Financial Market Stability under Manipulation Crisis Conditions
### A Network-Based Approach with LPPL Backtesting and ML Signal Generation

## Project Overview

This project analyzes financial market stability during manipulation and crisis conditions using:
- **Network-based modeling** of market structure and contagion
- **LPPL (Log-Periodic Power Law) backtesting system** for bubble/crash detection
- **Machine learning signal generator** producing actionable trading signals

## Signal Output Format

```
Signal:             buy / sell / hold
Confidence:         0.00 – 1.00   (e.g. 0.76)
Suggested exposure: % of portfolio (e.g. 60%)
Maximum leverage:   risk-capped   (e.g. 1.5x)
```

## Repository Structure

```
├── data/               # Raw and processed market data
├── lppl/               # LPPL model implementation and backtester
├── network/            # Network-based market analysis
├── ml/                 # ML models for signal generation
├── signals/            # Signal output and evaluation
├── notebooks/          # Exploratory analysis notebooks
└── tests/              # Unit and integration tests
```

## Core Components

### 1. LPPL Backtesting System
- Parameter fitting (A, B, C, tc, m, ω, φ)
- Bubble detection and crash timing
- Historical backtesting with walk-forward validation
- Performance metrics (Sharpe, drawdown, hit rate)

### 2. Network Analysis
- Correlation/partial-correlation market graphs
- Systemic risk and contagion modeling
- Manipulation detection via network anomalies

### 3. ML Signal Generator
- Features from LPPL fits + network metrics
- Outputs: signal label, confidence score, exposure %, max leverage
- Risk-adjusted position sizing

## Setup

```bash
pip install -r requirements.txt
```

## Authors

Belygg + Claude Code
