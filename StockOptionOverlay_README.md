## **Stock Option Overlay Strategies**

### **Project Overview:**
This project models theoretical stock option overlay strategies on SPY by simulating monthly option pricing using historical VIX data, SOFR rates, and dividend yields. The analysis evaluates portfolio performance based on dynamic out-of-the-money (OTM) strike selection and statistical price movements.

### **Key Features:**
1. **Options Pricing:** Uses the VIX index to compute 1-month implied volatilities.
2. **OTM Strike Selection:** Dynamically selects OTM call and put strikes using 1-standard deviation price moves.
3. **Portfolio Analysis:** Compares different stock overlay strategies (e.g., covered calls, protective puts).
4. **Data Sources:** Includes historical SOFR rates and SPY option data.

### **Installation:**
```bash
pip install pandas numpy yfinance matplotlib
```

### **Usage:**
```bash
python StockOptionOverlay.py
```

### **Results:**
1. **Implied Volatility vs Time:** A plot showing the historical VIX values used for the options pricing.
2. **Portfolio Performance:** A line chart comparing the performance of different overlay strategies.
