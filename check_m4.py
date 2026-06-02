import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cache import load_data
from signal_engine import tax_period_analysis, compute_m2_signals, compute_m5_signals, compute_m4_signals

data = load_data()
signals = {
    "m2": compute_m2_signals(data["m2"]),
    "m4": compute_m4_signals(data["m4"]),
    "m5": compute_m5_signals(data["m5"]),
}

m4 = data["m4"]
print("М4: флаги")
print(f"tax_week_flag=1: {m4['tax_week_flag'].mean()*100:.1f}%")
print(f"end_of_month_flag=1: {m4['end_of_month_flag'].mean()*100:.1f}%")
print(f"end_of_quarter_flag=1: {m4['end_of_quarter_flag'].mean()*100:.1f}%")
print(f"seasonal_factor: {m4['seasonal_factor'].min():.2f} – {m4['seasonal_factor'].max():.2f}")

print()
print("налоговые vs обычные периоды:")
print(tax_period_analysis(signals).to_string(index=False))
