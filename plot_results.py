import pandas as pd
import matplotlib.pyplot as plt

# Load the latest benchmark run
df = pd.read_csv("results/run_2.csv")

# Sort by concurrency in case the CSV isn't ordered
df = df.sort_values("concurrency")

# -------------------------
# Throughput Plot
# -------------------------
plt.figure()
plt.plot(df["concurrency"], df["tokens_per_sec"], marker="o")
plt.xlabel("Concurrency")
plt.ylabel("Tokens / Second")
plt.title("Throughput vs Concurrency")
plt.grid(True)
plt.savefig("results/throughput.png")

# -------------------------
# Latency Percentiles Plot
# -------------------------
plt.figure()
plt.plot(df["concurrency"], df["lat_p50_s"], marker="o", label="p50")
plt.plot(df["concurrency"], df["lat_p95_s"], marker="o", label="p95")
plt.plot(df["concurrency"], df["lat_p99_s"], marker="o", label="p99")

plt.xlabel("Concurrency")
plt.ylabel("Latency (seconds)")
plt.title("Latency Percentiles vs Concurrency")
plt.legend()
plt.grid(True)

plt.savefig("results/latency_percentiles.png")

print("Saved plots:")
print("results/throughput.png")
print("results/latency_percentiles.png")
