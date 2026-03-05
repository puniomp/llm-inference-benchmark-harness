import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/run_1.csv")

plt.figure()
plt.plot(df["concurrency"], df["tokens_per_sec_est"], marker="o")
plt.xlabel("Concurrency")
plt.ylabel("Tokens / sec")
plt.title("vLLM Throughput Scaling")
plt.grid()

plt.savefig("results/throughput.png")

print("Saved results/throughput.png")
