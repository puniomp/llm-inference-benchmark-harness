import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity, schedule

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRACE_PATH = "results/profiling/pytorch/trace_lean.json"

os.makedirs("results/profiling/pytorch", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cuda",
)

model.eval()

prompt = "Explain how GPU batching impacts LLM inference throughput."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

def run_inference():
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

for _ in range(2):
    run_inference()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
) as prof:
    for _ in range(2):
        run_inference()
        prof.step()

prof.export_chrome_trace(TRACE_PATH)
print(f"trace exported to {TRACE_PATH}")
