#!/usr/bin/env bash
set -e

# --- ADD THIS LINE ---
# 0) Create the output directory. -p means "don't error if it already exists"
echo "Ensuring 'models/' and 'out/' directories exist..."
mkdir -p models
mkdir -p out # This will be needed for step 2's output

# 1) Export and quantize DistilBERT to ONNX (first run only)
echo "Exporting and quantizing model..."
python -m src.export_onnx --model distilbert-base-uncased --max_length 64 --out models/distilbert-base-uncased.onnx --quant_out models/distilbert-base-uncased.int8.onnx

# 2) Run pipeline
echo "Running pipeline..."
python run_pipeline.py --onnx models/distilbert-base-uncased.int8.onnx

# 3) Evaluate metrics
echo "Evaluating metrics..."
python evaluate.py --pred out/corrected.jsonl --gold data/gold.jsonl --names data/names_lexicon.txt

# 4) Measure latency
echo "Measuring latency..."
python measure_latency.py --onnx models/distilbert-base-uncased.int8.onnx --runs 100 --warmup 10

echo "---"
echo "Script finished!"