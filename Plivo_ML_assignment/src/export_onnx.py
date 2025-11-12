import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os # Make sure os is imported

def export(model_name: str, max_length: int, out_path: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    mdl.eval()

    # Dummy inputs
    sample = tok("hello world", return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        torch.onnx.export(
            mdl,
            (sample["input_ids"], sample["attention_mask"]),
            out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            
            # --- THIS IS THE CORRECTED BLOCK ---
            # It should be 'dynamic_axes', NOT 'dynamic_shapes'
            # The values should be strings, NOT torch.export.Dim
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_len"},
                "attention_mask": {0: "batch_size", 1: "sequence_len"},
                "logits": {0: "batch_size", 1: "sequence_len"}
            },
            # ------------------------------------
        )
    
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

def quantize(in_path: str, out_path: str):
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--out", default="models/distilbert-base-uncased.onnx")
    ap.add_argument("--quant_out", default="models/distilbert-base-uncased.int8.onnx")
    args = ap.parse_args()

    # --- THIS IS ALSO NEEDED ---
    # Make sure the directories exist before writing to them
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.quant_out), exist_ok=True)
    # ---------------------------

    print(f"Exporting model to {args.out}...") # Added print for clarity
    export(args.model, args.max_length, args.out)
    
    print(f"Quantizing model to {args.quant_out}...") # Added print for clarity
    quantize(args.out, args.quant_out)
    
    print("Exported:", args.out)
    print("Quantized:", args.quant_out)