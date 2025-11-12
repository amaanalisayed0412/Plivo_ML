import json, time
from typing import Dict, List, Set # <-- Import Set
from .rules import generate_candidates
from .ranker_onnx import PseudoLikelihoodRanker

class PostProcessor:
    def __init__(self, names_lex_path: str, misspell_map_path: str, onnx_model_path: str = None, device: str = "cpu", max_length: int = 64):
        # Load the lexicon as a list (used by rules.py)
        self.names_lex_list = [x.strip() for x in open(names_lex_path, 'r', encoding='utf-8').read().splitlines() if x.strip()]
        
        # --- ADD THIS ---
        # Create a lowercase set for fast O(1) name checking in process_one
        self.names_lex_lower_set: Set[str] = {name.lower() for name in self.names_lex_list}
        # --- END ADD ---

        self.ranker = PseudoLikelihoodRanker(onnx_path=onnx_model_path, device=device, max_length=max_length)
        
        with open(misspell_map_path, 'r', encoding='utf-8') as f:
            self.misspell = json.load(f)

    def process_one(self, text: str) -> str:
        # 1. Generate and rank candidates
        # We pass the original list here for the rules to use
        cands = generate_candidates(text, self.names_lex_list, self.misspell)
        best = self.ranker.choose_best(cands)

        # --- START NEW/MODIFIED LOGIC ---

        # 2. Capitalize names (Goal 1), but skip emails
        processed_tokens = []
        for token in best.split():
            # Check 1: Is it an email? If yes, leave it alone.
            if '@' in token:
                processed_tokens.append(token)
            
            # Check 2: Is it a name?
            # We strip punctuation to match "Alok." or "Ansh?"
            elif token.lower().strip('.,?!') in self.names_lex_lower_set:
                # .capitalize() handles "alok" -> "Alok"
                processed_tokens.append(token.capitalize())
            else:
                processed_tokens.append(token)
        
        best = ' '.join(processed_tokens)

        # 3. Final polish: Capitalize first letter (Goal 2) & add punctuation
        best = best.strip()
        if not best:
            return "."

        # Capitalize the first letter of the entire sentence
        best = best[0].upper() + best[1:]
        
        # --- END NEW/MODIFIED LOGIC ---
        
        # (Original punctuation logic)
        lower = best.lower() # Use the new 'best'
        if lower.endswith(('?', '.', ',')) is False:
            first_word = lower.split()[0] if lower.split() else ""
            if first_word in ('can','shall','will','could','would','is','are','do','does','did','should','what','where','when','why','who','how'):
                best = best.rstrip() + '?'
            else:
                best = best.rstrip() + '.'
        return best

# ... (rest of the file is unchanged) ...

def run_file(input_path: str, output_path: str, names_lex_path: str, misspell_map_path: str, onnx_model_path: str = None, device: str = "cpu", max_length: int = 64):
    pp = PostProcessor(names_lex_path, misspell_map_path=misspell_map_path, onnx_model_path=onnx_model_path, device=device, max_length=max_length)
    
    # --- ADDED: Ensure output directory exists ---
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # --- END ADD ---

    rows = [json.loads(line) for line in open(input_path, 'r', encoding='utf-8')]
    out = []
    for r in rows:
        pred = pp.process_one(r["text"])
        out.append({"id": r["id"], "text": pred})
    with open(output_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

# --- ADDED: Argument parsing to make the script runnable ---
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    # Add arguments for all paths
    ap.add_argument("--input", default="data/noisy_transcripts.jsonl", help="Input noisy transcripts file")
    ap.add_argument("--output", default="out/corrected.jsonl", help="Output corrected transcripts file")
    ap.add_argument("--names", default="data/names_lexicon.txt", help="Path to names lexicon")
    ap.add_argument("--misspell", default="data/misspell_map.json", help="Path to misspell map")
    ap.add_argument("--onnx", default="models/distilbert-base-uncased.int8.onnx", help="Path to ONNX model")
    ap.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    ap.add_argument("--max_length", type=int, default=64, help="Max sequence length for ranker")
    
    args = ap.parse_args()

    run_file(
        input_path=args.input,
        output_path=args.output,
        names_lex_path=args.names,
        misspell_map_path=args.misspell,
        onnx_model_path=args.onnx,
        device=args.device,
        max_length=args.max_length
    )
    
    print(f"Processing complete. Output written to {args.output}")
# --- END ADD ---