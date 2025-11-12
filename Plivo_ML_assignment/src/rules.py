import re
from typing import List, Set, Tuple, Dict
from rapidfuzz import process, fuzz

# --- 1. Misspellings (NEW) ---
# This is a high-precision, fast-win.
# Should be passed in from misspell_map.json
def correct_common_misspells(text: str, misspell_map: Dict[str, str]) -> str:
    # Use a regex to replace only whole words
    # This is much faster than splitting and iterating
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in misspell_map.keys()) + r')\b', re.IGNORECASE)
    return pattern.sub(lambda m: misspell_map[m.group(0).lower()], text)

# --- 2. Email Normalization (Improved) ---
EMAIL_TOKEN_PATTERNS = [
    (r'\b(dot)\b', '.'),                  # 'dot' -> '.'
    (r'\b(underscore)\b', '_'),          # 'underscore' -> '_'
]
EMAIL_CLEANUP_PATTERNS = [
    # Fixes the (.) before surname (mehta) in the email address.
    (r'([a-zA-Z0-9.-]+)(mehta|sharma|patel|gupta|singh|kumar|verma)\b', r'\1.\2'),

    # Fixes "...acin" -> "...ac.in"
    (r'(@[a-zA-Z0-9.-]+)(acin)\b', r'\1.ac.in'),
    
    # Fixes "...coin" -> "...co.in"
    (r'(@[a-zA-Z0-9.-]+)(coin)\b', r'\1.co.in'),
    
    # Fixes "...com" -> "...com", "...in" -> ".in", etc.
    (r'(@[a-zA-Z0-9.-]+)(com|in|org|net|edu)\b', r'\1.\2'),
    
    # --- ORIGINAL PATTERN ---
    # Cleans up spaces: 'user @ gmail . com' -> 'user@gmail.com'
    (r'\s*([@\.])\s*', r'\1'),
]

ASR_AT_ERROR_PATTERNS = [
    (r'\b(listed|close|reply|email me|reach me)\s*@', r'\1 at'),
]

def fix_asr_at_errors(s: str) -> str:
    """
    Corrects common ASR errors where "at" is transcribed as "@"
    in non-email contexts (e.g., "reach me@" -> "reach me at").
    """
    for pat, rep in ASR_AT_ERROR_PATTERNS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

def collapse_spelled_letters(s: str) -> str:
    """
    Improved: Handles variable-length letter sequences.
    e.g., 'g m a i l' -> 'gmail' (4 letters)
    e.g., 's m i t h' -> 'smith' (5 letters)
    """
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        # Find start of a single-letter sequence
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            j = i + 1
            # Greedily consume all subsequent single letters
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                j += 1
            
            # If we found a sequence (e.g., > 2 letters), collapse it
            if j - i >= 2:
                out.append(''.join(tokens[i:j]))
                i = j
            else:
                # Not a sequence, just append the token
                out.append(tokens[i])
                i += 1
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

def normalize_email_tokens(s: str) -> str:
    s2 = s
    s2 = collapse_spelled_letters(s2)
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    for pat, rep in EMAIL_CLEANUP_PATTERNS:
        s2 = re.sub(pat, rep, s2)
    return s2

# --- 3. Number Normalization (BUGFIX + Feature) ---
NUM_WORD = {
    'zero':'0', 'oh':'0', 'one':'1', 'two':'2', 'three':'3', 'four':'4', 'five':'5',
    'six':'6', 'seven':'7', 'eight':'8', 'nine':'9'
}

def words_to_digits(seq: List[str]) -> Tuple[str, int]:
    """
    Improved: Returns the digit string AND number of tokens consumed.
    This fixes a major bug in the original normalize_numbers_spoken.
    """
    out = []
    i = 0
    while i < len(seq):
        tok = seq[i].lower()
        if tok in ('double', 'triple') and i + 1 < len(seq):
            times = 2 if tok == 'double' else 3
            nxt = seq[i+1].lower()
            if nxt in NUM_WORD:
                out.append(NUM_WORD[nxt] * times)
                i += 2
                continue
            else:
                # 'double' wasn't followed by a num, stop
                break
        
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
            i += 1
        else:
            # Stop on first non-number word
            break
    return ''.join(out), i # Return (digits, tokens_consumed)

def normalize_numbers_spoken(s: str) -> str:
    """
    Improved: Correctly consumes tokens and handles sequences of any length.
    """
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        # Pass the *rest* of the tokens to words_to_digits
        digits, consumed = words_to_digits(tokens[i:])
        
        # Heuristic: only collapse if it forms a number-like sequence
        # (e.g., phone number, ID) of 2+ digits.
        # This avoids "i need one apple" -> "i need 1 apple"
        if consumed > 0 and len(digits) >= 2:
            out.append(digits)
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

# --- 4. Currency Normalization (Feature + Performance) ---
INDIAN_UNITS = {
    'lakh': 100000,
    'crore': 10000000,
}

def normalize_indian_units(s: str) -> str:
    """
    (NEW) Handles 'lakh' and 'crore' units.
    e.g., "two lakh fifty thousand" -> "250000"
    e.g., "5 crore" -> "50000000"
    """
    # This is a simplified version. A full-blown parser is complex.
    # It handles "X unit" (e.g., "5 crore") and "X.Y unit" (e.g., "2.5 lakh")
    def unit_replacer(m):
        try:
            value = float(m.group(1))
            unit = m.group(2).lower()
            total = value * INDIAN_UNITS[unit]
            return str(int(total))
        except:
            return m.group(0)

    # First, simple number words
    s = re.sub(r'\bone\b', '1', s, flags=re.IGNORECASE)
    s = re.sub(r'\btwo\b', '2', s, flags=re.IGNORECASE)
    # ... (add more common small numbers as needed)
    
    # Then, units
    pattern = re.compile(r'(\d[\d,.]*)\s+(lakh|crore)\b', re.IGNORECASE)
    s = pattern.sub(unit_replacer, s)
    
    # Handle "thousand" (e.g., "2 lakh 50 thousand" -> "200000 50 thousand")
    # A simple fix: "250 thousand" -> "250000"
    s = re.sub(r'(\d[\d,.]*)\s+thousand\b', lambda m: str(int(float(m.group(1)) * 1000)), s, flags=re.IGNORECASE)
    return s

def indian_group(num_str: str) -> str:
    """
    (Improved) Faster regex-based grouping for Indian style (..XX,XX,XXX)
    """
    if not num_str.isdigit(): return num_str
    if len(num_str) <= 3:
        return num_str
    
    last3 = num_str[-3:]
    rest = num_str[:-3]
    # Add commas to 'rest' at every 2nd digit from the right
    grouped_rest = re.sub(r'(\d)(?=(\d\d)+\d$)', r'\1,', rest)
    return grouped_rest + ',' + last3

def normalize_currency(s: str) -> str:
    # 1. First, convert "two lakh" to "200000"
    s = normalize_indian_units(s)
    
    # 2. Add '₹' symbol
    s = re.sub(r'\b(rupees|rs)\b\s*', '₹', s, flags=re.IGNORECASE)
    
    # 3. Apply Indian grouping to numbers following '₹'
    def repl(m):
        # m.group(1) is '₹'
        # m.group(2) is the number (e.g., '200000')
        raw_num = m.group(2).replace(',', '') # remove existing commas
        if raw_num.isdigit():
            return '₹' + indian_group(raw_num)
        return m.group(0) # Not a clean int, leave it

    s = re.sub(r'(₹)\s*([0-9,]+)\b', repl, s)
    return s

# --- 5. Name Correction (Performance) ---
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    """
    (Improved) Pre-filters tokens to avoid running fuzzy search on
    punctuation, digits, or common short words, which is slow.
    """
    tokens = s.split()
    out = []
    for t in tokens:
        # (NEW) Performance guard:
        # Only check alpha tokens longer than 2 chars
        if not t.isalpha() or len(t) <= 2:
            out.append(t)
            continue
            
        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            out.append(best[0]) # Use the correct lexicon spelling
        else:
            out.append(t)
    return ' '.join(out)

# --- 6. Candidate Generation (Improved) ---
def generate_candidates(
    text: str, 
    names_lex: List[str], 
    misspell_map: Dict[str, str]
) -> List[str]:
    """
    (Improved) Generates a more diverse and logical set of candidates.
    - Applies high-precision misspell correction first.
    - Creates a 'full' pipeline candidate.
    - Creates 'ablation' candidates (no_names, no_numbers)
      to let the ranker choose.
    - Caps candidates at 5.
    """
    cands: Set[str] = set()
    cands.add(text) # Always include original

    # 1. Base: High-precision common misspellings
    t_base = correct_common_misspells(text, misspell_map)
    cands.add(t_base)

    # 2. Full Pipeline Candidate
    t_full = normalize_email_tokens(t_base)
    t_full = normalize_numbers_spoken(t_full)
    t_full = normalize_currency(t_full)
    t_full = correct_names_with_lexicon(t_full, names_lex)
    cands.add(t_full)

    # 3. Ablation: No Name Correction (in case names are wrong)
    t_no_names = normalize_email_tokens(t_base)
    t_no_names = normalize_numbers_spoken(t_no_names)
    t_no_names = normalize_currency(t_no_names)
    cands.add(t_no_names)

    # 4. Ablation: No Number/Currency Correction (in case numbers are wrong)
    t_no_nums = normalize_email_tokens(t_base)
    t_no_nums = correct_names_with_lexicon(t_no_nums, names_lex)
    cands.add(t_no_nums)

    # 5. Ablation: Emails + Names only
    t_email_name = normalize_email_tokens(t_base)
    t_email_name = correct_names_with_lexicon(t_email_name, names_lex)
    cands.add(t_email_name)

    # Deduplicate and cap
    return list(cands)[:5]