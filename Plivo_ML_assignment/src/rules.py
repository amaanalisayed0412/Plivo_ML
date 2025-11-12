import re
from typing import List, Set, Tuple, Dict
from rapidfuzz import process, fuzz


# For correct_common_misspells (dynamic, compiled in function)


# For normalize_email_tokens
EMAIL_TOKEN_PATTERNS = [
    (re.compile(r'\b(dot)\b', re.IGNORECASE), '.'),
    (re.compile(r'\b(underscore)\b', re.IGNORECASE), '_'),
]
EMAIL_CLEANUP_PATTERNS = [
    # (BUGFIX) Handle "g mail", "y hoo", etc.
    (re.compile(r'\b(g)\s+(mail)\b', re.IGNORECASE), r'gmail'),
    (re.compile(r'\b(y)\s+(hoo)\b', re.IGNORECASE), r'yahoo'),
    
    # Fixes the (.) before surname (mehta) in the email address.
    (re.compile(r'([a-zA-Z0-9.-]+)(mehta|sharma|patel|gupta|singh|kumar|verma)\b', re.IGNORECASE), r'\1.\2'),
    
    # TLD fixes (order matters: specific 'acin'/'coin' first)
    (re.compile(r'(@[a-zA-Z0-9.-]+)(acin)\b', re.IGNORECASE), r'\1.ac.in'),
    (re.compile(r'(@[a-zA-Z0-9.-]+)(coin)\b', re.IGNORECASE), r'\1.co.in'),
    (re.compile(r'(@[a-zA-Z0-9.-]+)(com|in|org|net|edu)\b', re.IGNORECASE), r'\1.\2'),
    
    # Cleans up spaces: 'user @ gmail . com' -> 'user@gmail.com'
    (re.compile(r'\s*([@\.])\s*'), r'\1'),
]

# For normalize_indian_units
SMALL_NUM_PATTERNS = [
    (re.compile(r'\bone\b', re.IGNORECASE), '1'),
    (re.compile(r'\btwo\b', re.IGNORECASE), '2'),
    # (add more as needed)
]
INDIAN_UNIT_PATTERN = re.compile(r'(\d[\d,.]*)\s+(lakh|crore)\b', re.IGNORECASE)
THOUSAND_PATTERN = re.compile(r'(\d[\d,.]*)\s+thousand\b', re.IGNORECASE)

# For indian_group
INDIAN_GROUPING_PATTERN = re.compile(r'(\d)(?=(\d\d)+\d$)')

# For normalize_currency
CURRENCY_SYMBOL_PATTERN = re.compile(r'\b(rupees|rs)\b\s*', re.IGNORECASE)
CURRENCY_FORMAT_PATTERN = re.compile(r'(₹)\s*([0-9,]+)\b')

# --- 2. Misspellings ---
def correct_common_misspells(text: str, misspell_map: Dict[str, str]) -> str:

    single_word_keys = [k for k in misspell_map.keys() if ' ' not in k]
    if not single_word_keys:
        return text

    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in single_word_keys) + r')\b', re.IGNORECASE)
    return pattern.sub(lambda m: misspell_map[m.group(0).lower()], text)


def collapse_spelled_letters(s: str) -> str:
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                j += 1
            if j - i >= 2:
                out.append(''.join(tokens[i:j]))
                i = j
            else:
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
        s2 = pat.sub(rep, s2)
    for pat, rep in EMAIL_CLEANUP_PATTERNS:
        s2 = pat.sub(rep, s2)
    return s2

# --- 4. Number Normalization ---
NUM_WORD = {
    'zero':'0', 'oh':'0', 'one':'1', 'two':'2', 'three':'3', 'four':'4', 'five':'5',
    'six':'6', 'seven':'7', 'eight':'8', 'nine':'9'
}

def words_to_digits(seq: List[str]) -> Tuple[str, int]:
    # (Your function is good, keep as-is)
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
                break
        if tok in NUM_WORD:
            out.append(NUM_WORD[tok])
            i += 1
        else:
            break
    return ''.join(out), i

def normalize_numbers_spoken(s: str) -> str:
    # (Your function is good, keep as-is)
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        digits, consumed = words_to_digits(tokens[i:])
        if consumed > 0 and len(digits) >= 2:
            out.append(digits)
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)

INDIAN_UNITS = {'lakh': 100000, 'crore': 10000000}

def normalize_indian_units(s: str) -> str:
    def unit_replacer(m):
        try:
            value = float(m.group(1))
            unit = m.group(2).lower()
            total = value * INDIAN_UNITS[unit]
            return str(int(total))
        except:
            return m.group(0)

    for pat, rep in SMALL_NUM_PATTERNS:
        s = pat.sub(rep, s)
    
    s = INDIAN_UNIT_PATTERN.sub(unit_replacer, s)
    s = THOUSAND_PATTERN.sub(lambda m: str(int(float(m.group(1)) * 1000)), s)
    return s

def indian_group(num_str: str) -> str:
    if not num_str.isdigit() or len(num_str) <= 3:
        return num_str
    
    last3 = num_str[-3:]
    rest = num_str[:-3]
    grouped_rest = INDIAN_GROUPING_PATTERN.sub(r'\1,', rest)
    return grouped_rest + ',' + last3

def normalize_currency(s: str) -> str:
    s = normalize_indian_units(s)
    s = CURRENCY_SYMBOL_PATTERN.sub('₹', s)
    
    def repl(m):
        raw_num = m.group(2).replace(',', '')
        if raw_num.isdigit():
            return '₹' + indian_group(raw_num)
        return m.group(0)

    s = CURRENCY_FORMAT_PATTERN.sub(repl, s)
    return s

# --- 5. Name Correction (Performance) ---
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    tokens = s.split()
    out = []
    for t in tokens:
        # (LATENCY FIX) Strip punctuation *before* the check
        clean_t = t.strip('.,?!')
        
        # Guard condition to skip slow fuzzy search
        if not clean_t.isalpha() or len(clean_t) <= 2:
            out.append(t)
            continue
            
        best = process.extractOne(clean_t, names_lex, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            # Re-apply punctuation by replacing the clean part
            out.append(t.replace(clean_t, best[0]))
        else:
            out.append(t)
    return ' '.join(out)

# --- 6. Candidate Generation (Improved) ---
def generate_candidates(
    text: str, 
    names_lex: List[str], 
    misspell_map: Dict[str, str]
) -> List[str]:
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

    # 3. Ablation: No Name Correction
    t_no_names = normalize_email_tokens(t_base)
    t_no_names = normalize_numbers_spoken(t_no_names)
    t_no_names = normalize_currency(t_no_names)
    cands.add(t_no_names)

    # 4. Ablation: No Number/Currency Correction
    t_no_nums = normalize_email_tokens(t_base)
    t_no_nums = correct_names_with_lexicon(t_no_nums, names_lex)
    cands.add(t_no_nums)

    # 5. Ablation: Emails + Names only
    t_email_name = normalize_email_tokens(t_base)
    t_email_name = correct_names_with_lexicon(t_email_name, names_lex)
    cands.add(t_email_name)

    # Deduplicate and cap
    return list(cands)[:5]