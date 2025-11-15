#!/usr/bin/env python3
# make_title_to_CAS_map.py
import os
import re
import pandas as pd
from difflib import get_close_matches

pred3d_csv      = r".\analysis_outputs\pred_3d_best_coral.csv"   
test_withfp_csv = r"F:\molnet_dataset_nodot_flat\dataset_scaffold_s3_test_flat.csv" 
pred3d_key      = "title"  
test_key        = "CAS"    
out_csv         = r".\analysis_outputs\title_to_CAS_map_candidate.csv"
max_suggestions = 3

CAS_RE = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")

def normalize_cas(s: str) -> str:
    """Return hyphenated CAS if present or derivable; else return input uppercased."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    # If hyphenated CAS is embedded somewhere in the string, extract it
    m = CAS_RE.search(s)
    if m:
        return m.group(1)
    # Try to format from digits only
    digits = re.sub(r"\D+", "", s)
    if len(digits) >= 3:
        body = digits[:-3] or "0"
        yy   = digits[-3:-1]
        z    = digits[-1]
        try:
            return f"{int(body)}-{yy}-{z}"
        except Exception:
            pass
    return s.upper()

def numeric_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s)) if s is not None else ""

def leading_digits(s: str) -> str:
    if s is None:
        return ""
    m = re.match(r"^(\d+)", str(s).strip())
    return m.group(1) if m else ""

def pad_row(base: list, n_pad: int) -> list:
    """Pad base list with empty strings to add n_pad suggestion columns."""
    return base + [""] * n_pad

def main():
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    P3 = pd.read_csv(pred3d_csv)
    T  = pd.read_csv(test_withfp_csv)

    if pred3d_key not in P3.columns:
        raise KeyError(f"3D file missing '{pred3d_key}'")
    if test_key not in T.columns:
        raise KeyError(f"Test file missing '{test_key}'")

    titles = P3[pred3d_key].astype(str).drop_duplicates().reset_index(drop=True)
    cas_all = T[test_key].astype(str).drop_duplicates().reset_index(drop=True)

    # Normalize CAS space for the test set
    cas_norm   = cas_all.map(normalize_cas)
    cas_digits = cas_norm.map(numeric_only)

    # Lookup sets for fast membership checks
    cas_set        = set(cas_norm)
    cas_digits_set = set(cas_digits)

    rows = []
    for title in titles:
        title_stripped = title.strip()

        m = CAS_RE.search(title_stripped)
        if m:
            cand = m.group(1)
            if cand in cas_set:
                # 4 base fields + 3 blank suggestions (or max_suggestions blanks)
                rows.append(pad_row([title, cand, "embedded_cas", 1.0], max_suggestions))
                continue

        lead = leading_digits(title_stripped)
        if lead and (lead in cas_digits_set):
            idx = cas_digits[cas_digits == lead].index
            if len(idx) > 0:
                rows.append(pad_row([title, cas_norm.loc[idx[0]], "leading_digits==cas_digits", 0.9], max_suggestions))
                continue

        t_digits = numeric_only(title_stripped)
        if t_digits and (t_digits in cas_digits_set):
            idx = cas_digits[cas_digits == t_digits].index
            if len(idx) > 0:
                rows.append(pad_row([title, cas_norm.loc[idx[0]], "title_digits==cas_digits", 0.8], max_suggestions))
                continue

        suggestions = get_close_matches(t_digits, cas_digits.tolist(), n=max_suggestions, cutoff=0.6) if t_digits else []
        sug_cas = [cas_norm.iloc[cas_digits[cas_digits == d].index[0]] for d in suggestions] if suggestions else []
        # Pad suggestions to fixed length
        sug_cas = (sug_cas + [""] * max(0, max_suggestions - len(sug_cas)))[:max_suggestions]

        # No confident match; leave CAS blank, include suggestions
        rows.append([title, "", "unmatched", 0.0, *sug_cas])

    # Column header ALWAYS has 4 + max_suggestions columns
    cols = [pred3d_key, test_key, "match_strategy", "confidence"] + [f"suggested_{i+1}" for i in range(max_suggestions)]

    # Ensure every row length matches len(cols)
    fixed_rows = []
    for r in rows:
        if len(r) < len(cols):
            fixed_rows.append(r + [""] * (len(cols) - len(r)))
        elif len(r) > len(cols):
            fixed_rows.append(r[:len(cols)])
        else:
            fixed_rows.append(r)

    out = pd.DataFrame(fixed_rows, columns=cols)
    out.to_csv(out_csv, index=False)

    # Summary
    n_unmatched = int((out[test_key] == "").sum())
    print(f"[OK] Wrote candidate map ? {out_csv}")
    print(f"[INFO] Total titles: {len(out)} | Unmatched: {n_unmatched}")

if __name__ == "__main__":
    main()
