"""
Validation script for the Financial Table Alignment Engine.

Tests the two real examples from the Attijari bank documents:
  - Table 1: Etat des engagements hors bilan (31.12.2024 / 31.12.2023)
  - Table 2: Bilan Consolide (31/12/2024 / 31/12/2023)

Also tests edge-cases:
  - Note code leaked into a date column (classic OCR misalignment)
  - Negative value in parentheses: (317 205) -> -317205
  - Percentage in parentheses: (56,4%) -> -56.4%
  - NUMBER in Variation % column (must be auto-corrected)
  - Schema with 3 date columns
  - Section/total rows with no numeric values
"""

import json
import sys
from table_alignment_engine import (
    run_alignment_engine,
    classify_value,
    detect_schema_from_columns,
    validate_row,
    VALUE_TYPE_TEXT, VALUE_TYPE_NOTE, VALUE_TYPE_PERCENT,
    VALUE_TYPE_NUMBER, VALUE_TYPE_EMPTY,
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SEP  = "=" * 65

errors = []

def check(desc, actual, expected):
    ok = actual == expected
    marker = PASS if ok else FAIL
    print(f"  {marker} {desc}")
    if not ok:
        print(f"         expected: {expected!r}")
        print(f"         got:      {actual!r}")
        errors.append(desc)

# =============================================================================
# B. VALUE CLASSIFICATION
# =============================================================================
print(f"\n{SEP}")
print("  B. VALUE CLASSIFICATION LAYER")
print(SEP)

cases = [
    # (input,           expected_type,         expected_norm)
    ("",                VALUE_TYPE_EMPTY,       ""),
    ("-",               VALUE_TYPE_EMPTY,       ""),
    ("N/A",             VALUE_TYPE_EMPTY,       ""),
    ("(4.1)",           VALUE_TYPE_NOTE,        "(4.1)"),
    ("(1-2)",           VALUE_TYPE_NOTE,        "(1-2)"),
    ("(III.1)",         VALUE_TYPE_NOTE,        "(III.1)"),
    ("(3)",             VALUE_TYPE_NOTE,        "(3)"),
    ("1-2",             VALUE_TYPE_NOTE,        "1-2"),
    ("II.3",            VALUE_TYPE_NOTE,        "II.3"),
    ("22,5%",           VALUE_TYPE_PERCENT,     "22.5"),
    ("(5,4%)",          VALUE_TYPE_PERCENT,     "-5.4"),
    ("(30,9%)",         VALUE_TYPE_PERCENT,     "-30.9"),
    ("0.0%",            VALUE_TYPE_PERCENT,     "0.0"),
    ("-15,4 %",         VALUE_TYPE_PERCENT,     "-15.4"),
    ("4,4%",            VALUE_TYPE_PERCENT,     "4.4"),
    ("147 120",         VALUE_TYPE_NUMBER,      "147120"),
    ("799 892",         VALUE_TYPE_NUMBER,      "799892"),
    ("(317 205)",       VALUE_TYPE_NUMBER,      "-317205"),
    ("<123 456>",       VALUE_TYPE_NUMBER,      "-123456"),
    ("1 730 727",       VALUE_TYPE_NUMBER,      "1730727"),
    ("0",               VALUE_TYPE_NUMBER,      "0"),
    ("-97",             VALUE_TYPE_NUMBER,      "-97"),
    ("12 434 225",      VALUE_TYPE_NUMBER,      "12434225"),
    ("Caisse et avoirs",VALUE_TYPE_TEXT,        "Caisse et avoirs"),
    ("TOTAL ACTIF",     VALUE_TYPE_TEXT,        "TOTAL ACTIF"),
]

for val, exp_type, exp_norm in cases:
    vtype, vnorm = classify_value(val)
    check(f"classify_value({val!r}) -> type", vtype, exp_type)
    check(f"classify_value({val!r}) -> norm", vnorm, exp_norm)

# =============================================================================
# A. SCHEMA DETECTION
# =============================================================================
print(f"\n{SEP}")
print("  A. SCHEMA DETECTION LAYER")
print(SEP)

s1 = detect_schema_from_columns([
    "Label", "Note", "31.12.2024", "31.12.2023",
    "Variation Montant", "Variation %"
])
check("Table-1 label_col",              s1.label_col,            "Label")
check("Table-1 note_col",               s1.note_col,             "Note")
check("Table-1 date_cols",              s1.date_cols,            ["31.12.2024", "31.12.2023"])
check("Table-1 variation_amount_col",   s1.variation_amount_col, "Variation Montant")
check("Table-1 variation_percent_col",  s1.variation_percent_col,"Variation %")
check("Table-1 unresolved",             s1.unresolved_cols,      [])

s2 = detect_schema_from_columns([
    "Label", "Note", "31/12/2024", "31/12/2023",
    "Variation Montant", "Variation %"
])
check("Table-2 label_col",              s2.label_col,            "Label")
check("Table-2 date_cols",              s2.date_cols,            ["31/12/2024", "31/12/2023"])
check("Table-2 variation_percent_col",  s2.variation_percent_col,"Variation %")

s3 = detect_schema_from_columns([
    "Libellé", "Notes", "30/06/2022", "30/06/2021", "31/12/2021",
    "En Montant", "En %"
])
check("3-date label_col",               s3.label_col,            "Libellé")
check("3-date note_col",                s3.note_col,             "Notes")
check("3-date date_cols count",         len(s3.date_cols),       3)
check("3-date variation_amount_col",    s3.variation_amount_col, "En Montant")
check("3-date variation_percent_col",   s3.variation_percent_col,"En %")

# =============================================================================
# C. ALIGNMENT ENGINE — Table 1 misalignment
# =============================================================================
print(f"\n{SEP}")
print("  C. ALIGNMENT ENGINE — Table 1 (note in date column)")
print(SEP)

# Classic OCR error: (4.1) ended up in "31.12.2024", values shifted right
misaligned_table1 = {
    "columns": ["Label", "Note", "31.12.2024", "31.12.2023",
                "Variation Montant", "Variation %"],
    "rows": [
        {
            "type": "data",
            "Label": "HB1 - Cautions, avals et autres garanties données",
            "Note": "",
            "31.12.2024": "(4.1)",      # <-- NOTE in wrong column
            "31.12.2023": "799 892",    # shifted
            "Variation Montant": "652 772",  # shifted
            "Variation %": "147 120",   # NUMBER in % column!
        },
        {
            "type": "data",
            "Label": "HB2 - Crédits documentaires",
            "Note": "(4.2)",
            "31.12.2024": "210 424",
            "31.12.2023": "206 353",
            "Variation Montant": "4 071",
            "Variation %": "2,0%",
        },
        {
            "type": "total",
            "Label": "Total des passifs éventuels",
            "Note": "",
            "31.12.2024": "1 010 316",
            "31.12.2023": "859 125",
            "Variation Montant": "151 191",
            "Variation %": "17,6%",
        },
        {
            "type": "section",
            "Label": "Engagements reçus",
            "Note": "",
            "31.12.2024": "",
            "31.12.2023": "",
            "Variation Montant": "",
            "Variation %": "",
        },
    ],
}

result1 = run_alignment_engine(misaligned_table1)
rows1 = result1["rows"]
schema1 = result1.get("_schema", {})

check("T1 schema dates",          schema1.get("dates"),             ["31.12.2024", "31.12.2023"])
check("T1 schema variation_pct",  schema1.get("variation_percent"), "Variation %")

r0 = rows1[0]
# (4.1) was in date column → engine rescues it to Note
check("T1 row0 Note (4.1) rescued",    r0.get("Note"),             "(4.1)")
# Dates shift back correctly
check("T1 row0 date1 = 799892",        r0.get("31.12.2024"),        "799892")
check("T1 row0 date2 = 652772",        r0.get("31.12.2023"),        "652772")
# "147 120" was a NUMBER in the Variation % slot → no % sign → goes to variation_amount
# (% value 22.5% is unrecoverable - it was never in the VLM output)
check("T1 row0 var_montant = 147120",  r0.get("Variation Montant"), "147120")
check("T1 row0 var_pct = '' (lost)",   r0.get("Variation %"),       "")


r1 = rows1[1]
check("T1 row1 note (4.2)",           r1.get("Note"),              "(4.2)")
check("T1 row1 date1 = 210424",       r1.get("31.12.2024"),        "210424")
check("T1 row1 var_pct = 2.0%",       r1.get("Variation %"),       "2.0%")

r2 = rows1[2]
check("T1 total date1 = 1010316",     r2.get("31.12.2024"),        "1010316")
check("T1 total var_pct = 17.6%",     r2.get("Variation %"),       "17.6%")

r3 = rows1[3]
check("T1 section type",              r3.get("type"),              "section")
check("T1 section date1 empty",       r3.get("31.12.2024"),        "")

# =============================================================================
# C. ALIGNMENT ENGINE — Table 2 (negative values, 3-date schema)
# =============================================================================
print(f"\n{SEP}")
print("  C. ALIGNMENT ENGINE — Table 2 (Bilan Consolidé)")
print(SEP)

table2 = {
    "columns": ["Label", "Note", "31/12/2024", "31/12/2023",
                "Variation Montant", "Variation %"],
    "rows": [
        {
            "type": "data",
            "Label": "Créances sur les établissements bancaires et financiers *",
            "Note": "(1-2)",
            "31/12/2024": "244 740",
            "31/12/2023": "561 945",
            "Variation Montant": "(317 205)",
            "Variation %": "(56,4%)",
        },
        {
            "type": "data",
            "Label": "Goodwill",
            "Note": "",
            "31/12/2024": "216",
            "31/12/2023": "313",
            "Variation Montant": "(97)",
            "Variation %": "(30,9%)",
        },
        {
            "type": "total",
            "Label": "TOTAL ACTIF",
            "Note": "",
            "31/12/2024": "12 434 225",
            "31/12/2023": "11 912 261",
            "Variation Montant": "521 965",
            "Variation %": "4,4%",
        },
    ],
}

result2 = run_alignment_engine(table2)
rows2 = result2["rows"]
schema2 = result2.get("_schema", {})

check("T2 schema dates", schema2.get("dates"), ["31/12/2024", "31/12/2023"])

r0b = rows2[0]
check("T2 row0 note (1-2)",            r0b.get("Note"),             "(1-2)")
check("T2 row0 date1 = 244740",        r0b.get("31/12/2024"),       "244740")
check("T2 row0 date2 = 561945",        r0b.get("31/12/2023"),       "561945")
check("T2 row0 var_montant = -317205", r0b.get("Variation Montant"),"-317205")
check("T2 row0 var_pct = -56.4%",      r0b.get("Variation %"),      "-56.4%")

r1b = rows2[1]
check("T2 goodwill var_montant = -97", r1b.get("Variation Montant"),"-97")
check("T2 goodwill var_pct = -30.9%",  r1b.get("Variation %"),      "-30.9%")

r2b = rows2[2]
check("T2 total date1 = 12434225",     r2b.get("31/12/2024"),       "12434225")
check("T2 total var_pct = 4.4%",       r2b.get("Variation %"),      "4.4%")

# =============================================================================
# D. VALIDATION LAYER
# =============================================================================
print(f"\n{SEP}")
print("  D. VALIDATION LAYER")
print(SEP)

from table_alignment_engine import detect_schema_from_columns, ColumnSchema

schema_v = detect_schema_from_columns(
    ["Label", "Note", "31/12/2024", "31/12/2023",
     "Variation Montant", "Variation %"]
)

# Valid row: no errors
valid_row = {
    "Label": "Caisse", "Note": "(1-1)",
    "31/12/2024": "1730727", "31/12/2023": "1542910",
    "Variation Montant": "187817", "Variation %": "12.2%",
}
errs = validate_row(valid_row, schema_v)
check("Valid row -> 0 errors", len(errs), 0)

# % in date column
bad_row_pct_in_date = {
    "Label": "X", "Note": "",
    "31/12/2024": "22.5%",   # <-- WRONG
    "31/12/2023": "100",
    "Variation Montant": "50", "Variation %": "",
}
errs2 = validate_row(bad_row_pct_in_date, schema_v)
check("% in date column -> error detected", len(errs2) >= 1, True)

# Note in date column
bad_row_note_in_date = {
    "Label": "Y", "Note": "",
    "31/12/2024": "(1-1)",   # <-- WRONG
    "31/12/2023": "100",
    "Variation Montant": "50", "Variation %": "5.0%",
}
errs3 = validate_row(bad_row_note_in_date, schema_v)
check("Note in date column -> error detected", len(errs3) >= 1, True)

# Number in Variation % column
bad_row_num_in_pct = {
    "Label": "Z", "Note": "",
    "31/12/2024": "100", "31/12/2023": "90",
    "Variation Montant": "10", "Variation %": "10",  # <-- WRONG (no % sign)
}
errs4 = validate_row(bad_row_num_in_pct, schema_v)
check("NUMBER in Variation % -> error detected", len(errs4) >= 1, True)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{SEP}")
total = len(cases) * 2 + 52   # approximate total checks
if errors:
    print(f"  RESULT: {len(errors)} FAILURE(S)")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(f"  ALL CHECKS PASSED ✓")
print(SEP)
