import json

BEFORE = "../atc_synonyms_meta.json"
AFTER  = "atc_synonyms_meta_cleaned.json"

REPORT = "synonym_diff_report.txt"

# ===== LOAD =====
with open(BEFORE, "r", encoding="utf-8") as f:
    meta_before = json.load(f)

with open(AFTER, "r", encoding="utf-8") as f:
    meta_after = json.load(f)

removed_global = 0
added_global = 0

lines = []
lines.append("=== SYNONYM DIFF REPORT ===\n")

for code in meta_before:
    before_syn = set(meta_before[code].get("synonyms", []))
    after_syn  = set(meta_after.get(code, {}).get("synonyms", []))

    removed = sorted(list(before_syn - after_syn))
    added   = sorted(list(after_syn - before_syn))

    if not removed and not added:
        continue

    lines.append(f"\n--------------------------------------------")
    lines.append(f"ATC: {code}")
    lines.append(f"--------------------------------------------")

    # Removed synonyms
    if removed:
        removed_global += len(removed)
        lines.append("  Removed:")
        for s in removed:
            lines.append(f"    - {s}")
    else:
        lines.append("  Removed: (none)")

    # Added synonyms
    if added:
        added_global += len(added)
        lines.append("  Added:")
        for s in added:
            lines.append(f"    + {s}")
    else:
        lines.append("  Added: (none)")

lines.append("\n==============================================")
lines.append(f"Total removed synonyms: {removed_global}")
lines.append(f"Total added synonyms:   {added_global}")
lines.append("==============================================")

# Save report
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Done! Diff report generated →", REPORT)
