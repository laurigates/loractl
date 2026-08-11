#!/usr/bin/env bash
# collate-reception.sh — mechanical rollup of reception-panel persona reports.
#
# One agent reading eight reports averages away the disagreement, and the
# disagreement is the payload. This script does not synthesize: it validates,
# drops, tallies, and prints. Every judgment it applies is a rule stated here,
# not an opinion formed at read time.
#
# Usage:  scripts/collate-reception.sh docs/reception/<date>
# Input:  <dir>/raw/*.txt        persona reports in the fixed schema
#         <dir>/partition.tsv    optional: FINDING_ID<TAB>KNOWN:Pn|NEW|WRONG:Pn
# Output: rollup on stdout; <dir>/findings.tsv written for the partition pass.
#
# Enforcement rules (applied here, never by an agent):
#   * a FINDING with no file:line and no quoted string is DROPPED
#   * a persona whose drop rate exceeds 50% is DISCARDED ENTIRELY
#     (instrument failure, not a finding)
#   * BOUNCE_POINT=NONE from a tier-0 persona is flagged as a sycophancy signal
set -euo pipefail
export LC_ALL=C

DIR="${1:?usage: collate-reception.sh <docs/reception/DATE>}"
[ -d "$DIR/raw" ] || { echo "collate: no $DIR/raw" >&2; exit 2; }

python3 - "$DIR" <<'PYEOF'
import os
import re
import sys

DIR = sys.argv[1]
RAW = os.path.join(DIR, "raw")

DROP_RATE_DISCARD = 0.50
EVIDENCE_OK = re.compile(r"[\w./-]+\.(?:rs|md|toml|yaml|yml|txt|json|sh|py):\d+|\"[^\"]{3,}\"|'[^']{3,}'")


def parse(path):
    rec = {"_file": os.path.basename(path), "findings": [], "unverifiable": []}
    for raw_line in open(path, encoding="utf-8", errors="replace"):
        line = raw_line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("FINDING|"):
            # The claim field legitimately contains `|` — a README enum such as
            # `backend: ndarray | wgpu | cuda` is exactly the kind of thing a
            # finding quotes. Anchor on the first three fields and the last one
            # and rejoin the middle, rather than truncating the claim at its
            # first pipe.
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                rec["findings"].append({
                    "severity": parts[1].upper(),
                    "evidence": parts[2],
                    "claim": " | ".join(parts[3:-1]),
                    "falsifier": parts[-1],
                })
            continue
        if line.startswith("UNVERIFIABLE|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                rec["unverifiable"].append((parts[1], parts[2]))
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            rec[k.strip()] = v.strip()
    return rec


reports = [parse(os.path.join(RAW, f))
           for f in sorted(os.listdir(RAW))
           if f.endswith(".txt") and not f.startswith("slop-metrics")]

if not reports:
    print("collate: no persona reports in", RAW)
    sys.exit(0)

# ------------------------------------------------------------------ validation
kept, discarded = [], []
for r in reports:
    good, dropped = [], []
    for f in r["findings"]:
        (good if EVIDENCE_OK.search(f["evidence"]) else dropped).append(f)
    total = len(good) + len(dropped)
    rate = (len(dropped) / total) if total else 0.0
    r["_good"], r["_dropped"], r["_rate"] = good, dropped, rate
    (discarded if total and rate > DROP_RATE_DISCARD else kept).append(r)

W = "=" * 72
print(W)
print("RECEPTION PANEL — MECHANICAL ROLLUP")
print(W)
print(f"personas reporting : {len(reports)}")
print(f"personas kept      : {len(kept)}")
print(f"personas discarded : {len(discarded)}"
      + (f"  ({', '.join(d.get('PERSONA', d['_file']) for d in discarded)})" if discarded else ""))

print("\n-- instrument health -----------------------------------------------")
for r in reports:
    name = r.get("PERSONA", r["_file"])
    status = "DISCARDED" if r in discarded else "kept"
    print(f"  {name:<22} tier={r.get('TIER','?'):<3} model={r.get('MODEL','?'):<26}"
          f" findings={len(r['_good'])} dropped={len(r['_dropped'])}"
          f" drop_rate={r['_rate']*100:.0f}%  [{status}]")
    for d in r["_dropped"]:
        print(f"      DROPPED (no file:line or quote): {d['claim'][:60]}")

# --------------------------------------------------------------- bounce points
print("\n-- BOUNCE_POINT (where the reader first considered leaving) --------")
for r in kept:
    bp = r.get("BOUNCE_POINT", "MISSING")
    flag = ""
    if bp.upper() == "NONE" and str(r.get("TIER", "")).startswith("0"):
        flag = "   <<< SYCOPHANCY SIGNAL: no bounce on a 480-line README"
    print(f"  {r.get('PERSONA','?'):<22} {bp}{flag}")
    q = r.get("BOUNCE_QUOTE", "")
    if q:
        print(f"      quote : {q}")
    if r.get("BOUNCE_REASON"):
        print(f"      reason: {r['BOUNCE_REASON']}")

# ----------------------------------------------------------------- worst thing
print("\n-- WORST_THING (one per persona, mandatory) ------------------------")
for r in kept:
    print(f"  {r.get('PERSONA','?'):<22} {r.get('WORST_THING_EVIDENCE','?')}")
    print(f"      {r.get('WORST_THING_DEFECT','(missing)')}")

# ------------------------------------------------------------------ action tally
print("\n-- ACTION TALLY ----------------------------------------------------")
actions = ["WOULD_CLONE", "WOULD_RUN", "WOULD_STAR", "WOULD_FILE_ISSUE", "WOULD_RECOMMEND"]
print(f"  {'persona':<22}" + "".join(a.replace('WOULD_', '')[:9].rjust(11) for a in actions))
for r in kept:
    cells = []
    for a in actions:
        v = r.get(a, "?").split("|")[0].strip().upper()[:3]
        cells.append(v.rjust(11))
    print(f"  {r.get('PERSONA','?'):<22}" + "".join(cells))
for a in actions:
    yes = sum(1 for r in kept if r.get(a, "").split("|")[0].strip().upper().startswith("Y"))
    print(f"  {a:<22} YES={yes}/{len(kept)}")

# --------------------------------------------------------------- prior → delta
print("\n-- PRIOR vs DELTA (did reading change the verdict?) ----------------")
for r in kept:
    print(f"  {r.get('PERSONA','?')}")
    print(f"      expect: {r.get('EXPECT','-')}")
    print(f"      doubt : {r.get('EXPECT_DOUBT','-')}")
    print(f"      delta : {r.get('DELTA','-')}")

# -------------------------------------------------------------------- findings
findings = []
for r in kept:
    for f in r["_good"]:
        findings.append((r.get("PERSONA", "?"), f))

order = {"BLOCKER": 0, "HIGH": 1, "MEDIUM": 2, "MED": 2, "LOW": 3}
findings.sort(key=lambda t: (order.get(t[1]["severity"], 9), t[0]))

tsv = os.path.join(DIR, "findings.tsv")
with open(tsv, "w", encoding="utf-8") as fh:
    fh.write("id\tpersona\tseverity\tevidence\tclaim\tfalsifier\n")
    for i, (persona, f) in enumerate(findings, 1):
        fh.write(f"F{i:02d}\t{persona}\t{f['severity']}\t{f['evidence']}\t{f['claim']}\t{f['falsifier']}\n")

print(f"\n-- FINDINGS ({len(findings)} surviving validation) ---------------------------")
for i, (persona, f) in enumerate(findings, 1):
    print(f"  F{i:02d} [{f['severity']}] ({persona}) {f['evidence']}")
    print(f"      claim    : {f['claim']}")
    print(f"      falsifier: {f['falsifier']}")

# ------------------------------------------------------- convergence/divergence
print("\n-- CONVERGENCE (same evidence cited by 2+ personas) ----------------")
byev = {}
for persona, f in findings:
    key = f["evidence"].split(",")[0].strip()
    byev.setdefault(key, []).append(persona)
multi = {k: v for k, v in byev.items() if len(set(v)) > 1}
if multi:
    for k in sorted(multi):
        print(f"  {k}  <- {', '.join(sorted(set(multi[k])))}")
else:
    print("  (none — every finding is single-source; treat all as unconfirmed)")

# ---------------------------------------------------------------- unverifiable
print("\n-- UNVERIFIABLE (artifact the repo does not currently supply) ------")
seen = set()
for r in kept:
    for claim, artifact in r["unverifiable"]:
        key = artifact.lower()
        if key in seen:
            continue
        seen.add(key)
        print(f"  need: {artifact}")
        print(f"        to settle: {claim}")

# ------------------------------------------------------------------- partition
part_path = os.path.join(DIR, "partition.tsv")
print("\n-- KNOWN / NEW / WRONG vs predictions.md ---------------------------")
if not os.path.exists(part_path):
    print(f"  partition.tsv absent — assign each id in {tsv} then re-run.")
    sys.exit(0)

part = {}
for line in open(part_path, encoding="utf-8"):
    if not line.strip() or line.startswith("#") or line.startswith("id\t"):
        continue
    cols = line.rstrip("\n").split("\t")
    if len(cols) >= 2:
        part[cols[0].strip()] = cols[1].strip()

buckets = {"KNOWN": [], "NEW": [], "WRONG": []}
for i, (persona, f) in enumerate(findings, 1):
    fid = f"F{i:02d}"
    label = part.get(fid, "UNASSIGNED")
    buckets.setdefault(label.split(":")[0], []).append((fid, label, persona, f))

for b in ("KNOWN", "NEW", "WRONG", "UNASSIGNED"):
    items = buckets.get(b, [])
    if not items:
        continue
    print(f"\n  [{b}] {len(items)}")
    for fid, label, persona, f in items:
        tag = f" ({label.split(':',1)[1]})" if ":" in label else ""
        print(f"    {fid}{tag} [{f['severity']}] {f['claim']}")

new_n = len(buckets.get("NEW", []))
print("\n" + W)
print(f"MARGINAL INFORMATION: NEW={new_n}  KNOWN={len(buckets.get('KNOWN', []))}"
      f"  WRONG={len(buckets.get('WRONG', []))}")
if new_n >= 3:
    print("VERDICT: the panel EARNED ITS COST (>=3 findings absent from predictions.md).")
elif new_n == 0:
    print("VERDICT: the panel added NOTHING. Do not run it on the next project.")
else:
    print(f"VERDICT: MARGINAL — only {new_n} new finding(s). Cheap lane only next time.")
print(W)
PYEOF
