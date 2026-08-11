#!/usr/bin/env bash
# slop-metrics.sh — deterministic code-discipline metrics for a repo.
#
# Answers the only part of "is this slop?" a machine can answer: measurable
# test discipline, error-handling discipline, and documentation ratios. It
# takes no opinion — an absolute number here means nothing. Run it against
# known-good comparables and read the delta.
#
# Usage:   scripts/slop-metrics.sh [REPO_PATH]      (default: repo containing this script)
# Output:  KEY=VALUE lines on stdout, one per metric. Nothing else.
#
# Determinism contract: two runs over the same tree emit byte-identical output.
# No timestamps, no `date`, no wall-clock, no network, no locale-sensitive sort.
set -euo pipefail
export LC_ALL=C

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -d "$REPO" ]; then
  echo "slop-metrics: not a directory: $REPO" >&2
  exit 2
fi
REPO="$(cd "$REPO" && pwd)"

python3 - "$REPO" <<'PYEOF'
import os
import re
import subprocess
import sys

REPO = sys.argv[1]

# Directories that are never the author's own code.
SKIP_DIRS = {
    ".git", "target", "node_modules", "venv", ".venv", "__pycache__",
    "vendor", "third_party", "dist", "build", ".tox", ".mypy_cache",
    "site-packages", ".eggs", "docs/_build",
}

RUST_EXT = ".rs"
PY_EXT = ".py"
DOC_EXT = (".md", ".rst", ".adoc")


def _tracked():
    """Tracked files, or None when this is not a git checkout.

    git ls-files is the only discovery that reliably excludes gitignored trees.
    An os.walk denylist cannot: loractl carries stale agent worktrees under
    .claude/worktrees/ that are full copies of the repo, and counting them
    inflated every metric ~4x. Tracked-files-only is also the right definition
    of "code the author wrote" for every comparable.
    """
    try:
        out = subprocess.run(
            ["git", "-C", REPO, "ls-files", "-z"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return sorted(p for p in out.stdout.split("\0") if p)


_TRACKED = _tracked()


def walk(root, exts):
    if _TRACKED is not None:
        return [
            os.path.join(root, p) for p in _TRACKED
            if p.endswith(exts) and not any(part in SKIP_DIRS for part in p.split("/"))
        ]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".git"))
        for fn in sorted(filenames):
            if fn.endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def nonblank(text):
    return sum(1 for ln in text.splitlines() if ln.strip())


# ---------------------------------------------------------------- language pick
rust_files = walk(REPO, (RUST_EXT,))
py_files = walk(REPO, (PY_EXT,))
rust_loc = sum(nonblank(read(f)) for f in rust_files)
py_loc = sum(nonblank(read(f)) for f in py_files)

if rust_loc >= py_loc:
    lang, src_files, src_loc = "rust", rust_files, rust_loc
else:
    lang, src_files, src_loc = "python", py_files, py_loc

# ------------------------------------------------------------ test-fn extraction
# Rust: an attribute line marking a test, then the next `fn`, then brace-match
# the body. Python: a `def test_*` and its indented block.

RUST_TEST_ATTR = re.compile(r"^\s*#\[(?:\w+::)*(?:test|tokio::test|rstest|test_case)\b")
RUST_FN = re.compile(r"\bfn\s+\w+")

# Two assertion shapes, both real. The macro form is the obvious one; the
# method form (`.assert_approx_eq::<f32>(&other, tol)`) is how burn, insta and
# approx assert on tensors and snapshots. Counting only macros under-reports
# numerics-heavy suites badly — it scored loractl at 9% zero-assertion tests
# when the "empty" tests were in fact the tensor-equality proofs.
ASSERT_RS = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|assert_matches|debug_assert|debug_assert_eq"
    r"|debug_assert_ne|panic|unreachable|assert_snapshot|assert_json_eq)\s*!"
    r"|\.\s*assert\w*\s*(?:::<[^>]*>)?\s*\("
)
# A #[should_panic] test asserts via its attribute and has no assertion in body.
SHOULD_PANIC_RS = re.compile(r"#\[should_panic")
# A "trivial" assertion: the whole thing only checks a Result/Option discriminant
# without inspecting the value. `assert!(x.is_ok())`, `assert!(r.is_some());`
TRIVIAL_RS = re.compile(
    r"^assert\s*!\s*\(\s*[^,;]*\.\s*is_(?:ok|err|some|none)\s*\(\s*\)\s*,?\s*\)?\s*;?\s*$"
)

ASSERT_PY = re.compile(r"(?:^|\s)(?:assert\s|self\.assert\w+\s*\(|pytest\.raises\s*\()")
TRIVIAL_PY = re.compile(r"^assert\s+\w+(?:\.\w+)*\s*(?:is\s+not\s+None|is\s+None)?\s*$")


def rust_test_bodies(text):
    """Yield the source of each #[test] function body."""
    lines = text.splitlines()
    bodies = []
    i = 0
    n = len(lines)
    while i < n:
        if RUST_TEST_ATTR.match(lines[i]):
            # scan forward for the fn signature, then the opening brace
            j = i + 1
            while j < n and not RUST_FN.search(lines[j]):
                # bail if we hit another item before finding a fn
                if lines[j].strip().startswith("}"):
                    break
                j += 1
            if j >= n or not RUST_FN.search(lines[j]):
                i += 1
                continue
            # brace-match from the first `{` at or after j
            depth = 0
            started = False
            body = []
            k = j
            while k < n:
                ln = lines[k]
                for ch in ln:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                if started:
                    body.append(ln)
                    if depth <= 0:
                        break
                k += 1
            # keep the attribute lines so #[should_panic] is visible to the counter
            bodies.append("\n".join(lines[i:j] + body))
            i = k + 1
            continue
        i += 1
    return bodies


def py_test_bodies(text):
    lines = text.splitlines()
    bodies = []
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^(\s*)(?:async\s+)?def\s+test_\w*\s*\(", lines[i])
        if m:
            indent = len(m.group(1))
            body = []
            k = i + 1
            while k < n:
                ln = lines[k]
                if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
                    break
                body.append(ln)
                k += 1
            bodies.append("\n".join(body))
            i = k
            continue
        i += 1
    return bodies


def count_asserts(body, lang):
    if lang == "rust":
        stmts = [s.strip() for s in re.split(r";\s*\n|\n", body)]
        total = len(ASSERT_RS.findall(body))
        if SHOULD_PANIC_RS.search(body):
            total += 1
        trivial = sum(1 for s in stmts if TRIVIAL_RS.match(s.strip()))
        return total, trivial
    stmts = [s.strip() for s in body.splitlines()]
    total = sum(1 for s in stmts if ASSERT_PY.search(" " + s))
    trivial = sum(1 for s in stmts if TRIVIAL_PY.match(s))
    return total, trivial


test_fns = 0
assert_total = 0
tests_with_zero = 0
tests_trivial_only = 0

for f in src_files:
    text = read(f)
    bodies = rust_test_bodies(text) if lang == "rust" else py_test_bodies(text)
    for b in bodies:
        test_fns += 1
        total, trivial = count_asserts(b, lang)
        assert_total += total
        if total == 0:
            tests_with_zero += 1
        elif trivial == total:
            tests_trivial_only += 1

# --------------------------------------------------------- error-handling density
# Split library code from test code: an .unwrap() in a test is a deliberate
# fail-loud, the same call in library code is a panic path a user can reach.
# File-level split only — a #[cfg(test)] mod inside a src file counts as
# library code here, so the NONTEST numbers are an upper bound.
def is_test_file(path):
    rel = os.path.relpath(path, REPO).replace(os.sep, "/")
    parts = rel.split("/")
    base = parts[-1]
    return (
        "tests" in parts or "benches" in parts or "test" in parts
        or base.startswith("test_") or base.endswith("_test.py")
        or base.endswith("_tests.rs") or base == "conftest.py"
    )


nontest_files = [f for f in src_files if not is_test_file(f)]
nontest_loc = sum(nonblank(read(f)) for f in nontest_files)
all_nontest = "\n".join(read(f) for f in nontest_files)

all_src = "\n".join(read(f) for f in src_files)
if lang == "rust":
    unwraps = len(re.findall(r"\.unwrap\s*\(\s*\)", all_src))
    expects = len(re.findall(r"\.expect\s*\(", all_src))
    panics = len(re.findall(r"\bpanic\s*!\s*\(", all_src))
    nt_unwraps = len(re.findall(r"\.unwrap\s*\(\s*\)", all_nontest))
    nt_expects = len(re.findall(r"\.expect\s*\(", all_nontest))
    nt_panics = len(re.findall(r"\bpanic\s*!\s*\(", all_nontest))
    todos = len(re.findall(r"\b(?:todo|unimplemented)\s*!", all_src))
    suppressions = len(re.findall(r"#!?\[allow\(", all_src))
    swallow = len(re.findall(r"\blet\s+_\s*=\s*", all_src))
else:
    unwraps = len(re.findall(r"except\s*:", all_src))
    expects = len(re.findall(r"except\s+Exception", all_src))
    panics = len(re.findall(r"\braise\s+\w+", all_src))
    nt_unwraps = len(re.findall(r"except\s*:", all_nontest))
    nt_expects = len(re.findall(r"except\s+Exception", all_nontest))
    nt_panics = len(re.findall(r"\braise\s+\w+", all_nontest))
    todos = len(re.findall(r"\b(?:TODO|FIXME|XXX)\b", all_src))
    suppressions = len(re.findall(r"#\s*(?:noqa|type:\s*ignore|pylint:\s*disable)", all_src))
    swallow = len(re.findall(r"\bpass\s*$", all_src, re.M))

# ------------------------------------------------------------- comment density
comment_lines = 0
doc_comment_lines = 0
for f in src_files:
    for ln in read(f).splitlines():
        s = ln.strip()
        if lang == "rust":
            if s.startswith("///") or s.startswith("//!"):
                comment_lines += 1
                doc_comment_lines += 1
            elif s.startswith("//"):
                comment_lines += 1
        else:
            if s.startswith("#"):
                comment_lines += 1
            elif s.startswith('"""') or s.startswith("'''"):
                comment_lines += 1
                doc_comment_lines += 1

doc_files = walk(REPO, DOC_EXT)
doc_loc = sum(nonblank(read(f)) for f in doc_files)


def git(*args):
    try:
        out = subprocess.run(
            ["git", "-C", REPO, *args],
            capture_output=True, text=True, check=True,
        )
        return out.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


subjects = [s for s in git("log", "--no-merges", "--format=%s").splitlines() if s.strip()]
CONV = re.compile(
    r"^(?:feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)"
    r"(?:\([^)]*\))?!?:\s"
)
conv_ok = sum(1 for s in subjects if CONV.match(s))
bodies_raw = git("log", "--no-merges", "--format=%b")
coauth_claude = len(re.findall(r"Co-authored-by:\s*Claude", bodies_raw, re.I))

# commit span in days, from the git graph only (no wall clock)
dates = [d for d in git("log", "--no-merges", "--format=%at").splitlines() if d.strip()]
if len(dates) >= 2:
    ts = sorted(int(d) for d in dates)
    span_days = max(1, (ts[-1] - ts[0]) // 86400)
else:
    span_days = 1


def ratio(a, b, places=2):
    if b == 0:
        return "NA"
    return f"{a / b:.{places}f}"


def pct(a, b):
    if b == 0:
        return "NA"
    return f"{100.0 * a / b:.2f}"


rows = [
    ("REPO", os.path.basename(REPO)),
    ("LANG", lang),
    ("SRC_FILES", len(src_files)),
    ("SRC_LOC", src_loc),
    ("TEST_FNS", test_fns),
    ("ASSERTS", assert_total),
    ("ASSERTS_PER_TEST", ratio(assert_total, test_fns)),
    ("TESTS_ZERO_ASSERT_PCT", pct(tests_with_zero, test_fns)),
    ("TESTS_TRIVIAL_ONLY_PCT", pct(tests_trivial_only, test_fns)),
    ("TEST_FNS_PER_KLOC", ratio(1000 * test_fns, src_loc)),
    ("UNWRAP_PER_KLOC", ratio(1000 * unwraps, src_loc)),
    ("EXPECT_PER_KLOC", ratio(1000 * expects, src_loc)),
    ("PANIC_PER_KLOC", ratio(1000 * panics, src_loc)),
    ("NONTEST_LOC", nontest_loc),
    ("NONTEST_UNWRAP_PER_KLOC", ratio(1000 * nt_unwraps, nontest_loc)),
    ("NONTEST_EXPECT_PER_KLOC", ratio(1000 * nt_expects, nontest_loc)),
    ("NONTEST_PANIC_PER_KLOC", ratio(1000 * nt_panics, nontest_loc)),
    ("TODO_MARKERS", todos),
    ("SUPPRESSIONS", suppressions),
    ("SUPPRESSIONS_PER_KLOC", ratio(1000 * suppressions, src_loc)),
    ("IGNORED_RESULT_PER_KLOC", ratio(1000 * swallow, src_loc)),
    ("COMMENT_PCT", pct(comment_lines, src_loc)),
    ("DOC_COMMENT_SHARE_PCT", pct(doc_comment_lines, comment_lines)),
    ("DOC_FILES", len(doc_files)),
    ("DOC_LOC", doc_loc),
    ("DOC_TO_CODE", ratio(doc_loc, src_loc)),
    ("COMMITS", len(subjects)),
    ("COMMIT_SPAN_DAYS", span_days),
    ("COMMITS_PER_DAY", ratio(len(subjects), span_days)),
    ("CONVENTIONAL_COMMIT_PCT", pct(conv_ok, len(subjects))),
    ("COAUTHOR_CLAUDE_PCT", pct(coauth_claude, len(subjects))),
]

for k, v in rows:
    print(f"{k}={v}")
PYEOF
