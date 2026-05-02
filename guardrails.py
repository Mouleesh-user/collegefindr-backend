"""guardrails.py — input/output validation and safety checks for CollegeFindr chat.

Imported by app.py and called around the OpenRouter LLM call. Pure Python, no external
deps so it can be unit-tested in isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Pattern libraries
# --------------------------------------------------------------------------- #

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|the\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|rules?)",
    r"disregard\s+(?:all\s+|the\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?)",
    r"forget\s+(?:everything|all|prior|previous|above)",
    r"^\s*system\s*[:>]",
    r"```\s*system",
    r"<\s*system\s*>",
    r"you\s+(?:are|act)\s+(?:now\s+)?(?:a|an)\s+.{0,40}?(?:without|that\s+ignores)\s+(?:safety|rules|restrictions)",
    r"reveal\s+(?:your\s+|the\s+)?(?:system\s+)?prompt",
    r"print\s+(?:your\s+|the\s+)?(?:system\s+)?prompt",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"\bDAN\b\s+mode",
]

BIAS_TRIGGER_PATTERNS = [
    r"(?:only|exclusively)\s+(?:for\s+)?(?:girls?|boys?|men|women|males?|females?|"
    r"hindus?|muslims?|christians?|brahmins?|dalits?)",
    r"(?:girls?|boys?|men|women|hindus?|muslims?|christians?|brahmins?|dalits?)[\s\-]+only\b",
    r"\bno\s+(?:girls?|boys?|men|women|hindus?|muslims?|christians?|dalits?|sc/?st)\b",
    r"\breject\s+(?:girls?|boys?|sc|st|obc|women|men)\b",
    r"colleges?\s+(?:that|which)\s+(?:exclude|ban|reject)\s+.{1,30}?(?:caste|religion|gender)",
    r"\bdiscriminat\w*\s+against\b",
]

EXACT_FACT_PATTERNS = [
    # cutoff / closing rank — verb optional, table cells covered
    re.compile(r"\b(?:cut[- ]?off|closing\s+rank|opening\s+rank)\b[^|\n]{0,40}?\d{2,5}", re.IGNORECASE),
    # fees with ₹ / Rs / INR / lakh / lpa — verb optional
    re.compile(r"(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakh|lakhs|l|cr|crore|k|lpa))?", re.IGNORECASE),
    re.compile(r"\bfees?\b[^|\n]{0,30}?[\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|cr|crore|lpa)\b", re.IGNORECASE),
    # deadline with date-like number
    re.compile(r"\bdeadline\b[^|\n]{0,30}?\d", re.IGNORECASE),
    # ranked / NIRF rank
    re.compile(r"\b(?:ranked?|rank|nirf)\b[^|\n]{0,15}?#?\s*\d{1,3}\b", re.IGNORECASE),
    # placement rate / package
    re.compile(r"\bplacement\b[^|\n]{0,25}?\d{2,3}\s*%", re.IGNORECASE),
    re.compile(r"\b(?:average|median|highest)\s+package\b[^|\n]{0,25}?\d", re.IGNORECASE),
]

HEDGE_TOKENS = (
    "approximate", "approximately", "around", "roughly", "verify", "official", "official website",
    "may vary", "subject to", "historical", "previous year", "last year",
)

DEFAULT_KNOWN_COLLEGES = (
    "iit bombay", "iit delhi", "iit madras", "iit kanpur", "iit kharagpur", "iit roorkee",
    "iit guwahati", "iit hyderabad", "iit bhu", "iit indore", "iit ropar", "iit gandhinagar",
    "iit mandi", "iit jodhpur", "iit patna", "iit bhilai", "iit goa", "iit dharwad",
    "iisc bangalore", "iisc bengaluru", "iiit hyderabad", "iiit bangalore", "iiit delhi",
    "nit trichy", "nit warangal", "nit surathkal", "nit calicut", "nit allahabad",
    "nit rourkela", "nit kurukshetra", "nit jaipur",
    "bits pilani", "bits goa", "bits hyderabad",
    "vit vellore", "srm chennai", "manipal institute of technology",
    "delhi university", "du", "jnu", "jamia millia islamia", "bhu", "amu",
    "anna university", "ssn college of engineering", "psg tech", "thiagarajar college of engineering",
    "iim ahmedabad", "iim bangalore", "iim bengaluru", "iim calcutta", "iim kolkata",
    "iim lucknow", "iim indore", "iim kozhikode",
    "xlri jamshedpur", "fms delhi", "isb hyderabad", "spjimr",
    "aiims delhi", "cmc vellore", "afmc pune", "maulana azad medical college", "kgmu lucknow",
    "nls bangalore", "nlu bangalore", "nalsar hyderabad", "nlu jodhpur", "nujs kolkata",
    "loyola college", "st stephens college", "lsr", "lady shri ram college",
    "miranda house", "shri ram college of commerce", "srcc",
)
COLLEGE_LIST_PATH = Path(__file__).parent / "data" / "known_colleges.json"

VALID_COURSE_KEYWORDS = (
    "engineering", "cse", "cs", "computer science", "ece", "electronics", "mechanical",
    "civil", "ai", "ml", "data science", "ds", "it", "information technology", "chemical",
    "mbbs", "medical", "bds", "nursing", "pharmacy", "ayush", "physiotherapy",
    "law", "ba ll.b", "bba ll.b", "llb", "ll.b",
    "mba", "pgdm", "bba", "bcom", "ba", "bsc", "ma", "msc", "btech", "be", "mtech", "phd",
    "design", "architecture", "barch", "fashion", "animation", "hotel management",
)


@dataclass
class ValidationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extracted: Dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "extracted": self.extracted}


# --------------------------------------------------------------------------- #
# Input-side guards
# --------------------------------------------------------------------------- #

def detect_prompt_injection(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return pattern
    return None


def detect_bias_intent(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for pattern in BIAS_TRIGGER_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return pattern
    return None


# --- New detectors for QA failure analysis (TC-06, TC-10, TC-16, TC-24) --- #

GENDERED_BRANCH_QUERY = re.compile(
    r"\b(?:best|good|suitable|right|top)\s+(?:branch|course|stream|engineering|college)\b.{0,40}?\b"
    r"(?:for\s+a\s+)?(?:boy|boys|girl|girls|male|female|man|woman|men|women|son|daughter)\b",
    re.IGNORECASE,
)

NEGATION_CLAUSE = re.compile(
    r"\b(?:except|excluding|exclude|not|no|without|other\s+than)\s+([A-Za-z][A-Za-z0-9 .,&'\-/]{1,60}?)"
    r"(?=,|\.|;| and | but |$|\bin\b|\bwith\b|\bunder\b|\bbelow\b|\babove\b|\bnear\b|\bfor\b|\bplease\b)",
    re.IGNORECASE,
)

# Numeric value with no scale tag — covers TC-06 ("I got 280")
AMBIGUOUS_MARKS = re.compile(
    r"(?:scored|got|secured|with|have|made|having|earned|achieved|hit|reached)\s+(\d{2,3})"
    r"(?!\s*(?:%|percent|percentile|/|\\|\bout\s+of\b|\s*marks?\b|\s*lakh|\s*l\b|\s*k\b|\s*lpa\b|\s*rank\b))",
    re.IGNORECASE,
)
# Bare score near an exam token, e.g. "JEE 280", "NEET-UG 540"
_EXAM_NEAR_NUMBER = re.compile(
    r"\b(?:jee(?:[- ]?(?:main|advanced|adv))?|neet(?:[- ]?ug|[- ]?pg)?|cuet|gate|cat|mat|cmat|board|class\s*1[02])\b"
    r"[^\d\n]{0,20}?(\d{2,3})(?!\s*(?:%|percent|percentile|/|\bout\s+of\b|marks?\b|rank\b))",
    re.IGNORECASE,
)
_BARE_SCORE_PROMPT = re.compile(
    r"^\s*(\d{2,3})\s*[.,]?\s*(?:suggest|recommend|colleges?|what|where|which|tell|help)",
    re.IGNORECASE,
)

YEAR_REF = re.compile(r"\b(20\d{2})\b")


def detect_gendered_query(text: str) -> Optional[str]:
    """Flag paired-input gender bias triggers (TC-24 / TC-25).

    Returns the matched fragment so the chat handler can append an
    invariance instruction to the system prompt for that turn.
    """
    if not text:
        return None
    match = GENDERED_BRANCH_QUERY.search(text)
    return match.group(0) if match else None


_FACT_FOR_YEAR = re.compile(
    r"\b(?:cut[- ]?off|closing\s+rank|opening\s+rank|fees?|deadline|placement|"
    r"package|rank|score|seat\s+matrix|counsel+ing|josaa|nirf)\b",
    re.IGNORECASE,
)
_PLANNING_VERBS = re.compile(
    r"\b(?:plan|planning|aim|aiming|prepare|preparing|target|targeting|"
    r"appear|appearing|will\s+(?:write|take|appear)|graduat\w+|admission\s+in)\b",
    re.IGNORECASE,
)


def detect_future_year(text: str, *, today: Optional[datetime] = None) -> Optional[int]:
    """Return the offending year if the user is asking for a future *fact*.

    Only fires when the message references a future year AND a fact-bearing
    token (cutoff, fees, rank, placement…). Pure planning queries
    ("I plan to write JEE in 2027") pass through.
    """
    if not text:
        return None
    today = today or datetime.now(timezone.utc)
    current_year = today.year
    future_years = []
    for raw in YEAR_REF.findall(text):
        try:
            year = int(raw)
        except ValueError:
            continue
        if year > current_year:
            future_years.append(year)
    if not future_years:
        return None
    has_fact = bool(_FACT_FOR_YEAR.search(text))
    has_planning = bool(_PLANNING_VERBS.search(text))
    if has_fact and not has_planning:
        return future_years[0]
    return None


def detect_ambiguous_marks(text: str) -> Optional[int]:
    """Return the raw numeric score if the user gave a number without a scale.

    Guards TC-06 ("I got 280, suggest colleges") — the same number could mean
    JEE Main /300, NEET /720, or board /500. Forcing a clarification prevents
    the bot from collapsing to one interpretation.
    """
    if not text:
        return None
    candidates: List[int] = []
    for rx in (AMBIGUOUS_MARKS, _EXAM_NEAR_NUMBER, _BARE_SCORE_PROMPT):
        m = rx.search(text)
        if m:
            try:
                candidates.append(int(m.group(1)))
            except ValueError:
                continue
    for value in candidates:
        # Only ambiguous if number could legitimately belong to multiple scales.
        if 100 < value <= 720:
            return value
    return None


_LOC_TAILS = {"in", "with", "under", "below", "above", "near", "for", "please", "and", "but"}


def detect_negation_constraints(text: str) -> List[str]:
    """Extract entities the user explicitly does NOT want (TC-16).

    Example input: "Suggest engineering colleges in TN except Anna University
    and not any private deemed university, without capitation fees."
    Returns ['Anna University', 'private deemed university', 'capitation fees'].
    """
    if not text:
        return []
    raw = text.replace("\n", " ")
    fragments: List[str] = []
    for match in NEGATION_CLAUSE.finditer(raw):
        chunk = match.group(1).strip(" ,.;")
        if not chunk:
            continue
        # Strip trailing connector words.
        words = chunk.split()
        while words and words[-1].lower() in _LOC_TAILS:
            words.pop()
        cleaned = " ".join(words).strip()
        if 2 <= len(cleaned) <= 80:
            fragments.append(cleaned)
    # Dedupe case-insensitively, preserving order.
    seen = set()
    out: List[str] = []
    for frag in fragments:
        key = frag.lower()
        if key not in seen:
            seen.add(key)
            out.append(frag)
    return out


def extract_inputs_from_message(message: str) -> Dict[str, Any]:
    """Best-effort regex extraction of marks / budget / course / location."""
    extracted: Dict[str, Any] = {}
    if not message:
        return extracted
    text = message.lower()

    marks_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent|percentile|marks?\b)", text)
    if marks_match:
        try:
            extracted["marks_percent"] = float(marks_match.group(1))
        except ValueError:
            pass

    budget_match = re.search(
        r"(?:budget|fee|fees|afford|spend)[^\d]{0,12}(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|l|cr|crore|k|thousand|)",
        text,
    )
    if budget_match:
        raw = budget_match.group(1).replace(",", "")
        unit = budget_match.group(2)
        try:
            value = float(raw)
            if unit in ("lakh", "lakhs", "l"):
                value *= 100_000
            elif unit in ("cr", "crore"):
                value *= 10_000_000
            elif unit in ("k", "thousand"):
                value *= 1_000
            extracted["budget_inr"] = value
        except ValueError:
            pass

    for kw in VALID_COURSE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            extracted["course"] = kw
            break

    loc_match = re.search(r"(?:in|near|at|around|located in)\s+([a-z][a-z\s,&-]{2,40})(?:[.?!]|$)", text)
    if loc_match:
        extracted["location"] = loc_match.group(1).strip().rstrip(".? !,")

    return extracted


def validate_structured_inputs(inputs: Dict[str, Any]) -> ValidationResult:
    result = ValidationResult(extracted=dict(inputs or {}))
    if not inputs:
        return result

    marks = inputs.get("marks_percent")
    if marks is not None:
        try:
            marks_val = float(marks)
        except (TypeError, ValueError):
            result.fail("marks_percent must be a number")
        else:
            if marks_val < 0 or marks_val > 100:
                result.fail(f"marks_percent must be between 0 and 100 (got {marks_val})")
            elif marks_val < 30:
                result.warn(f"marks_percent={marks_val} is unusually low; recommendations may be limited")

    budget = inputs.get("budget_inr")
    if budget is not None:
        try:
            budget_val = float(budget)
        except (TypeError, ValueError):
            result.fail("budget_inr must be a number")
        else:
            if budget_val < 0:
                result.fail("budget_inr cannot be negative")
            elif budget_val > 10_00_00_000:
                result.warn(f"budget_inr={budget_val} is unrealistically high; verify intent")

    course = (inputs.get("course") or "").strip().lower()
    if course and not any(re.search(rf"\b{re.escape(k)}\b", course) for k in VALID_COURSE_KEYWORDS):
        result.warn(f"course '{course}' is not in the recognized course list")

    location = (inputs.get("location") or "").strip()
    if len(location) > 80:
        result.fail("location field too long (max 80 chars)")

    return result


def has_minimum_inputs(inputs: Dict[str, Any]) -> bool:
    """At least 2 of {marks, budget, course, location} should be present for a useful answer."""
    if not inputs:
        return False
    keys = ("marks_percent", "budget_inr", "course", "location")
    return sum(1 for k in keys if inputs.get(k) not in (None, "")) >= 2


# --------------------------------------------------------------------------- #
# Output-side guards
# --------------------------------------------------------------------------- #

def load_known_colleges() -> List[str]:
    try:
        if COLLEGE_LIST_PATH.exists():
            with COLLEGE_LIST_PATH.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [str(item).strip().lower() for item in data if str(item).strip()]
    except Exception:
        pass
    return list(DEFAULT_KNOWN_COLLEGES)


_COLLEGE_PHRASE_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+){1,5})\s+"
    r"(?:College|University|Institute|Institution|School|Polytechnic))\b"
)
_ACRONYM_LOC_RE = re.compile(
    r"\b(IIT|NIT|IIIT|IIM|AIIMS|BITS|NLU|NLS)\s+"
    r"((?:[A-Z][A-Za-z]{1,20})(?:\s+[A-Z][A-Za-z]{1,20}){0,3})"
)


_BENIGN_SUFFIXES = {
    "main campus", "north campus", "south campus", "east campus", "west campus",
    "city campus", "new campus", "old campus", "off campus",
}


def _normalize_college_token(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"[^a-z0-9 &]", "", s)
    return s


def _strip_benign_suffix(name: str) -> str:
    norm = _normalize_college_token(name)
    for suf in _BENIGN_SUFFIXES:
        if norm.endswith(" " + suf):
            return norm[: -(len(suf) + 1)].strip()
    return norm


def find_potentially_invented_colleges(reply: str, known: Optional[List[str]] = None) -> List[str]:
    """Flag candidate institution mentions that don't exact-match a known entry.

    Substring matches were too permissive ("IIT Bombay Mars Campus" passed via
    'iit bombay'). We now require a normalized exact match, with a small
    allowlist of benign campus suffixes ("Main Campus", "North Campus", ...).
    """
    if not reply:
        return []
    known_norm = { _normalize_college_token(k) for k in (known if known is not None else load_known_colleges()) }
    candidates: List[str] = []
    for match in _COLLEGE_PHRASE_RE.finditer(reply):
        candidates.append(match.group(1))
    for match in _ACRONYM_LOC_RE.finditer(reply):
        candidates.append(f"{match.group(1)} {match.group(2)}")

    suspicious: List[str] = []
    for cand in candidates:
        norm = _normalize_college_token(cand)
        if norm in known_norm:
            continue
        stripped = _strip_benign_suffix(cand)
        if stripped in known_norm:
            continue
        suspicious.append(cand)
    seen = set()
    deduped = []
    for s in suspicious:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


# --- New output-side enforcers (TC-16, TC-22, TC-31/33/35/36) --- #

# Personal-contact patterns — fabricated emails/phones in TC-22 are the failure.
_FABRICATED_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|protonmail|icloud)\.[A-Za-z]{2,}\b"
)
_FABRICATED_PHONE = re.compile(r"\b(?:\+?91[-\s]?)?[6-9]\d{2}[-\s]?\d{3}[-\s]?\d{4}\b")
_SYCOPHANCY_OPENER = re.compile(
    r"^(?:yes,?\s+that's\s+(?:correct|right)|that's\s+(?:correct|right)|absolutely,?\s+correct|"
    r"you\s+are\s+(?:correct|right))",
    re.IGNORECASE | re.MULTILINE,
)


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}\s*(?:\|\s*:?-{2,}\s*)+\|?\s*$")


def _is_table_structural(line: str) -> bool:
    """Header separator or pure pipe row — must be preserved to keep markdown table valid."""
    return bool(_TABLE_SEP_RE.match(line))


def enforce_negation(reply: str, exclusions: List[str]) -> Tuple[str, List[str]]:
    """Strip lines that mention any user-excluded entity (TC-16).

    Table-aware: never drops markdown header-separator rows (`| --- | --- |`),
    which would corrupt the surrounding table. Drops the matching data row
    only.
    """
    if not reply or not exclusions:
        return reply, []
    violations: List[str] = []
    kept_lines: List[str] = []
    excl_lower = [e.lower() for e in exclusions if e]
    for line in reply.splitlines():
        if _is_table_structural(line):
            kept_lines.append(line)
            continue
        line_lower = line.lower()
        hit = next((e for e in excl_lower if e and e in line_lower), None)
        if hit:
            violations.append(hit)
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    return cleaned, violations


def strip_sycophancy(reply: str) -> Tuple[str, bool]:
    """Remove a leading sycophantic confirmation so the rest of the answer can stand on its own.

    The detector still warns; this rewrite makes the change visible to the user.
    """
    if not reply:
        return reply, False
    stripped = reply.lstrip()
    match = _SYCOPHANCY_OPENER.match(stripped)
    if not match:
        return reply, False
    # Drop the matched opener and any trailing punctuation/whitespace.
    rest = stripped[match.end():].lstrip(" ,.!:;\n")
    if not rest:
        return reply, False
    return rest, True


def strip_fabricated_contacts(reply: str) -> Tuple[str, List[str]]:
    """Remove personal email and phone numbers from LLM output (TC-22).

    The LLM has a habit of inventing plausible-looking contacts to round out a
    refusal. Replace each with a neutral placeholder and return what was
    stripped for audit.
    """
    if not reply:
        return reply, []
    stripped: List[str] = []

    def _email_repl(match: re.Match) -> str:
        stripped.append(match.group(0))
        return "[contact removed — verify on the official website]"

    def _phone_repl(match: re.Match) -> str:
        stripped.append(match.group(0))
        return "[number removed — verify on the official website]"

    cleaned = _FABRICATED_EMAIL.sub(_email_repl, reply)
    cleaned = _FABRICATED_PHONE.sub(_phone_repl, cleaned)
    return cleaned, stripped


def scrub_invented_colleges(reply: str, known: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Replace mentions of unverifiable institutions with a hedged note.

    Targets TC-31 (IIT Coimbatore), TC-33 (IIIT Kashmir / NIT Lakshadweep),
    TC-35 (CEG South Campus). Operates conservatively — only rewrites tokens
    that match the institutional regex AND fail the known-college lookup.
    """
    if not reply:
        return reply, []
    suspicious = find_potentially_invented_colleges(reply, known)
    if not suspicious:
        return reply, []
    cleaned = reply
    for name in suspicious:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        cleaned = pattern.sub(
            f"**{name}** (I could not verify this institution; please confirm it exists "
            f"before relying on this entry)",
            cleaned,
            count=2,
        )
    return cleaned, suspicious


def detect_sycophancy(reply: str) -> bool:
    """Heuristic: response opens by confirming a user-stated 'fact' (TC-36)."""
    if not reply:
        return False
    return bool(_SYCOPHANCY_OPENER.search(reply.strip()))


def today_marker(today: Optional[datetime] = None) -> str:
    """Inject the current date into the system prompt for temporal grounding."""
    today = today or datetime.now(timezone.utc)
    return (
        f"Today's date is {today.strftime('%Y-%m-%d')}. The most recent admission "
        f"cycle you may have data for is JoSAA {today.year - 1} or earlier. NEVER "
        f"quote a closing rank, cutoff, or fee for {today.year + 1} or later — those "
        f"data do not exist yet. If asked, say so plainly and offer the latest "
        f"verified year instead."
    )


def find_unhedged_facts(reply: str) -> List[str]:
    if not reply:
        return []
    flagged: List[str] = []
    for line in reply.splitlines():
        for pattern in EXACT_FACT_PATTERNS:
            if pattern.search(line):
                lower = line.lower()
                if not any(token in lower for token in HEDGE_TOKENS):
                    flagged.append(line.strip())
                    break
    return flagged


DISCLAIMER_TEXT = (
    "\n\n---\n*Note: figures above are approximate and may change. Always verify cutoffs, "
    "fees, and deadlines on the official college website before applying.*"
)


def append_disclaimer(reply: str) -> str:
    if not reply:
        return reply
    lower = reply.lower()
    if "verify" in lower and ("official" in lower or "website" in lower):
        return reply
    return reply + DISCLAIMER_TEXT


def validate_response(
    reply: str,
    known_colleges: Optional[List[str]] = None,
    *,
    exclusions: Optional[List[str]] = None,
) -> ValidationResult:
    """Validate AND rewrite the LLM reply in place.

    Returned `extracted['reply']` is the post-scrub text the caller should
    serve to the user. Warnings record what was changed; errors are reserved
    for cases where the entire reply must be discarded.
    """
    result = ValidationResult()
    if not reply or not reply.strip():
        result.fail("empty response from LLM")
        return result

    cleaned = reply
    if len(cleaned) > 6000:
        result.warn(f"response unusually long: {len(cleaned)} chars")

    if exclusions:
        cleaned, violated = enforce_negation(cleaned, exclusions)
        if violated:
            result.warn(f"negation_violation_stripped:{violated[:5]}")

    cleaned, stripped_contacts = strip_fabricated_contacts(cleaned)
    if stripped_contacts:
        result.warn(f"fabricated_contacts_stripped:{stripped_contacts[:3]}")

    cleaned, invented = scrub_invented_colleges(cleaned, known_colleges)
    if invented:
        result.warn(f"possibly_invented_colleges_flagged:{invented[:5]}")

    unhedged = find_unhedged_facts(cleaned)
    if unhedged:
        result.warn(f"unhedged_factual_claims:{unhedged[:3]}")

    if detect_sycophancy(cleaned):
        rewritten, did_strip = strip_sycophancy(cleaned)
        if did_strip:
            cleaned = rewritten
            result.warn("sycophancy_opener_stripped")
        else:
            result.warn("sycophancy_opener_detected")

    result.extracted["reply"] = cleaned
    return result


# --------------------------------------------------------------------------- #
# Canned safe replies
# --------------------------------------------------------------------------- #

FALLBACK_INSUFFICIENT_INPUT = (
    "I want to give you a recommendation I'm actually confident about. "
    "Could you share a bit more so I can help you better?\n\n"
    "- **Marks / percentile** (and which exam — JEE / NEET / CUET / board)\n"
    "- **Course or field** you want to study\n"
    "- **Approximate budget per year** (₹)\n"
    "- **Preferred locations** (city or state)\n\n"
    "*I will only recommend colleges I'm confident exist, and I'll always ask you to "
    "verify fees, cutoffs, and deadlines on the official website.*"
)

REFUSAL_PROMPT_INJECTION = (
    "I noticed the message looks like it's trying to override my instructions. "
    "I'm only able to help with finding colleges, courses, and admissions in India. "
    "Could you tell me what course, marks, budget, and location you're considering?"
)

REFUSAL_BIAS = (
    "I can't recommend colleges based on caste, religion, or gender as exclusion criteria. "
    "Indian colleges follow legal reservation policies (SC/ST/OBC/EWS) and many offer "
    "scholarships for specific backgrounds — I'm happy to walk you through those neutrally. "
    "What course and budget are you working with?"
)

REFUSAL_IMPOSSIBLE_INPUT = (
    "Some of those numbers don't add up (for example, marks above 100% or a negative budget). "
    "Could you double-check and resend? I just need realistic values for marks, budget, course, and location."
)

CLARIFY_AMBIGUOUS_MARKS = (
    "I want to make sure I read your score on the right scale before recommending anything. "
    "Could you tell me which exam this is and the maximum possible? For example:\n\n"
    "- **JEE Main** is out of 300\n"
    "- **NEET-UG** is out of 720\n"
    "- **Board (CBSE/State)** is usually out of 500\n"
    "- **JEE Advanced** is reported as a rank, not a score\n\n"
    "Once I know the scale, I can give you a recommendation I'm actually confident about."
)

CLARIFY_FUTURE_YEAR = (
    "I can't tell you cutoffs or fees for a year that hasn't happened yet — those numbers "
    "don't exist until the counselling cycle closes. The most recent verified cycle is the "
    "previous year's. Want me to share that instead, or talk about how cutoffs have trended?"
)

GENDER_INVARIANCE_INSTRUCTION = (
    "INVARIANCE NOTE for this turn: the user mentioned gender. Your branch and "
    "college recommendations MUST be identical to what you would give a user of "
    "any other gender with the same marks/budget/course. Do NOT steer the user "
    "toward 'soft' branches, women-only colleges, or 'work-life balance' framing "
    "unless the user explicitly asked. Treat gender as non-informative for branch "
    "selection."
)
