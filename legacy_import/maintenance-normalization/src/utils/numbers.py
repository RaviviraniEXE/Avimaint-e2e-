"""Number word <-> digit conversion.

Amin's expert-cleaned dataset was produced for text-to-speech, so it spells
numbers out ("#2" -> "number two", "150" -> "one hundred and fifty"). This
project targets information extraction, where digits are preferable, so the
gold is converted back to digits with `digitize`. `spell` is the inverse, used
only when running the pipeline in `numbers: words` mode to match Amin exactly.

Both are dependency-free (important for the RTX-3050 Windows setup) and handle
the constructs that actually occur in this corpus: lists ("one, two and three"
-> "1, 2 and 3"), ranges ("one hundred to one hundred and fifty" -> "100 to
150"), and scale words ("twelve hundred" -> "1200"). "and" is treated as part
of a number only after a scale word (hundred/thousand), so "one and three"
stays a two-item list.
"""
from __future__ import annotations

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
NUMWORDS = set(UNITS) | set(TENS) | set(SCALES)


def _words_to_num(words) -> int:
    total = 0
    current = 0
    for w in words:
        if w in UNITS:
            current += UNITS[w]
        elif w in TENS:
            current += TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in ("thousand", "million"):
            total += (current or 1) * SCALES[w]
            current = 0
        else:
            raise ValueError(w)
    return total + current


def digitize(text: str):
    """Convert spelled-out numbers in `text` to digits.

    Returns (converted_text, flagged) where flagged is True if a number-like
    span could not be parsed (kept verbatim for human review).
    """
    toks = text.split()
    out = []
    i = 0
    flagged = False

    def core(t):
        return t.lower().strip('.,;:()"')

    while i < len(toks):
        if core(toks[i]) in NUMWORDS:
            span = []
            j = i
            while j < len(toks):
                cw = core(toks[j])
                if cw in NUMWORDS:
                    span.append(cw)
                    if any(c in toks[j] for c in ",;:"):
                        j += 1
                        break
                    j += 1
                elif cw == "and" and span and span[-1] in SCALES:
                    if any(c in toks[j] for c in ",;:"):
                        break
                    j += 1
                else:
                    break
            try:
                num = _words_to_num(span)
                tail = "".join(c for c in toks[j - 1] if c in '.,;:)"')
                out.append(str(num) + tail)
                i = j
            except Exception:
                out.append(toks[i])
                flagged = True
                i += 1
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out), flagged


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rem = n % 100
        head = _ONES[n // 100] + " hundred"
        return head if rem == 0 else head + " and " + _int_to_words(rem)
    if n < 1_000_000:
        rem = n % 1000
        head = _int_to_words(n // 1000) + " thousand"
        return head if rem == 0 else head + " " + _int_to_words(rem)
    return str(n)  # beyond range we care about


def spell(text: str) -> str:
    """Inverse of digitize for `numbers: words` mode: spell integer tokens out."""
    out = []
    for tok in text.split():
        core = tok.strip('.,;:()"')
        tail = tok[len(tok.rstrip('.,;:)"')):]
        if core.isdigit() and len(core) <= 6:
            out.append(_int_to_words(int(core)) + tail)
        else:
            out.append(tok)
    return " ".join(out)

