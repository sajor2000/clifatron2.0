"""Site-side log and traceback redaction (U5 review findings #7, #21, #33, #34).

Split out of `schema.py`: that module owns the export contract, and logging shares
nothing with it but a word list. Keeping them together meant one set of constants
served two very different matching semantics — structured field names on one side,
free prose on the other — so tightening either silently changed the other.

`configs/artifact_policy.yaml` declares `operational_logs.prohibited_content`
(identifiers, patient_rows, local_source_paths, free_text). This module is what makes
that declaration a control rather than a document.

Two rules this module exists to enforce, both of which it previously got wrong:

  1. **It fails CLOSED.** A record the filter cannot format is redacted wholesale, not
     passed through. A disclosure control that lets a record escape on its own internal
     error has the failure direction backwards.
  2. **It covers tracebacks.** `record.msg` is not the only channel: `exc_info` and
     `stack_info` carry the same strings, and this codebase raises with local paths
     embedded, so an operator-returned traceback used to carry exactly what the
     redacted message no longer did.
"""

from __future__ import annotations

import re
import traceback

# Free-text redaction terms. Deliberately SEPARATE from schema.PROHIBITED_SUBSTRINGS,
# which matches structured field names. The two lists overlap today but must be free to
# diverge: a schema field name is matched as a bare substring (correct for `hosp_id`),
# while prose needs to avoid shredding ordinary debuggable words.
LOG_REDACTION_TERMS = (
    "patient_id", "patientid", "hospitalization_id", "hosp_id", "encounter_id",
    "subject_id", "stay_id", "mrn",
)

# A path-shaped token: an absolute POSIX path, a Windows path, or a ~-rooted path.
# Anchored on a separator so an ordinary word containing a slash-free segment is safe.
_PATH_RE = re.compile(r"(?:~|\.{0,2})?(?:/[\w.\-]+){2,}/?|[A-Za-z]:\\[\w.\\\-]+")

# `key=value` or `key: value` where the key is an identifier term. Redacts the VALUE,
# which is the part that identifies a patient, and keeps the key so logs stay debuggable.
_KV_RE = re.compile(
    r"\b(" + "|".join(LOG_REDACTION_TERMS) + r")\b\s*[=:]\s*\S+",
    flags=re.IGNORECASE,
)


def redact(text: str) -> str:
    """Replace path-shaped tokens and identifier key/value pairs in a log line.

    Order matters: key/value pairs are collapsed before the bare-term pass, so
    `patient_id=12345` becomes one redaction rather than a redacted key beside a
    surviving value.
    """
    out = str(text)
    out = _KV_RE.sub(lambda m: f"{m.group(1)}=<redacted>", out)
    out = _PATH_RE.sub("<redacted:path>", out)
    for term in LOG_REDACTION_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", "<redacted>", out, flags=re.IGNORECASE)
    return out


class SanitizingFilter:
    """Redact prohibited content from a log record before it reaches any sink.

    Implements the `logging.Filter` protocol. Returns True (keep the record) in every
    path — including its own error path — but only ever after replacing the content.
    """

    def filter(self, record) -> bool:  # logging.Filter protocol
        try:
            message = record.getMessage()
        except Exception:
            # FAIL CLOSED. An unformattable record is one whose content we could not
            # inspect, so it must not be emitted as-is (review finding #7).
            record.msg = "<redacted: log record could not be formatted for inspection>"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            return True

        record.msg = redact(message)
        record.args = ()

        # Tracebacks are a separate channel and carry the same strings (finding #21).
        if record.exc_info:
            try:
                formatted = "".join(traceback.format_exception(*record.exc_info))
            except Exception:
                formatted = "<redacted: traceback could not be formatted>"
            record.exc_text = redact(formatted)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)

        if record.stack_info:
            record.stack_info = redact(record.stack_info)

        return True


def install_log_sanitizer(logger) -> None:
    """Attach the sanitizing filter to a logger and every handler it owns.

    Both are needed: a logger-level filter does not run for records that reach a handler
    via propagation from a child logger, and a handler-level filter does not run for
    records the logger drops first.
    """
    filt = SanitizingFilter()
    logger.addFilter(filt)
    for handler in logger.handlers:
        handler.addFilter(filt)


__all__ = ["LOG_REDACTION_TERMS", "SanitizingFilter", "install_log_sanitizer", "redact"]
