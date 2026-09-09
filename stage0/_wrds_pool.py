# -*- coding: utf-8 -*-
"""
_wrds_pool.py
=============
One WRDS connection per worker process, opened safely.

Stage 0 fetches its CUSIP chunks from WRDS. Running several chunks at once means
several connections, and there are two traps in that which are worth stating plainly
because neither announces itself.

FIRST: a connection failure does not look like one. Whenever the wrds package cannot
connect -- for ANY reason -- it falls through to re-prompting for a username, and in a
batch job, where stdin is closed, that prompt dies as

    EOFError: EOF when reading a line

So a server-side refusal and a missing password arrive as the same keyboard error, and
neither says which it is. That ambiguity is the real trap: it is easy to read every
EOFError as the connection cap and go hunting a limit that is not the problem. Two
causes, in the order worth checking:

  * No usable credentials -- WRDS_USERNAME unset, so config.py falls back to the
    literal "your_wrds_username", or no ~/.pgpass entry. Ruled out FIRST below, before
    a connection is even attempted, because it is much the commoner cause and costs
    nothing to check.
  * The per-user connection limit, once several workers are open at once.

SECOND: never inherit a connection across a fork. A psycopg2 socket shared between a
parent and its children corrupts the protocol in ways that surface much later as
unrelated errors. Every worker therefore builds its OWN connection inside the child,
in the pool initializer, and never receives one as an argument.

MEASURED on this account, 2026-09-09 (tests/probe_wrds_connections.py):

  * 7 connections can be held simultaneously; the 8th fails.
  * Opening 6 AT ONCE, with no stagger and no lock, succeeded -- so the ceiling is on
    connections HELD, not on how fast they are opened.
  * A connect costs about 5 s, so a pool of N takes roughly 5N seconds to start.

The handshake is still serialised and staggered here. It costs a few seconds once,
it makes start-up deterministic rather than a race, and it means a lower cap on
someone else's account degrades into slow rather than broken.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import logging
import os
import threading
import time

import wrds

# One connection per PROCESS, created by the pool initializer and reused for every
# chunk that worker handles. Never share it across processes, and never re-create it
# per chunk: at ~5 s a connect, doing that for a few hundred chunks would cost more
# than the parallelism saves, and would storm the connection limit besides.
_CONNECTION = None
_USERNAME = None

# Serialise the handshake. Query execution stays fully parallel; only connecting is
# one-at-a-time.
_CONNECT_LOCK = threading.Lock()
_CONNECT_STAGGER_S = 0.4

# Substrings that mean "the server refused us for capacity reasons". The retry logic
# in the engines only knew about dropped connections, so a refusal was fatal.
CONNECTION_LIMIT_SIGNATURES = (
    "too many connections",
    "remaining connection slots",
    "connection limit",
    "too many clients",
)

# config.py's fallback when WRDS_USERNAME is not exported. Connecting with it fails
# indistinguishably from the connection limit, so catch it before we try.
PLACEHOLDER_USERNAME = "your_wrds_username"


class WRDSConnectionLimit(RuntimeError):
    """Raised when a WRDS connection is refused and the cap is a live possibility."""


class WRDSCredentialsMissing(RuntimeError):
    """Raised when there is no usable username to connect with."""


def _is_connection_limit(exc: BaseException) -> bool:
    """Is this worth retrying as a capacity problem?"""
    if isinstance(exc, WRDSCredentialsMissing):
        return False              # no amount of retrying will find a password
    if isinstance(exc, (EOFError, WRDSConnectionLimit)):
        return True               # ambiguous, but retryable either way
    text = str(exc).lower()
    return any(sig in text for sig in CONNECTION_LIMIT_SIGNATURES)


def connect(username: str | None = None, *, attempts: int = 3):
    """Open one connection, with the handshake serialised and staggered."""
    user = username or _USERNAME or os.environ.get("WRDS_USERNAME") or ""
    if not user or user == PLACEHOLDER_USERNAME:
        raise WRDSCredentialsMissing(
            """No WRDS username (got %r). Export it before running:
    export WRDS_USERNAME="your_id"
Without it the wrds package prompts on stdin and the job dies with
'EOFError: EOF when reading a line' -- which looks exactly like the connection
limit, and is not.""" % user)

    last = None
    for attempt in range(1, attempts + 1):
        try:
            with _CONNECT_LOCK:
                time.sleep(_CONNECT_STAGGER_S)
                return wrds.Connection(wrds_username=user)
        except Exception as exc:      # EOFError included -- it IS an Exception
            last = exc
            if not _is_connection_limit(exc) or attempt == attempts:
                break
            wait = 2.0 * attempt
            logging.warning(
                "WRDS refused a connection (attempt %d/%d); waiting %.0fs before "
                "retrying.", attempt, attempts, wait)
            time.sleep(wait)

    if _is_connection_limit(last):
        raise WRDSConnectionLimit(
            """WRDS refused a connection for user %r after %d attempts.
The wrds package reports EVERY connect failure as 'EOF when reading a line',
because its failure path calls input(). It is not a stdin problem, and it does
not say which failure it was. Check, in order:
  1. ~/.pgpass has a line for wrds-pgdata.wharton.upenn.edu:9737 with this
     username, and is chmod 600.
  2. The per-user CONNECTION LIMIT -- lower CONCURRENCY in
     stage0/_trace_settings.py, or re-measure it with
     tests/probe_wrds_connections.py.""" % (user, attempts)) from last
    raise last


def worker_init(username: str | None = None) -> None:
    """Pool initializer: give this process its own connection, once."""
    global _CONNECTION, _USERNAME
    _USERNAME = username
    _CONNECTION = connect(username)
    logging.info("[worker %s] WRDS connection established", os.getpid())


def worker_close() -> None:
    """Release this process's connection. A leaked session counts against the cap."""
    global _CONNECTION
    if _CONNECTION is not None:
        try:
            _CONNECTION.close()
        except Exception:
            pass
        _CONNECTION = None


def get_connection():
    """This process's connection, opening one if the initializer did not run."""
    global _CONNECTION
    if _CONNECTION is None:
        _CONNECTION = connect(_USERNAME)
    return _CONNECTION


def reconnect():
    """Replace this process's connection after a drop."""
    worker_close()
    return get_connection()


def raw_sql(sql: str, params=None, *, max_retries: int = 3, base_sleep: float = 2.0):
    """Run a query on this process's connection, reconnecting on transient failures.

    Mirrors the engines' _raw_sql_with_retry, with two differences that matter for a
    pool: it operates on this PROCESS's connection rather than a shared attribute --
    so one worker reconnecting cannot close a connection another is mid-query on --
    and it treats a refused connection as retryable rather than fatal.
    """
    from sqlalchemy.exc import OperationalError as SAOperationalError
    try:
        from psycopg2 import OperationalError as PGOperationalError
    except Exception:                                       # pragma: no cover
        PGOperationalError = ()

    transient_drops = (
        "ssl connection has been closed",
        "server closed the connection",
        "connection not open",
        "terminating connection",
        "connection reset",
        "connection already closed",
    )

    attempt = 0
    while True:
        try:
            return get_connection().raw_sql(sql, params=params)
        except (SAOperationalError, PGOperationalError, EOFError,
                WRDSConnectionLimit) as exc:
            attempt += 1
            msg = str(exc).lower()
            retryable = (any(s in msg for s in transient_drops)
                         or _is_connection_limit(exc))
            if not retryable or attempt > max_retries:
                logging.exception("WRDS query failed (attempt %d/%d)", attempt, max_retries)
                raise
            sleep_s = base_sleep * (2 ** (attempt - 1))
            logging.warning("WRDS issue (%s). Reconnecting, retry in %.1fs ...",
                            type(exc).__name__, sleep_s)
            time.sleep(sleep_s)
            reconnect()
