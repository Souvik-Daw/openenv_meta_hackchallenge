from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Sequence, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import contextmanager
from functools import lru_cache
import sqlite3
import json
import re

############## Database Utilities ##############

_identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Return a sqlite3 connection (not a context manager). Kept for compatibility.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def open_connection(db_path: str):
    """
    Context manager that opens a sqlite3 connection and ensures it's closed.
    Use: with open_connection(path) as conn: ...
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def _safe_identifier(name: str) -> str:
    """
    Validate SQL identifier (table or column). Raises ValueError if invalid.
    This avoids SQL injection for identifiers since sqlite3 only parameterizes values.
    """
    if not _identifier_re.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name

def _columns_clause(columns: Optional[Sequence[str]]) -> str:
    if not columns:
        return "*"
    return ", ".join(_safe_identifier(c) for c in columns)

def search(
    conn_or_path: Union[sqlite3.Connection, str],
    table: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    params: Optional[Sequence] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Search rows from a table.

    - conn_or_path: sqlite3.Connection or path to DB file
    - table: table name (validated as identifier)
    - columns: list of column names to select; if None selects all
    - where: optional WHERE clause (use ? placeholders for values)
    - params: parameters for the WHERE clause placeholders
    - limit: optional integer limit

    Returns list of rows as dictionaries.
    """
    cols = _columns_clause(columns)
    tname = _safe_identifier(table)

    sql = f"SELECT {cols} FROM {tname}"
    if where:
        sql += f" WHERE {where}"
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        sql += f" LIMIT {limit}"

    if isinstance(conn_or_path, sqlite3.Connection):
        conn = conn_or_path
        cur = conn.execute(sql, params or ())
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    else:
        with open_connection(conn_or_path) as conn:
            cur = conn.execute(sql, params or ())
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            return rows

def search_like(
    conn_or_path: Union[sqlite3.Connection, str],
    table: str,
    column: str,
    term: str,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Convenience helper to search a single column using LIKE (case-insensitive).
    Example: search_like(db, "patients", "name", "john")
    """
    _safe_identifier(column)
    where = f"LOWER({column}) LIKE ?"
    param = f"%{term.lower()}%"
    return search(conn_or_path, table, columns=columns, where=where, params=(param,), limit=limit)

def insert_db(
    conn_or_path: Union[sqlite3.Connection, str],
    table: str,
    data: Dict,
) -> Union[int, List[int]]:
    """
    Insert a single row (dict).
    """
    keys = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    values = tuple(data.values())

    sql = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"

    if isinstance(conn_or_path, sqlite3.Connection):
        conn = conn_or_path
        cur = conn.execute(sql, values)
        conn.commit()  # Commit the transaction
        cur.close()
        return True
    else:
        with open_connection(conn_or_path) as conn:
            cur = conn.execute(sql, values)
            conn.commit()  # Commit the transaction
            cur.close()
            return True
    return False

    

############## Modeling Utilities ##############

@lru_cache(maxsize=None)
def train_tfidf():
    
    with open("server/symptoms.json") as f:
        __data = json.load(f)
    
    conditions = [item["condition"] for item in __data]
    departments = [item["department"] for item in __data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(conditions)

    return X,vectorizer,departments

def find_department_tfidf(query):
    model,vectorizer,departments = train_tfidf()
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, model)[0]

    best_idx = sims.argmax()
    
    return departments[best_idx]


# find_department_tfidf("chest pain, heart attack, hypertension")
# find_department_tfidf("chest pain, heart attack")
# find_department_tfidf("headache, migraine, heart attack")

