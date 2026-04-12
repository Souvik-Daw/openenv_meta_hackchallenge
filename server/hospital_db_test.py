import sqlite3
import os
path='./hospital.db'
print('exists', os.path.exists(path))
conn=sqlite3.connect(path)
cur=conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name")
for row in cur.fetchall():
    print(row)
conn.close()