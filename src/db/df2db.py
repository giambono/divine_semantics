import os
import pandas as pd
import sqlite3

import config

# Load the ODS file
df = pd.read_excel(config.TRANSLATIONS_REPO, engine="odf")

conn = sqlite3.connect(config.DP_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    volume TEXT NOT NULL,
    canto TEXT NOT NULL,
    verse TEXT NOT NULL,
    dante TEXT NOT NULL,
    singleton TEXT,
    musa TEXT,
    kirkpatrick TEXT,
    durling TEXT,
    UNIQUE(volume, canto, verse)  -- Prevents duplicate rows
);
""")
conn.commit()

df.to_sql("translations", conn, if_exists="replace", index=False)

conn.close()
