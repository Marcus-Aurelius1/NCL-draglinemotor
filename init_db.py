import sqlite3
import pandas as pd

# Load the Excel file
df = pd.read_excel("data_ttf.xlsx")

# Reshape to long format
df_long = df.melt(id_vars=["Machine"], var_name="component", value_name="value")

# Drop missing values
df_long = df_long.dropna(subset=["value"])

# Connect to DB
conn = sqlite3.connect("ttf.db")
cursor = conn.cursor()

# Recreate tables with machine column
cursor.execute("DROP TABLE IF EXISTS ttf_data")
cursor.execute("DROP TABLE IF EXISTS hmr_data")

cursor.execute("""
CREATE TABLE ttf_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine TEXT NOT NULL,
    component TEXT NOT NULL,
    value REAL NOT NULL
)
""")

cursor.execute("""
CREATE TABLE hmr_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine TEXT,
    component TEXT,
    hmr_value REAL,
    failure_description TEXT,
    working_hours REAL,
    maintenance_hours REAL,
    breakdown_hours REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Insert into ttf_data
for _, row in df_long.iterrows():
    cursor.execute("INSERT INTO ttf_data (machine, component, value) VALUES (?, ?, ?)",
                   (row['Machine'], row['component'], row['value']))

conn.commit()
conn.close()
print("✅ Database initialized with machine, component, and value.")
