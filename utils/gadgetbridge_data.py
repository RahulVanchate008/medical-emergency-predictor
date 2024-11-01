import sqlite3
import pandas as pd
import time

db_path = "C:/Users/local_user/Downloads/Gadgetbridge.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

query = """
SELECT HEART_RATE, TIMESTAMP
FROM HUAMI_EXTENDED_ACTIVITY_SAMPLE
WHERE HEART_RATE != 0 AND HEART_RATE != 255
ORDER BY TIMESTAMP DESC
LIMIT 60
"""

cursor.execute(query)
rows = cursor.fetchall()

conn.close()

df = pd.DataFrame(rows, columns=['heart_rate', 'timestamp'])
df = df.iloc[::-1]  

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

csv_path = "/latest_heart_rate_data.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved as {csv_path}")
