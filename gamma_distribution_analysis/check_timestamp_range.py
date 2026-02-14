"""
Quick check for data timestamp range
"""
import sys
sys.path.insert(0, '/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/orderbook_construction')
from orderbook import iter_orderupdate_file
import pandas as pd

orderbook_file = "/Users/charles/Documents/Tâches/Mathématiques/EA/EA_recherche/euronextparis/EuronextParis/EuronextParis_20191001_FR0000120578/FR0000120578/OrderUpdate_20191001_FR0000120578.csv"

min_time = float('inf')
max_time = 0
count = 0

for u in iter_orderupdate_file(orderbook_file):
    count += 1
    if u.event_time < min_time:
        min_time = u.event_time
    if u.event_time > max_time:
        max_time = u.event_time

print(f"Total updates: {count}")
print(f"Min timestamp: {min_time} ns")
print(f"Max timestamp: {max_time} ns")

# Convert to readable time
min_dt = pd.to_datetime(min_time, unit='ns', utc=True)
max_dt = pd.to_datetime(max_time, unit='ns', utc=True)

print(f"Min time: {min_dt}")
print(f"Max time: {max_dt}")
