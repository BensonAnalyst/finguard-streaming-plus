import csv
import os
from pathlib import Path

files = [
    Path('data/customers.csv'),
    Path('data/merchants.csv'),
]

ALERT_EMAIL = os.getenv('ALERT_EMAIL', 'alerts@example.com')

for path in files:
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['email']
        rows = list(reader)

    for row in rows:
        row['email'] = ALERT_EMAIL

    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Updated {path} with {len(rows)} rows; header={fieldnames}')
