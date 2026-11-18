Place your own historical CSV files here (Histdata, Dukascopy exports, broker exports, etc.).

Expected columns (aliases such as Date/Open/High/Low/Close are accepted):

timestamp,open,high,low,close,volume

Timestamps should be ISO-8601. If they have no timezone, the config `data.timezone` is used (default UTC).
