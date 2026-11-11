# Market datasets and interchange

`hftlab.datasets` generates, validates, inspects and imports local offline files.
CSV is the native runner's interchange. Versioned binary and optional Parquet
sources are converted to canonical CSV by the job manager before launching the
native process.

## Canonical CSV v1

The exact header and field order are:

```csv
event_id,timestamp_ns,instrument_id,event_type,side,price_ticks,quantity,order_id,reference_order_id
1,1000,0,QUOTE,BUY,10000,100,0,0
2,2000,0,LIMIT,SELL,10020,10,1,0
3,3000,0,CANCEL,SELL,0,0,0,1
```

Prices are integer ticks and quantities integer units. Currency conversion and
tick monetary value are simulation conventions. Numeric fields must fit
nonnegative signed 64-bit integers. Event IDs are positive and unique.
Instrument IDs are 0..65535. Timestamp nanoseconds never decrease; equal
timestamps preserve file order. Supported event types are:

| Type | Required fields | Meaning |
|---|---|---|
| QUOTE | Positive price and quantity | Reference market mark; native replay refreshes external top-of-book liquidity around this price |
| LIMIT | Positive order ID, price, quantity | Submit a passive or crossing limit order |
| MARKET | Positive order ID, price, quantity | Submit a market order; price remains a positive reference input |
| CANCEL | Earlier LIMIT reference ID; quantity/price may be zero | Request cancellation |
| MODIFY | New positive order ID, positive price/quantity, earlier LIMIT reference | Cancel/replace a referenced order |

Sides are BUY and SELL. Non-cancel/non-modify events must have a zero reference
ID. Submitted LIMIT, MARKET and MODIFY IDs are unique. A reference must identify
an earlier LIMIT or replacement for the same instrument.

Validation uses streaming parsing and disk-backed SQLite indexes for exact ID
and reference checks. It reports counts, type distribution, instrument counts,
price extrema, total quantity, timestamps, byte size and file SHA256. Empty
files, wrong headers, fractional/missing values, invalid ranges, duplicate IDs,
decreasing timestamps and invalid references fail with a useful error.

Reference validation is historical, not a second matching engine. An earlier
limit may already have filled during strategy-dependent replay; a later cancel
then becomes an exchange rejection. Imported flow is not guaranteed to have
liquidity for every market order. The native order book, rather than the data
validator, resolves executions, partial fills and state-dependent rejections.

## HFTDATA binary v1

All integers are explicitly little endian; no native struct padding is persisted.
The 56-byte header is Python `struct` format `<8sIIQ32s`:

| Field | Bytes |
|---|---:|
| Magic `HFTDATA\0` | 8 |
| Version, exactly 1 | 4 |
| Record size, exactly 54 | 4 |
| Record count | 8 |
| SHA256 of the entire record payload | 32 |

Each 54-byte record uses `<QQIBBqQQQ`:

| Field | Encoding |
|---|---|
| event_id, timestamp_ns | uint64 each, constrained to signed-int64 maximum |
| instrument_id | uint32, constrained to 65535 |
| event_type | uint8: QUOTE=0, LIMIT=1, MARKET=2, CANCEL=3, MODIFY=4 |
| side | uint8: SELL=0, BUY=1 |
| price_ticks | int64, constrained nonnegative |
| quantity, order_id, reference_order_id | uint64 each, constrained to signed-int64 maximum |

The reader checks version, record size, a maximum 100-million record count,
exact file length and the full payload checksum before yielding records. It then
applies the canonical per-record validation. Binary export and import tests
include round-trip equality, corrupt payloads and impossible record counts.

Generate directly to `.hftdata` or call `export_binary(csv_path, binary_path)`.
Direct binary generation stages a streaming CSV before packing. This bounds RAM
but temporarily requires both representations on disk. CSV generation itself
does not allocate all event records in memory.

## Synthetic scenarios

```text
python -m hftlab generate --events 100000 --instruments 8 --seed 42 --scenario correlated --output data/generated/market.csv
python -m hftlab data validate data/generated/market.csv
python -m hftlab data inspect data/generated/market.csv
python -m hftlab data import source.hftdata --output data/processed/import.csv
```

The seed controls Python's local random generator. Output bytes are deterministic
for the same parameters and interpreter algorithm. Each dataset has a JSON
manifest containing schema version, hash, size, counts, scenario and parameters.
Creation uses a temporary file and atomic replacement.

The base mark process combines Gaussian common/idiosyncratic innovations,
regime-changing volatility, sinusoidal intraday arrival intensity and exponential
interarrival times. Scenario differences are concrete:

| Scenario | Implemented behavior |
|---|---|
| normal | Moderate volatility, weak pull toward the initial instrument price |
| high-volatility | Fivefold innovation volatility |
| high-liquidity | Tenfold mean order quantity |
| low-liquidity | One-tenth mean order quantity and doubled volatility |
| shock | A default 4% negative mark jump near the dataset midpoint |
| correlated | Shared common innovations, default correlation .75, interleaved instruments |
| stress | Interleaved instruments, larger quantities, triple volatility |
| predictable | Educational sinusoidal drift with small noise and drift-correlated event side |

Approximately 65% of events refresh quotes; remaining events add passive limits,
market orders and cancellation flow. Generator reference queues are bounded per
instrument. Prices and positive quantities are floored. The Python API additionally
accepts initial price, volatility, arrival intensity, correlation, buy imbalance,
cancellation rate and shock fraction. Quantities, spread proxies and event-flow
rules are simplified research models, not calibrated exchange microstructure.
The predictable scenario is deliberately learnable and must not be treated as
evidence of real-market profit.

Optional `.parquet` import uses installed `pyarrow` and iterates batches of 8192
rows. Its columns must contain the same canonical values. There is no network
market-data downloader or live exchange connection. Source files are preserved;
importing to the same path is rejected.
