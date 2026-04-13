# decode-wfs — [GitHub](https://github.com/spier16/decode-wfs)

Decode binary `.WFS` (Waveform Stream) files produced by Physical Acoustics
**AEwin64** / EasyAE acoustic emission software into numpy arrays.

The AEwin64 manual describes a format using message ID 173 (0xAD), but real
streaming captures use ID 174 (0xAE) with a slightly different layout. This
library handles the empirically observed streaming format and has been verified
against real captures. Voltage values are recovered with the calibration factor
used by AEwin64's own ASCII waveform exporter (`raw_counts / 16384`).

---

## Installation

```bash
pip install decode-wfs
```

Requires Python 3.10+ and numpy.

---

## Quick start

### `decode_wfs()` — full access to every record

```python
from decode_wfs import decode_wfs

wfs = decode_wfs("capture.wfs")

print(f"Sample rate : {wfs.sample_rate_hz:,} Hz")
print(f"Channels    : {wfs.channels()}")
print(f"Records     : {len(wfs.waveforms)}")

# Iterate individual waveform records
for rec in wfs.waveforms:
    print(rec.channel, rec.samples.shape, rec.samples.max())

# Time axis for each record (includes pretrigger offset when present)
t = wfs.waveforms[0].time_axis_s   # 1-D float64 array in seconds

# Stack all records from channel 1 into a 2-D array (records × samples)
arr = wfs.to_array(channel=1)      # shape (n_records, n_samples), float64 volts
```

### `wfs_to_numpy()` — one-liner for a 2-D array

```python
from decode_wfs import wfs_to_numpy

# shape: (n_waveforms, n_samples_per_waveform), dtype float64, units volts
arr = wfs_to_numpy("capture.wfs", channel=1)

# Optionally return the per-waveform time axis as well
arr, t = wfs_to_numpy("capture.wfs", channel=1, return_time_axis=True)
```

### `load_continuous()` — single concatenated time series

Reconstructs the continuous stream from all records, preserving absolute
timing when the file contains per-record position pointers.

```python
from decode_wfs import load_continuous

samples, time, sr = load_continuous("capture.wfs", channel=1)
# samples: 1-D float64 array, volts
# time   : 1-D float64 array, seconds (negative at pretrigger)
# sr     : sample rate in Hz

# Example: compute a power spectrum
import numpy as np
freqs = np.fft.rfftfreq(len(samples), d=1 / sr)
psd   = np.abs(np.fft.rfft(samples)) ** 2
```

---

## API summary

| Function / class | Returns | When to use |
|---|---|---|
| `decode_wfs(path)` | `WFSFile` | Need per-record metadata or multi-channel data |
| `wfs_to_numpy(path)` | `ndarray` (n×m) | Just need the voltage array |
| `load_continuous(path)` | `(samples, time, sr)` | Spectral analysis on a single channel |
| `WFSFile.to_array(channel)` | `ndarray` (n×m) | After `decode_wfs`, stack records into an array |
| `WFSFile.waveform_time_axis_s()` | `ndarray` (m,) | Shared time axis for all waveform records |
| `WFSFile.channels()` | `list[int]` | Which AE channel numbers are in the file |
| `WaveformRecord.time_axis_s` | `ndarray` (m,) | Per-record time axis including pretrigger |

### Logging

`decode_wfs` uses the standard `logging` module under the logger name
`decode_wfs`. To see progress messages:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## CLI

```
decode-wfs <file.wfs> [--channel N] [--max-records N]
```

Example:

```
$ decode-wfs capture.wfs --channel 1

File          : capture.wfs
Sample rate   : 1,000,000 Hz  (1.000 MSPS)
Sample format : ADT=2  (16-bit signed samples)
Setup blocks  : 2
Pretrigger    : -256 samples
Total records : 512
Channels      : [1, 2]

  [0] ch=1  n=4096 samples  min=-0.03125000 V  max=0.04394531 V  std=0.00201416 V
  [1] ch=1  n=4096 samples  min=-0.02832031 V  max=0.03955078 V  std=0.00198364 V
  ...

numpy array shape : (256, 4096)  dtype=float64
Done.
```

---

## Notes

- **Voltage calibration** — ADC counts are divided by 16384 to match the
  voltages in AEwin64's ASCII waveform exports. This factor was determined
  empirically and may not hold for all hardware configurations.
- **Pretrigger** — when a pretrigger window is configured, `time_axis_s` and
  the `time` axis from `load_continuous` start at a negative time offset so
  that t=0 corresponds to the trigger point.
- **Multi-channel files** — records from all channels are interleaved in the
  file. Use the `channel` parameter to filter, or iterate `WFSFile.waveforms`
  and check `rec.channel`.
- **Large files** — pass `max_records=N` to any function to stop after reading
  N waveform records, useful during development.
