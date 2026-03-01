# ObsPy MCP Server — TeslaQuake Edition

MCP server enabling Claude Desktop to download earthquake data, analyze waveform frequencies, and detect electromagnetic precursors via ObsPy.

**Forked from [Fahim-Azwad/simple-obspy](https://github.com/Fahim-Azwad/simple-obspy)** (UC Berkeley) and customized for [TeslaQuake](https://teslaquake.com) dual-frequency earthquake prediction research.

## TeslaQuake Additions

This fork adds spectral analysis tools for monitoring electromagnetic earthquake precursors:

- **FFT Spectral Analysis** — Extract frequency peaks from any seismic waveform
- **Dual-Frequency Monitor** — Track Schumann Resonance (7.83 Hz) + Tesla Telluric (11.78 Hz)
- **Anomaly Detection** — Z-score based alerting with NORMAL → CRITICAL severity levels
- **Supabase Integration** — Push findings directly to TeslaQuake's anomaly detection database
- **Precursor Flagging** — Detect frequency shifts that may indicate impending seismic activity

### Monitored Frequencies

| Band | Frequency | Significance |
|------|-----------|-------------|
| SR₁ | 7.83 Hz | Schumann Resonance fundamental |
| Tesla | 11.78 Hz | Tesla Telluric Frequency |
| SR₂ | 14.3 Hz | Schumann 2nd harmonic |
| SR₃ | 20.8 Hz | Schumann 3rd harmonic |
| Tesla 3-6-9 | 23.5 Hz | Tesla harmonic (discovered relationship) |
| SR₄ | 26.4 Hz | Schumann 4th harmonic |
| SR₅ | 33.8 Hz | Schumann 5th harmonic |

## Requirements

- **Python 3.9+**
- **obspy** — Seismic data processing
- **fastmcp** — MCP server framework
- **numpy** — FFT spectral analysis
- **matplotlib** — Spectrum visualization

## Quick Start

```bash
git clone https://github.com/Brianleewestfall/simple-obspy.git
cd simple-obspy

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure Claude Desktop

Go to **Claude → Settings → Developer → Edit Config** and add:

```json
{
  "mcpServers": {
    "obspy-teslaquake": {
      "command": "/FULL/PATH/TO/simple-obspy/.venv/bin/python",
      "args": ["/FULL/PATH/TO/simple-obspy/server.py"],
      "env": {
        "TESLAQUAKE_SUPABASE_URL": "https://your-project.supabase.co",
        "TESLAQUAKE_SUPABASE_KEY": "your-service-role-key"
      }
    }
  }
}
```

> Replace paths and Supabase credentials with your actual values. Supabase env vars are optional — spectral analysis works without them.

Restart Claude Desktop after editing config.

---

## Tools

### Original (from upstream)

| Tool | Purpose |
|------|---------|
| `diagnose_environment` | Health check (filesystem + FDSN + Supabase) |
| `find_recent_m7` | Search earthquake catalog |
| `get_nearby_stations` | Find stations near epicenter |
| `download_waveforms` | Download & save seismic data |
| `estimate_arrival_times` | Calculate P/S wave arrivals |
| `calculate_distance` | Great-circle distance |
| `auto_download_study` | One-shot automated workflow |

### TeslaQuake Additions

| Tool | Purpose |
|------|---------|
| `spectral_analysis` | FFT on any waveform — top peaks + spectrum PNG |
| `analyze_teslaquake_frequencies` | Dual-frequency monitor with anomaly detection |
| `push_to_supabase` | Write anomalies + baselines to TeslaQuake DB |

---

## TeslaQuake Workflow

```
1. "Find M6+ earthquakes from the last 7 days"
2. "Download waveforms from the closest station"
3. "Run analyze_teslaquake_frequencies on the waveforms"
4. "Push results to Supabase"
```

Or ask Claude to chain them:

```
Find a recent M6+ earthquake, download waveforms from a nearby station,
analyze for TeslaQuake frequencies, and push any anomalies to Supabase.
```

### Prompt Examples — TeslaQuake Tools

```
Run spectral_analysis on ./obspy_downloads/2025-12-08/waveforms.mseed
```
```
Analyze TeslaQuake frequencies on the downloaded waveforms — check for SR and Tesla anomalies
```
```
Push the analysis results to Supabase with station IU.ANMO and context "Pre-event scan Alaska"
```

---

## Output Files

Downloads saved to `./obspy_downloads/`:

```
obspy_downloads/2025-12-08T14-15-10Z_II.ERM/
├── waveforms.mseed          # Seismogram data (MiniSEED)
├── station.xml              # Instrument response (StationXML)
├── event.json               # Earthquake parameters
├── quickplot.png            # Waveform visualization
├── spectrum.png             # FFT frequency spectrum
└── teslaquake_analysis.png  # Dual-frequency analysis + z-score chart
```

---

## Supabase Tables Used

| Table | Purpose |
|-------|---------|
| `anomaly_detections` | Frequency bands with z-score ≥ 2.0 |
| `welford_baselines` | Running mean/variance for each frequency band |

Metric names: `obspy_sr1_amplitude`, `obspy_tesla_amplitude`, `obspy_sr2_amplitude`, etc.

---

## References

- [ObsPy Documentation](https://docs.obspy.org/)
- [IRIS Web Services](https://service.iris.edu/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [TeslaQuake Research](https://teslaquake.com)

## Acknowledgments

Original server developed at the request of **Professor Weiqiang Zhu**, Earth & Planetary Science, UC Berkeley.

TeslaQuake customization by **Brian Lee Westfall** — AI Vision Designs, Fort Worth, TX.
