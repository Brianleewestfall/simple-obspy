# ObsPy MCP Server

MCP server enabling Claude Desktop to download earthquake and waveform data via ObsPy.

This server exposes ObsPy functionality as MCP tools, enabling Claude to:
- Search earthquake catalogs (FDSN)
- Find nearby seismic stations
- Download waveforms, metadata, and plots
- Estimate phase arrival times (TauP)
## Requirements

- **Python 3.9+**
- **obspy** - Seismic data processing library
- **fastmcp** - MCP server framework

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Fahim-Azwad/simple-obspy.git
cd simple-obspy

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Claude Desktop

Go to **Claude → Settings → Developer → Edit Config** and add:

```json
{
  "mcpServers": {
    "obspy-mcp": {
      "command": "/FULL/PATH/TO/simple-obspy/.venv/bin/python",
      "args": ["/FULL/PATH/TO/simple-obspy/server.py"]
    }
  }
}
```

> **Note:** Replace `/FULL/PATH/TO/simple-obspy` with your actual path (e.g., `/Users/Fahim-Azwad/simple-obspy`)

Restart Claude Desktop (Cmd+Q → reopen).

---

## Tools

| Tool | Purpose |
|------|---------|
| `diagnose_environment` | Health check |
| `find_recent_m7` | Search earthquake catalog |
| `get_nearby_stations` | Find stations near epicenter |
| `download_waveforms` | Download & save seismic data |
| `estimate_arrival_times` | Calculate P/S wave arrivals |
| `calculate_distance` | Great-circle distance |
| `auto_download_study` | One-shot automated workflow |

---

## Prompt Examples

### 🏥 Health Check
```
Run diagnose_environment
```

### 🌍 Find Earthquakes
```
Find recent M7+ earthquakes from the last 30 days
```
```
Find M6+ earthquakes from the last 90 days, limit 10
```
```
Search for large earthquakes in the past 2 months
```

### 📡 Find Stations
```
Find seismic stations within 3 degrees of latitude 35.0, longitude 139.0
```
```
What broadband stations are near the Japan earthquake epicenter?
```
```
List stations within 500 km of 41.0, 142.0
```

### 📥 Download Waveforms
```
Download waveforms from station IU.MAJO for the earthquake at 2025-12-08T14:15:10Z, epicenter 41.0, 142.0
```
```
Get seismic data from II.ERM for the M7.6 Japan earthquake, include 5 minutes before and 30 minutes after
```
```
Download BHZ channel data from network IU, station ANMO for origin time 2025-12-08T14:15:10
```

### ⏱️ Arrival Times
```
Estimate P and S wave arrival times for an earthquake 45 degrees away at 30 km depth
```
```
What are the expected phase arrivals for a teleseismic event at 80 degrees distance?
```
```
Calculate when P, S, and PKP waves arrive for distance 120 degrees, depth 600 km
```

### 📏 Distance
```
What is the distance from Tokyo (35.68, 139.65) to Los Angeles (34.05, -118.24)?
```
```
Calculate distance between earthquake at 41.0, 142.0 and station at 42.0, 143.0
```

### 🚀 One-Shot Automatic Study (Most Powerful)
```
Use auto_download_study for the last 60 days, M7+, 5 degree radius
```
```
Run an automatic earthquake study: find a recent M6.5+ event, locate nearby stations, download waveforms
```
```
Automatically download seismic data for any M7+ earthquake in the past 30 days
```
```
Do a complete seismic study - find earthquake, pick station, download data, show me the files
```
```
auto_download_study with min_magnitude 6.0, days 90, maxradius 10 degrees
```

### 🔬 Advanced Workflows
```
1. Find M7+ earthquakes from last 60 days
2. For the first one, find stations within 5 degrees
3. Download waveforms from the closest station
4. Calculate distance and estimate P/S arrival times
```
```
Find the recent Japan M7.6 earthquake, download data from station II.ERM, then tell me when the P and S waves should arrive
```
```
Search for deep earthquakes (depth > 300 km) in the last 90 days, download waveforms from a nearby station
```

---

## Output Files

Downloads saved to `./obspy_downloads/`:

```
obspy_downloads/2025-12-08T14-15-10Z_II.ERM/
├── waveforms.mseed    # Seismogram data (MiniSEED)
├── station.xml        # Instrument response (StationXML)
├── event.json         # Earthquake parameters
└── quickplot.png      # Waveform visualization
```

---

## Channel Codes

| Code | Meaning |
|------|---------|
| `BH?` | Broadband high-gain (default) |
| `HH?` | High broadband |
| `LH?` | Long-period |
| `BHZ` | Broadband vertical only |
| `BH1`, `BH2` | Horizontal components |

---

## References

- [ObsPy Documentation](https://docs.obspy.org/)
- [IRIS Web Services](https://service.iris.edu/)
- [MCP Protocol](https://modelcontextprotocol.io/)

---

## Acknowledgments

Developed at the request of **Professor Weiqiang Zhu**, Earth & Planetary Science, UC Berkeley.
