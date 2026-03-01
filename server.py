from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from fastmcp import FastMCP
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


# Configuration
mcp = FastMCP("obspy-mcp")
FDSN_PROVIDER = "IRIS"  # Data center: IRIS, USGS, EMSC, GFZ, etc

# ═══════════════════════════════════════════════════════════════
# TeslaQuake Dual-Frequency Configuration
# Schumann Resonance (7.83 Hz) + Tesla Telluric (11.78 Hz)
# ═══════════════════════════════════════════════════════════════
TESLAQUAKE_FREQUENCIES = {
    "sr1": {"freq": 7.83, "label": "Schumann Resonance SR₁", "tolerance": 0.5},
    "sr2": {"freq": 14.3, "label": "Schumann Resonance SR₂", "tolerance": 0.5},
    "sr3": {"freq": 20.8, "label": "Schumann Resonance SR₃", "tolerance": 0.5},
    "sr4": {"freq": 26.4, "label": "Schumann Resonance SR₄", "tolerance": 0.5},
    "sr5": {"freq": 33.8, "label": "Schumann Resonance SR₅", "tolerance": 0.5},
    "tesla": {"freq": 11.78, "label": "Tesla Telluric Frequency", "tolerance": 0.5},
    "tesla_369": {"freq": 23.5, "label": "Tesla 3-6-9 Harmonic", "tolerance": 0.5},
}

# Anomaly thresholds (standard deviations above baseline)
ANOMALY_THRESHOLD_MODERATE = 2.0
ANOMALY_THRESHOLD_HIGH = 3.0
ANOMALY_THRESHOLD_CRITICAL = 4.0


# Helpers
def _fdsn() -> Client:
    # Get FDSN client for querying seismic data
    return Client(FDSN_PROVIDER)


def default_data_dir() -> Path:
    # Downloads go to ./obspy_downloads/ inside project
    return Path(__file__).parent / "obspy_downloads"


def ensure_writable_dir(path: Path) -> Path:
    # Create directory and verify write access
    path.mkdir(parents=True, exist_ok=True)
    test = path / ".write_test"
    try:
        test.write_text("ok")
        test.unlink(missing_ok=True)
    except Exception as e:
        raise PermissionError(f"Cannot write to {path}: {e}")
    return path


def safe_folder_name(s: str) -> str:
    # Sanitize string for use as folder name
    return s.replace(":", "-").replace("/", "-").replace(" ", "_")


def file_proof(folder: Path) -> Dict[str, Any]:
    # Return proof of saved files (names + sizes) to prevent hallucination
    files = []
    total = 0
    for p in sorted(folder.glob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            files.append({"name": p.name, "size": f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"})
    return {"folder": str(folder), "file_count": len(files), "total_size": f"{total/1024:.1f} KB", "files": files}


# ═══════════════════════════════════════════════════════════════
# FFT / Spectral Analysis Helpers
# ═══════════════════════════════════════════════════════════════
def _compute_fft(data: np.ndarray, sampling_rate: float, freq_min: float = 0.1, freq_max: float = 50.0):
    """Compute FFT and return frequencies + amplitudes within band."""
    n = len(data)
    # Remove mean (detrend)
    data = data - np.mean(data)
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    data_windowed = data * window
    # FFT
    fft_vals = np.fft.rfft(data_windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    amplitudes = np.abs(fft_vals) * 2.0 / n  # Normalize
    # Filter to frequency band of interest
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    return freqs[mask], amplitudes[mask]


def _find_peaks(freqs: np.ndarray, amps: np.ndarray, num_peaks: int = 10, min_prominence: float = 0.0):
    """Find top N frequency peaks by amplitude."""
    # Simple peak detection: local maxima
    peaks = []
    for i in range(1, len(amps) - 1):
        if amps[i] > amps[i-1] and amps[i] > amps[i+1] and amps[i] > min_prominence:
            peaks.append({"freq_hz": round(float(freqs[i]), 3), "amplitude": round(float(amps[i]), 6)})
    # Sort by amplitude descending
    peaks.sort(key=lambda x: x["amplitude"], reverse=True)
    return peaks[:num_peaks]


def _check_teslaquake_bands(freqs: np.ndarray, amps: np.ndarray):
    """Check amplitude at each TeslaQuake monitoring frequency."""
    results = {}
    # Compute baseline stats (mean + std of full spectrum)
    mean_amp = float(np.mean(amps))
    std_amp = float(np.std(amps))
    
    for key, config in TESLAQUAKE_FREQUENCIES.items():
        target = config["freq"]
        tol = config["tolerance"]
        # Find amplitude in the frequency band
        band_mask = (freqs >= target - tol) & (freqs <= target + tol)
        if np.any(band_mask):
            band_amps = amps[band_mask]
            band_freqs = freqs[band_mask]
            peak_idx = np.argmax(band_amps)
            peak_amp = float(band_amps[peak_idx])
            peak_freq = float(band_freqs[peak_idx])
            # Z-score: how many standard deviations above mean
            z_score = (peak_amp - mean_amp) / std_amp if std_amp > 0 else 0.0
            # Anomaly classification
            if z_score >= ANOMALY_THRESHOLD_CRITICAL:
                status = "CRITICAL"
            elif z_score >= ANOMALY_THRESHOLD_HIGH:
                status = "HIGH"
            elif z_score >= ANOMALY_THRESHOLD_MODERATE:
                status = "MODERATE"
            else:
                status = "NORMAL"
            results[key] = {
                "label": config["label"],
                "target_freq": target,
                "detected_freq": round(peak_freq, 3),
                "amplitude": round(peak_amp, 6),
                "z_score": round(z_score, 2),
                "status": status,
                "freq_shift": round(peak_freq - target, 3),
            }
        else:
            results[key] = {
                "label": config["label"],
                "target_freq": target,
                "detected_freq": None,
                "amplitude": 0.0,
                "z_score": 0.0,
                "status": "NO_DATA",
                "freq_shift": None,
            }
    return results, {"mean": round(mean_amp, 6), "std": round(std_amp, 6)}


# Tool: Health Check
@mcp.tool
def diagnose_environment(test_provider: str = "IRIS") -> Dict[str, Any]:
    # Verify filesystem is writable and FDSN service is reachable
    info = {"timestamp": datetime.now(timezone.utc).isoformat()}
    
    # Check filesystem
    try:
        info["writable_dir"] = str(ensure_writable_dir(default_data_dir()))
        info["writable"] = True
    except Exception as e:
        info["writable"], info["fs_error"] = False, str(e)

    # Check network
    try:
        Client(test_provider).get_stations(network="IU", station="ANMO", level="network")
        info["fdsn_ok"] = True
    except Exception as e:
        info["fdsn_ok"], info["fdsn_error"] = False, str(e)

    ok = info.get("writable", False) and info.get("fdsn_ok", False)
    info["status"] = "READY" if ok else "NOT READY"
    return {"ok": ok, "diagnostic": info}


# Tool: Find Earthquakes
@mcp.tool
def find_recent_m7(days: int = 30, limit: int = 5) -> Dict[str, Any]:
    # Search FDSN catalog for recent M7+ earthquakes
    # Returns: time, lat/lon, depth, magnitude for each event
    try:
        end = UTCDateTime(datetime.now(timezone.utc))
        cat = _fdsn().get_events(starttime=end - (days * 86400), endtime=end, minmagnitude=7.0, limit=limit)
        
        events = []
        for ev in cat:
            org = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            if org and mag:
                events.append({
                    "time_utc": str(org.time),
                    "latitude": float(org.latitude),
                    "longitude": float(org.longitude),
                    "depth_km": float(org.depth / 1000) if org.depth else None,
                    "magnitude": float(mag.mag) if mag.mag else None,
                    "magnitude_type": getattr(mag, "magnitude_type", None),
                })
        return {"ok": True, "event_count": len(events), "events": events}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tool: Find Stations
@mcp.tool
def get_nearby_stations(latitude: float, longitude: float, maxradius_deg: float = 2.0, channel: str = "BH?") -> Dict[str, Any]:
    # Find seismic stations within radius of a point
    # 1 degree ≈ 111 km. BH? = broadband high-gain (all components)
    try:
        inv = _fdsn().get_stations(latitude=latitude, longitude=longitude, maxradius=maxradius_deg, channel=channel, level="station")
        stations = [{"network": net.code, "station": sta.code, "latitude": float(sta.latitude), "longitude": float(sta.longitude)} 
                    for net in inv for sta in net]
        return {"ok": True, "station_count": len(stations), "stations": stations}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tool: Download Waveforms (Core Implementation)
def _download_waveforms_impl(network: str, station: str, latitude: float, longitude: float, origin_time_utc: str,
                              magnitude: float = None, depth_km: float = None, location: str = "*", channel: str = "BH?",
                              pre_seconds: int = 120, post_seconds: int = 1200, output_dir: str = "", make_plot: bool = True) -> Dict[str, Any]:
    # Core download logic
    # Saves:
    # 1. waveforms.mseed  : Raw seismogram data
    # 2. station.xml      : Instrument response metadata
    # 3. event.json       : Earthquake parameters
    # 4. quickplot.png    : Waveform visualization
    try:
        if not origin_time_utc:
            return {"ok": False, "error": "origin_time_utc is required"}

        client = _fdsn()
        t0 = UTCDateTime(origin_time_utc)
        
        # Setup output folder
        base = Path(output_dir).expanduser() if output_dir else default_data_dir()
        base = ensure_writable_dir(base)
        folder = base / f"{safe_folder_name(str(t0))}_{network}.{station}"
        folder.mkdir(parents=True, exist_ok=True)

        # Fetch waveforms (time window: origin ± pre/post seconds)
        st = client.get_waveforms(network, station, location, channel, t0 - pre_seconds, t0 + post_seconds)
        if len(st) == 0:
            return {"ok": False, "error": "No waveform data returned"}

        # Fetch station metadata (optional, for instrument response)
        inv = None
        try:
            inv = client.get_stations(network=network, station=station, location=location, channel=channel, level="response")
        except:
            pass

        # Save waveforms (MiniSEED format)
        mseed_path = folder / "waveforms.mseed"
        st.write(str(mseed_path), format="MSEED")

        # Save station metadata (StationXML format)
        stationxml_path = None
        if inv:
            stationxml_path = folder / "station.xml"
            inv.write(str(stationxml_path), format="STATIONXML")

        # Save event info (JSON)
        event_path = folder / "event.json"
        event_path.write_text(json.dumps({
            "time_utc": origin_time_utc, "latitude": latitude, "longitude": longitude,
            "magnitude": magnitude, "depth_km": depth_km, "network": network,
            "station": station, "channel": channel, "fdsn_provider": FDSN_PROVIDER,
        }, indent=2))

        # Generate plot (PNG)
        plot_path = None
        if make_plot:
            try:
                plot_path = folder / "quickplot.png"
                st.plot(outfile=str(plot_path))
            except:
                pass

        # Build response with trace info
        traces = [{"id": tr.id, "starttime": str(tr.stats.starttime), "endtime": str(tr.stats.endtime),
                   "sampling_rate": float(tr.stats.sampling_rate), "npts": int(tr.stats.npts)} for tr in st]

        return {
            "ok": True, "saved_folder": str(folder),
            "waveforms_mseed": str(mseed_path),
            "station_xml": str(stationxml_path) if stationxml_path else None,
            "event_json": str(event_path),
            "quickplot_png": str(plot_path) if plot_path else None,
            "trace_count": len(st), "traces": traces,
            "proof": file_proof(folder),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool
def download_waveforms(network: str, station: str, latitude: float, longitude: float, origin_time_utc: str,
                       magnitude: float = None, depth_km: float = None, location: str = "*", channel: str = "BH?",
                       pre_seconds: int = 120, post_seconds: int = 1200, output_dir: str = "", make_plot: bool = True) -> Dict[str, Any]:
    # Download waveforms for an earthquake and save to disk
    #
    # Parameters:
    #     network, station: e.g., "IU", "ANMO"
    #     latitude, longitude: Earthquake epicenter
    #     origin_time_utc: e.g., "2025-12-01T12:34:56"
    #     pre_seconds: Time before origin (default: 2 min)
    #     post_seconds: Time after origin (default: 20 min)
    return _download_waveforms_impl(network=network, station=station, latitude=latitude, longitude=longitude,
                                     origin_time_utc=origin_time_utc, magnitude=magnitude, depth_km=depth_km,
                                     location=location, channel=channel, pre_seconds=pre_seconds,
                                     post_seconds=post_seconds, output_dir=output_dir, make_plot=make_plot)


# ═══════════════════════════════════════════════════════════════
# Tool: Spectral Analysis (FFT)
# ═══════════════════════════════════════════════════════════════
@mcp.tool
def spectral_analysis(mseed_path: str, freq_min: float = 0.1, freq_max: float = 50.0,
                      num_peaks: int = 10, trace_index: int = 0,
                      save_plot: bool = True) -> Dict[str, Any]:
    """Perform FFT spectral analysis on a downloaded waveform file.
    
    Parameters:
        mseed_path: Path to .mseed waveform file
        freq_min: Minimum frequency to analyze (Hz)
        freq_max: Maximum frequency to analyze (Hz)
        num_peaks: Number of top frequency peaks to return
        trace_index: Which trace to analyze (0 = first)
        save_plot: Save a frequency spectrum PNG
    
    Returns:
        Top frequency peaks, spectral statistics, and optional plot path
    """
    try:
        st = read(mseed_path)
        if trace_index >= len(st):
            return {"ok": False, "error": f"Trace index {trace_index} out of range (have {len(st)} traces)"}
        
        tr = st[trace_index]
        data = tr.data.astype(float)
        sr = tr.stats.sampling_rate
        
        # Nyquist check
        nyquist = sr / 2.0
        if freq_max > nyquist:
            freq_max = nyquist * 0.95
        
        # Compute FFT
        freqs, amps = _compute_fft(data, sr, freq_min, freq_max)
        
        # Find peaks
        peaks = _find_peaks(freqs, amps, num_peaks)
        
        # Stats
        stats = {
            "trace_id": tr.id,
            "sampling_rate": float(sr),
            "nyquist_freq": float(nyquist),
            "duration_seconds": float(tr.stats.npts / sr),
            "freq_resolution": round(float(sr / tr.stats.npts), 4),
            "analyzed_band": [freq_min, freq_max],
            "mean_amplitude": round(float(np.mean(amps)), 6),
            "max_amplitude": round(float(np.max(amps)), 6),
            "dominant_freq": peaks[0]["freq_hz"] if peaks else None,
        }
        
        # Save spectrum plot
        plot_path = None
        if save_plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.semilogy(freqs, amps, color="#3498DB", linewidth=0.8, alpha=0.9)
                
                # Mark TeslaQuake frequencies
                colors = {"sr1": "#C49A3C", "sr2": "#C49A3C", "sr3": "#C49A3C",
                          "sr4": "#C49A3C", "sr5": "#C49A3C",
                          "tesla": "#FF4444", "tesla_369": "#FF8800"}
                for key, config in TESLAQUAKE_FREQUENCIES.items():
                    if freq_min <= config["freq"] <= freq_max:
                        ax.axvline(config["freq"], color=colors.get(key, "#888"),
                                   linestyle="--", alpha=0.6, linewidth=1)
                        ax.text(config["freq"], ax.get_ylim()[1] * 0.7,
                                f" {config['freq']} Hz", fontsize=7, color=colors.get(key, "#888"),
                                rotation=90, va="top")
                
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Amplitude (log scale)")
                ax.set_title(f"Spectral Analysis: {tr.id}")
                ax.set_xlim(freq_min, freq_max)
                ax.grid(True, alpha=0.3)
                
                out_dir = Path(mseed_path).parent
                plot_path = str(out_dir / "spectrum.png")
                fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#111214")
                plt.close(fig)
            except Exception as e:
                plot_path = f"plot_error: {e}"
        
        return {
            "ok": True,
            "peaks": peaks,
            "stats": stats,
            "spectrum_plot": plot_path,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Tool: TeslaQuake Dual-Frequency Analysis
# ═══════════════════════════════════════════════════════════════
@mcp.tool
def analyze_teslaquake_frequencies(mseed_path: str, trace_index: int = 0,
                                    save_plot: bool = True) -> Dict[str, Any]:
    """Analyze waveform for TeslaQuake signature frequencies.
    
    Monitors:
        - 7.83 Hz  : Schumann Resonance fundamental (SR₁)
        - 11.78 Hz : Tesla Telluric Frequency
        - 14.3 Hz  : SR₂
        - 20.8 Hz  : SR₃
        - 23.5 Hz  : Tesla 3-6-9 Harmonic
        - 26.4 Hz  : SR₄
        - 33.8 Hz  : SR₅
    
    Returns amplitude, z-score, and anomaly status for each frequency.
    Frequency shifts from expected values may indicate precursor activity.
    
    Parameters:
        mseed_path: Path to .mseed waveform file
        trace_index: Which trace to analyze (0 = first)
        save_plot: Save annotated spectrum PNG with TeslaQuake bands highlighted
    """
    try:
        st = read(mseed_path)
        if trace_index >= len(st):
            return {"ok": False, "error": f"Trace index {trace_index} out of range (have {len(st)} traces)"}
        
        tr = st[trace_index]
        data = tr.data.astype(float)
        sr = tr.stats.sampling_rate
        
        # Need at least 40 Hz sampling to see all SR harmonics
        nyquist = sr / 2.0
        freq_max = min(40.0, nyquist * 0.95)
        
        # Compute FFT
        freqs, amps = _compute_fft(data, sr, 1.0, freq_max)
        
        # Analyze TeslaQuake bands
        band_results, baseline_stats = _check_teslaquake_bands(freqs, amps)
        
        # Overall anomaly assessment
        anomalies = [k for k, v in band_results.items() if v["status"] in ("HIGH", "CRITICAL")]
        moderate = [k for k, v in band_results.items() if v["status"] == "MODERATE"]
        
        if anomalies:
            overall = "⚠️ ANOMALY DETECTED"
            alert_level = "HIGH" if any(band_results[k]["status"] == "CRITICAL" for k in anomalies) else "ELEVATED"
        elif moderate:
            overall = "📡 ELEVATED ACTIVITY"
            alert_level = "MODERATE"
        else:
            overall = "✅ NORMAL"
            alert_level = "NORMAL"
        
        # Check for frequency shifts (precursor indicator)
        sr1_shift = band_results.get("sr1", {}).get("freq_shift")
        tesla_shift = band_results.get("tesla", {}).get("freq_shift")
        precursor_flags = []
        if sr1_shift is not None and abs(sr1_shift) > 0.15:
            precursor_flags.append(f"SR₁ shifted {sr1_shift:+.3f} Hz from 7.83 Hz baseline")
        if tesla_shift is not None and abs(tesla_shift) > 0.15:
            precursor_flags.append(f"Tesla shifted {tesla_shift:+.3f} Hz from 11.78 Hz baseline")
        
        # Save annotated plot
        plot_path = None
        if save_plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
                fig.patch.set_facecolor("#111214")
                
                # Top: Full spectrum with TeslaQuake bands
                ax1.semilogy(freqs, amps, color="#3498DB", linewidth=0.8, alpha=0.9)
                
                status_colors = {"NORMAL": "#2ECC71", "MODERATE": "#F39C12",
                                 "HIGH": "#E74C3C", "CRITICAL": "#FF0000", "NO_DATA": "#555"}
                
                for key, result in band_results.items():
                    color = status_colors.get(result["status"], "#888")
                    freq = result["target_freq"]
                    if freq <= freq_max:
                        ax1.axvline(freq, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
                        ax1.text(freq + 0.1, ax1.get_ylim()[1] * 0.5 if ax1.get_ylim()[1] > 0 else 1,
                                 f"{result['label']}\n{freq} Hz\nz={result['z_score']}",
                                 fontsize=6, color=color, va="top")
                
                ax1.set_ylabel("Amplitude (log)", color="#9CA3AF")
                ax1.set_title(f"TeslaQuake Frequency Analysis: {tr.id}  |  {overall}",
                              color="#C49A3C", fontsize=12, fontweight="bold")
                ax1.set_xlim(1, freq_max)
                ax1.grid(True, alpha=0.2)
                ax1.set_facecolor("#1A1A2E")
                ax1.tick_params(colors="#9CA3AF")
                
                # Bottom: Bar chart of z-scores per band
                band_names = [v["label"].replace("Schumann Resonance ", "")
                              for v in band_results.values() if v["target_freq"] <= freq_max]
                z_scores = [v["z_score"] for v in band_results.values() if v["target_freq"] <= freq_max]
                bar_colors = [status_colors.get(v["status"], "#888")
                              for v in band_results.values() if v["target_freq"] <= freq_max]
                
                ax2.bar(band_names, z_scores, color=bar_colors, alpha=0.85)
                ax2.axhline(ANOMALY_THRESHOLD_MODERATE, color="#F39C12", linestyle=":", alpha=0.5, label="Moderate")
                ax2.axhline(ANOMALY_THRESHOLD_HIGH, color="#E74C3C", linestyle=":", alpha=0.5, label="High")
                ax2.set_ylabel("Z-Score", color="#9CA3AF")
                ax2.set_facecolor("#1A1A2E")
                ax2.tick_params(colors="#9CA3AF", axis="both")
                ax2.legend(fontsize=7, loc="upper right")
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
                
                plt.tight_layout()
                out_dir = Path(mseed_path).parent
                plot_path = str(out_dir / "teslaquake_analysis.png")
                fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#111214")
                plt.close(fig)
            except Exception as e:
                plot_path = f"plot_error: {e}"
        
        return {
            "ok": True,
            "overall_status": overall,
            "alert_level": alert_level,
            "trace_id": tr.id,
            "sampling_rate": float(sr),
            "duration_seconds": float(tr.stats.npts / sr),
            "frequencies": band_results,
            "baseline": baseline_stats,
            "anomalies": anomalies,
            "precursor_flags": precursor_flags,
            "teslaquake_plot": plot_path,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tool: Arrival Times
@mcp.tool
def estimate_arrival_times(distance_deg: float, depth_km: float = 10.0, phases: List[str] = None) -> Dict[str, Any]:
    # Calculate seismic phase arrival times using TauP (IASP91 Earth model)
    #
    # Common phases:
    # - P, S: Direct body waves
    # - pP, sS: Surface reflections
    # - PKP, SKS: Core phases (teleseismic)
    try:
        if phases is None:
            phases = ["p", "s", "P", "S", "Pn", "Sn"] if distance_deg < 10 else ["P", "S", "pP", "sS", "PP", "SS", "PKP", "SKS"]
        
        model = TauPyModel(model="iasp91")
        arrivals = model.get_travel_times(source_depth_in_km=depth_km, distance_in_degree=distance_deg, phase_list=phases)
        out = [{"phase": arr.name, "time_seconds": float(arr.time)} for arr in arrivals]
        return {"ok": True, "distance_deg": distance_deg, "depth_km": depth_km, "arrivals": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tool: Distance Calculator
@mcp.tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, Any]:
    # Great-circle distance between two points
    # Returns: distance_km, distance_deg, azimuth, back_azimuth
    try:
        from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
        dist_m, az, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        dist_km = dist_m / 1000.0
        return {"ok": True, "distance_km": dist_km, "distance_deg": kilometers2degrees(dist_km), "azimuth": az, "back_azimuth": baz}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tool: One-Shot Automatic Study
@mcp.tool
def auto_download_study(days: int = 30, min_magnitude: float = 7.0, maxradius_deg: float = 2.0, channel: str = "BH?",
                        pre_seconds: int = 120, post_seconds: int = 1200, output_dir: str = "") -> Dict[str, Any]:
    # Complete automated workflow:
    # 1. Find recent earthquake (M7+ by default)
    # 2. Locate nearby seismic station
    # 3. Download waveforms + metadata
    # 4. Save all files to disk
    #
    # Returns event info, station used, and proof of saved files
    try:
        client = _fdsn()
        output_dir = output_dir or str(default_data_dir())
        end = UTCDateTime()
        
        # Step 1: Find earthquake
        cat = client.get_events(starttime=end - (days * 86400), endtime=end, minmagnitude=min_magnitude, limit=10)
        
        chosen = None
        for ev in cat:
            org = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            if org and mag:
                chosen = (org, mag)
                break
        
        if not chosen:
            return {"ok": False, "error": f"No M{min_magnitude}+ events in last {days} days"}

        org, mag = chosen
        ev_lat, ev_lon = float(org.latitude), float(org.longitude)
        depth_km = float(org.depth / 1000) if org.depth else None
        origin_time = str(org.time)
        magnitude = float(mag.mag) if mag.mag else None
        event_info = {"time_utc": origin_time, "latitude": ev_lat, "longitude": ev_lon,
                      "magnitude": magnitude, "magnitude_type": getattr(mag, "magnitude_type", None), "depth_km": depth_km}

        # Step 2: Find nearby station
        inv = client.get_stations(latitude=ev_lat, longitude=ev_lon, maxradius=maxradius_deg, channel=channel, level="station")
        stations = [(net.code, sta.code, float(sta.latitude), float(sta.longitude)) for net in inv for sta in net]
        
        if not stations:
            return {"ok": False, "event": event_info, "error": f"No stations within {maxradius_deg}°"}

        net, sta, st_lat, st_lon = stations[0]
        station_info = {"network": net, "station": sta, "latitude": st_lat, "longitude": st_lon}

        # Step 3: Download waveforms
        result = _download_waveforms_impl(network=net, station=sta, latitude=ev_lat, longitude=ev_lon,
                                           origin_time_utc=origin_time, magnitude=magnitude, depth_km=depth_km,
                                           channel=channel, pre_seconds=pre_seconds, post_seconds=post_seconds,
                                           output_dir=output_dir, make_plot=True)
        
        # Add context
        result["event"] = event_info
        result["station_chosen"] = station_info
        result["stations_available"] = len(stations)
        return result

    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
