from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

from fastmcp import FastMCP
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


# Configuration
mcp = FastMCP("obspy-mcp")
FDSN_PROVIDER = "IRIS"  # Data center: IRIS, USGS, EMSC, GFZ, etc


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
