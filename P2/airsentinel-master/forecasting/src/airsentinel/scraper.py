"""
DPCC CAAQMS scraper.

The DPCC "Advance Search" endpoints render the requested series into an inline
Highcharts config inside the returned HTML, e.g.::

    xAxis: { categories: ['2026-07-19 00:00:00','2026-07-20 00:00:00'], ... }
    series: [{ name: 'Particulate Matter < 2.5 ug', data: [73.0,42.0] }]

We POST the same form the site's UI posts and parse those two arrays back out.

Form contract (reverse-engineered from the live site):
    POST <endpoint>?stName=<base64 station token>
    body: parameters=<code>&fDate=YYYY-MM-DD HH:MM&eDate=...&duration=<n>&graphType=Line&submit=Search
    - fDate/eDate use the site's '%Y-%m-%d %H:%M' datepicker format.
    - The window (eDate - fDate) must be <= 7 days.
    - duration is the aggregation bucket in hours (1 = hourly, 24 = daily).
    - submit MUST equal 'Search' or the server ignores the query and returns the blank form.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from . import config
from .stations import ENDPOINTS, Param, Station

_CATEGORIES_RE = re.compile(r"categories:\s*\[([^\]]*)\]")
_DATA_RE = re.compile(r"data:\s*\[([^\]]*)\]")
_DATE_FMT = "%Y-%m-%d %H:%M"

# Kept as an alias for backwards compatibility / discoverability; the real value (and the
# reasoning behind it) lives in config.SCRAPE_MAX_WINDOW_DAYS.
MAX_WINDOW_DAYS = config.SCRAPE_MAX_WINDOW_DAYS


class DpccScraper:
    def __init__(
        self,
        delay: float = config.SCRAPE_DELAY_SECONDS,
        timeout: int = config.SCRAPE_TIMEOUT_SECONDS,
        max_retries: int = config.SCRAPE_MAX_RETRIES,
    ):
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (AirSentinel research pipeline; ET AI Hackathon)",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html",
            }
        )

    # -- low level ----------------------------------------------------------------------
    def _post_window(
        self, station: Station, param: Param, start: datetime, end: datetime, duration: int
    ) -> pd.DataFrame:
        """Fetch a single <=7 day window for one station/param. Returns cols [timestamp, value]."""
        url = f"{ENDPOINTS[param.endpoint]}?stName={station.st_code}"
        body = {
            "parameters": param.code,
            "fDate": start.strftime(_DATE_FMT),
            "eDate": end.strftime(_DATE_FMT),
            "duration": str(duration),
            "graphType": "Line",
            "submit": "Search",
        }
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(url, data=body, timeout=self.timeout)
                r.raise_for_status()
                return self._parse(r.text)
            except Exception as e:  # noqa: BLE001 - network flakiness, retry
                last_err = e
                time.sleep(self.delay * attempt * 2)
        raise RuntimeError(f"failed {station.zone}/{param.column} {start:%Y-%m-%d}: {last_err}")

    @staticmethod
    def _parse(html: str) -> pd.DataFrame:
        if "Record Not found" in html:
            return pd.DataFrame(columns=["timestamp", "value"])
        cats = _CATEGORIES_RE.search(html)
        # The series `data:` is the last data array (the commented Tokyo sample is stripped by
        # the site, but guard anyway by taking the array that follows the series name).
        data_matches = _DATA_RE.findall(html)
        if not cats or not data_matches:
            return pd.DataFrame(columns=["timestamp", "value"])
        categories = [c.strip().strip("'\"") for c in cats.group(1).split(",") if c.strip()]
        values_raw = [v.strip() for v in data_matches[-1].split(",") if v.strip() != ""]
        rows = []
        for ts, raw in zip(categories, values_raw):
            try:
                val = float(raw)
            except ValueError:
                val = float("nan")
            rows.append((pd.to_datetime(ts, errors="coerce"), val))
        df = pd.DataFrame(rows, columns=["timestamp", "value"])
        return df.dropna(subset=["timestamp"])

    # -- windowing ----------------------------------------------------------------------
    @staticmethod
    def _windows(start: datetime, end: datetime):
        cur = start
        step = timedelta(days=MAX_WINDOW_DAYS)
        while cur < end:
            w_end = min(cur + step, end)
            yield cur, w_end
            cur = w_end

    def fetch_series(
        self, station: Station, param: Param, start: datetime, end: datetime, duration: int
    ) -> pd.DataFrame:
        """Fetch one station/param across an arbitrary range by chunking into <=7 day windows.

        Returns a tidy frame: [zone, param, timestamp, value].
        """
        frames = []
        for w_start, w_end in self._windows(start, end):
            df = self._post_window(station, param, w_start, w_end, duration)
            if not df.empty:
                frames.append(df)
            time.sleep(self.delay)
        if not frames:
            out = pd.DataFrame(columns=["timestamp", "value"])
        else:
            out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
        out.insert(0, "param", param.column)
        out.insert(0, "zone", station.zone)
        return out.sort_values("timestamp").reset_index(drop=True)
