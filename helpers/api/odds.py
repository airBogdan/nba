"""Odds API client for betting lines from the-odds-api."""

import os
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

ODDS_API_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Map API-Sports team names to the-odds-api team names
TEAM_NAME_MAP = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}


def _get_api_key() -> str:
    load_dotenv()
    return os.environ.get("THE_ODDS_API", "")


def normalize_team_name(name: str) -> str:
    """Normalize team name to match the-odds-api format."""
    return TEAM_NAME_MAP.get(name, name)


async def fetch_nba_odds(
    regions: str = "us",
    markets: str = "spreads,totals,h2h",
) -> Optional[List[Dict[str, Any]]]:
    """Fetch current NBA odds from the-odds-api.

    Returns list of events with odds from all available bookmakers.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    url = f"{ODDS_API_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",  # Get American odds (-110, +150)
    }

    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Odds API error {response.status}: {error_text[:100]}")
                    return None
                return await response.json()
    except Exception as e:
        print(f"Odds API error: {e}")
        return None


def find_game_odds(
    odds_data: List[Dict[str, Any]],
    home_team: str,
    away_team: str,
) -> Optional[Dict[str, Any]]:
    """Find odds for a specific game by team names.

    Handles team name normalization (e.g., 'LA Clippers' -> 'Los Angeles Clippers').
    """
    home_normalized = normalize_team_name(home_team)
    away_normalized = normalize_team_name(away_team)

    for event in odds_data:
        if (event.get("home_team") == home_normalized and
            event.get("away_team") == away_normalized):
            return event
    return None


def extract_odds(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract structured odds from an event.

    Uses first available bookmaker. Returns None if no bookmakers available.
    """
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    book = bookmakers[0]
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")

    result: Dict[str, Any] = {
        "spread": None,
        "total": None,
        "moneyline": None,
    }

    for market in book.get("markets", []):
        key = market.get("key")
        outcomes = market.get("outcomes", [])

        if key == "spreads":
            home = next((o for o in outcomes if o.get("name") == home_team), None)
            away = next((o for o in outcomes if o.get("name") == away_team), None)
            if home and away:
                result["spread"] = {
                    "home": home.get("point"),
                    "away": away.get("point"),
                }

        elif key == "totals":
            over = next((o for o in outcomes if o.get("name") == "Over"), None)
            if over:
                result["total"] = over.get("point")

        elif key == "h2h":
            home = next((o for o in outcomes if o.get("name") == home_team), None)
            away = next((o for o in outcomes if o.get("name") == away_team), None)
            if home and away:
                result["moneyline"] = {
                    "home": home.get("price"),
                    "away": away.get("price"),
                }

    # Return None if no useful odds were extracted
    if result["spread"] is None and result["total"] is None and result["moneyline"] is None:
        return None

    return result
