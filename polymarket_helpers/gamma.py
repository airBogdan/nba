import json
import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
NBA_SERIES_ID = "10345"
NBA_TAG_ID = "100639"

BET_TYPE_TO_SPORTS_MARKET_TYPE = {
    "moneyline": {"moneyline"},
    "spread": {"spread", "spreads"},
    "total": {"total", "totals"},
}


def _normalize_market(market: dict) -> dict:
    """Parse JSON-encoded string fields in a market dict."""
    for field in ("outcomes", "outcomePrices", "clobTokenIds"):
        val = market.get(field)
        if isinstance(val, str):
            market[field] = json.loads(val)
    return market


def fetch_nba_events(date: str) -> list[dict]:
    """Fetch NBA events from Gamma API for a given date (YYYY-MM-DD)."""
    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/events",
            params={
                "series_id": NBA_SERIES_ID,
                "tag_id": NBA_TAG_ID,
                "closed": "false",
                "active": "true",
            },
        )
        resp.raise_for_status()
        events = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching events: {e}")
        return []

    return [e for e in events if date in e.get("ticker", "")]


def find_market(event: dict, bet_type: str, line: float | None) -> dict | None:
    """Find a matching market in an event by bet type and line."""
    valid_types = BET_TYPE_TO_SPORTS_MARKET_TYPE.get(bet_type)
    if not valid_types:
        return None

    for market in event.get("markets", []):
        market = _normalize_market(market)
        sport_type = market.get("sportsMarketType", "")
        if sport_type not in valid_types:
            continue

        # Skip markets not accepting orders (handle both bool and string)
        accepting = market.get("acceptingOrders")
        if not accepting or str(accepting).lower() == "false":
            continue

        if bet_type == "moneyline":
            return market

        # For spread/total, require exact line match
        if line is None:
            continue
        market_line = market.get("line")
        if market_line is not None and float(market_line) == float(line):
            return market

    return None
