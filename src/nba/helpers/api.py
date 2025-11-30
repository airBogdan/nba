"""NBA API client and data processors."""

import os
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import TypedDict

import aiohttp

from .utils import get_current_nba_season_year


URL = "v2.nba.api-sports.io"
HEADERS = {
    "x-rapidapi-key": os.environ.get("NBA_RAPID_API_KEY", ""),
    "x-rapidapi-host": URL,
}


# --- TypedDicts for API responses and processed data ---


class TeamPlayerStatistics(TypedDict, total=False):
    """Raw player statistics from API."""
    player: Dict[str, Any]  # { id, firstname, lastname }
    team: Dict[str, Any]  # { id, name, nickname, code, logo }
    game: Dict[str, int]  # { id }
    points: int
    pos: Optional[Any]
    min: str
    fgm: int
    fga: int
    fgp: str
    ftm: int
    fta: int
    ftp: str
    tpm: int
    tpa: int
    tpp: str
    offReb: int
    defReb: int
    totReb: int
    assists: int
    pFouls: int
    steals: int
    turnovers: int
    blocks: int
    plusMinus: str
    comment: Optional[Any]


class ProcessedPlayerStats(TypedDict):
    """Aggregated player statistics."""
    id: int
    name: str
    games: int
    mpg: float
    ppg: float
    rpg: float
    apg: float
    spg: float
    bpg: float
    topg: float
    fgp: float
    tpp: float
    ftp: float
    plus_minus: float


class RawTeamStats(TypedDict, total=False):
    """Raw team statistics from API."""
    games: int
    points: int
    fgm: int
    fga: int
    fgp: str
    ftm: int
    fta: int
    ftp: str
    tpm: int
    tpa: int
    tpp: str
    offReb: int
    defReb: int
    totReb: int
    assists: int
    steals: int
    turnovers: int
    blocks: int
    plusMinus: int


class ProcessedTeamStats(TypedDict):
    """Processed team statistics."""
    games: int
    ppg: float
    apg: float
    rpg: float
    spg: float
    bpg: float
    topg: float
    ast_to_tov: float
    net_rating: float
    stocks_pg: float
    three_pt_rate: float
    tpp: str
    fgp: str
    ftp: str
    pace: float
    off_reb_rate: float
    ft_rate: float


class RecentGame(TypedDict):
    """Recent game result."""
    vs: str
    result: Literal["W", "L"]
    score: str
    home: bool
    margin: int
    date: str


# --- API functions ---


async def fetch_nba_api(endpoint: str) -> Optional[List[Any]]:
    """Fetch data from NBA API."""
    url = f"https://{URL}/{endpoint}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS) as response:
            data = await response.json()
            if data and "response" in data and len(data["response"]) > 0:
                return data["response"]
            return None


async def get_teams() -> Optional[List[Any]]:
    """Get all NBA teams."""
    return await fetch_nba_api("teams")


async def get_game_statistics(game_id: int) -> Optional[List[Any]]:
    """Get statistics for a specific game."""
    return await fetch_nba_api(f"games/statistics?id={game_id}")


async def get_team_id_by_name(name: str) -> Optional[int]:
    """Get team ID by team name."""
    teams = await get_teams()
    if not teams:
        return None

    team = next(
        (t for t in teams if t["name"].lower() == name.lower()),
        None
    )
    return team["id"] if team else None


async def get_head_to_head_games(team1_id: int, team2_id: int) -> Optional[List[Any]]:
    """Get head-to-head games between two teams."""
    return await fetch_nba_api(f"games?h2h={team1_id}-{team2_id}")


async def get_team_standings(team_id: int, season: int) -> Optional[List[Any]]:
    """Get team standings for a season."""
    return await fetch_nba_api(f"standings?team={team_id}&league=standard&season={season}")


async def get_team_statistics(team_id: int, season: int) -> Optional[List[Any]]:
    """Get team statistics for a season."""
    return await fetch_nba_api(f"teams/statistics?id={team_id}&season={season}")


async def get_team_players_statistics(team_id: int, season: int) -> Optional[List[Any]]:
    """Get all player statistics for a team in a season."""
    return await fetch_nba_api(f"players/statistics?team={team_id}&season={season}")


# --- Data processing functions ---


def parse_minutes(min_str: str) -> float:
    """Parse minutes string (e.g., '32:45') to float."""
    if not min_str:
        return 0.0
    parts = min_str.split(":")
    minutes = int(parts[0]) if parts[0] else 0
    seconds = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return minutes + seconds / 60


def process_player_statistics(
    raw_stats: List[TeamPlayerStatistics],
    top_n: int = 8,
    min_games: int = 5
) -> List[ProcessedPlayerStats]:
    """
    Process raw player statistics into aggregated per-game stats.

    Args:
        raw_stats: Raw player game logs from API
        top_n: Number of top players to return (by minutes)
        min_games: Minimum games played to be included

    Returns:
        List of processed player stats, sorted by minutes per game
    """
    if not raw_stats:
        return []

    # Group stats by player id
    by_player: Dict[int, Dict[str, Any]] = {}

    for stat in raw_stats:
        pid = stat["player"]["id"]
        if pid not in by_player:
            by_player[pid] = {
                "name": f"{stat['player']['firstname']} {stat['player']['lastname']}",
                "games": []
            }
        by_player[pid]["games"].append(stat)

    # Aggregate each player's stats
    aggregated: List[ProcessedPlayerStats] = []

    for player_id, data in by_player.items():
        games = data["games"]
        game_count = len(games)

        # Skip players with too few games
        if game_count < min_games:
            continue

        # Sum up all stats
        total_min = 0.0
        total_pts = 0
        total_reb = 0
        total_ast = 0
        total_stl = 0
        total_blk = 0
        total_tov = 0
        total_fgm = 0
        total_fga = 0
        total_tpm = 0
        total_tpa = 0
        total_ftm = 0
        total_fta = 0
        total_pm = 0

        for g in games:
            total_min += parse_minutes(g.get("min", ""))
            total_pts += g.get("points", 0) or 0
            total_reb += g.get("totReb", 0) or 0
            total_ast += g.get("assists", 0) or 0
            total_stl += g.get("steals", 0) or 0
            total_blk += g.get("blocks", 0) or 0
            total_tov += g.get("turnovers", 0) or 0
            total_fgm += g.get("fgm", 0) or 0
            total_fga += g.get("fga", 0) or 0
            total_tpm += g.get("tpm", 0) or 0
            total_tpa += g.get("tpa", 0) or 0
            total_ftm += g.get("ftm", 0) or 0
            total_fta += g.get("fta", 0) or 0
            pm_str = g.get("plusMinus", "0")
            total_pm += int(pm_str) if pm_str else 0

        aggregated.append({
            "id": player_id,
            "name": data["name"],
            "games": game_count,
            "mpg": round(total_min / game_count, 1),
            "ppg": round(total_pts / game_count, 1),
            "rpg": round(total_reb / game_count, 1),
            "apg": round(total_ast / game_count, 1),
            "spg": round(total_stl / game_count, 1),
            "bpg": round(total_blk / game_count, 1),
            "topg": round(total_tov / game_count, 1),
            "fgp": round((total_fgm / total_fga) * 100, 1) if total_fga > 0 else 0.0,
            "tpp": round((total_tpm / total_tpa) * 100, 1) if total_tpa > 0 else 0.0,
            "ftp": round((total_ftm / total_fta) * 100, 1) if total_fta > 0 else 0.0,
            "plus_minus": round(total_pm / game_count, 1),
        })

    # Sort by minutes per game and return top N
    aggregated.sort(key=lambda x: x["mpg"], reverse=True)
    return aggregated[:top_n]


def process_team_stats(raw: RawTeamStats) -> ProcessedTeamStats:
    """Process raw team statistics into derived metrics."""
    games = raw.get("games", 1) or 1
    points = raw.get("points", 0) or 0
    ppg = round(points / games, 1)

    # Pace estimate: possessions ≈ FGA + 0.44*FTA + TOV - OREB
    fga = raw.get("fga", 0) or 0
    fta = raw.get("fta", 0) or 0
    turnovers = raw.get("turnovers", 0) or 0
    off_reb = raw.get("offReb", 0) or 0
    tot_reb = raw.get("totReb", 0) or 0

    possessions = fga + 0.44 * fta + turnovers - off_reb
    pace = round(possessions / games, 1)

    # Offensive rebound rate: OREB / total rebounds
    off_reb_rate = round((off_reb / tot_reb) * 100, 1) if tot_reb > 0 else 0.0

    # Free throw rate: FTA per FGA
    ft_rate = round((fta / fga) * 100, 1) if fga > 0 else 0.0

    assists = raw.get("assists", 0) or 0
    steals = raw.get("steals", 0) or 0
    blocks = raw.get("blocks", 0) or 0
    plus_minus = raw.get("plusMinus", 0) or 0
    tpa = raw.get("tpa", 0) or 0

    return {
        "games": raw.get("games", 0),
        "ppg": ppg,
        "apg": round(assists / games, 1),
        "rpg": round(tot_reb / games, 1),
        "spg": round(steals / games, 1),
        "bpg": round(blocks / games, 1),
        "topg": round(turnovers / games, 1),
        "ast_to_tov": round(assists / (turnovers or 1), 2),
        "net_rating": round(plus_minus / games, 2),
        "stocks_pg": round((steals + blocks) / games, 1),
        "three_pt_rate": round(tpa / games, 1),
        "tpp": raw.get("tpp", "0"),
        "fgp": raw.get("fgp", "0"),
        "ftp": raw.get("ftp", "0"),
        "pace": pace,
        "off_reb_rate": off_reb_rate,
        "ft_rate": ft_rate,
    }


async def get_team_statistics_for_seasons(
    team_id: int,
    num_seasons: int = 2
) -> Optional[Dict[int, ProcessedTeamStats]]:
    """Get processed team statistics for multiple seasons."""
    current_season = get_current_nba_season_year()
    if not current_season:
        return None

    seasons_stats: Dict[int, ProcessedTeamStats] = {}

    for i in range(num_seasons):
        season_year = current_season - i
        stats = await get_team_statistics(team_id, season_year)

        if stats and len(stats) > 0:
            seasons_stats[season_year] = process_team_stats(stats[0])

    return seasons_stats


async def get_team_recent_games(
    team_id: int,
    season: int,
    limit: int = 5
) -> List[RecentGame]:
    """Get recent completed games for a team."""
    raw_games = await fetch_nba_api(f"games?team={team_id}&season={season}")
    if not raw_games:
        return []

    # Filter to completed games only (status.short === 3 means finished)
    completed = [
        g for g in raw_games
        if g.get("status", {}).get("short") == 3
        and g.get("scores", {}).get("home", {}).get("points") is not None
        and g.get("scores", {}).get("visitors", {}).get("points") is not None
    ]

    # Sort by date descending (most recent first)
    completed.sort(
        key=lambda g: g.get("date", {}).get("start", ""),
        reverse=True
    )

    # Take the last N games and process
    results: List[RecentGame] = []
    for game in completed[:limit]:
        is_home = game["teams"]["home"]["id"] == team_id
        team_points = (
            game["scores"]["home"]["points"]
            if is_home
            else game["scores"]["visitors"]["points"]
        )
        opp_points = (
            game["scores"]["visitors"]["points"]
            if is_home
            else game["scores"]["home"]["points"]
        )
        opponent = (
            game["teams"]["visitors"]["name"]
            if is_home
            else game["teams"]["home"]["name"]
        )

        results.append({
            "vs": opponent,
            "result": "W" if team_points > opp_points else "L",
            "score": f"{team_points}-{opp_points}",
            "home": is_home,
            "margin": team_points - opp_points,
            "date": game["date"]["start"].split("T")[0],
        })

    return results
