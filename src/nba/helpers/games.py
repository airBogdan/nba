"""Head-to-head games analysis."""

from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict

from .api import get_head_to_head_games, get_game_statistics
from .utils import get_current_nba_season_year
from .types import Game, GameStatistics, H2HResults, H2HSummary, QuarterAnalysis, TeamGameStats


class RawGameStats(TypedDict, total=False):
    """Raw game statistics from API."""
    points: int
    fgp: str
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
    plusMinus: str


class ProcessedGameStats(TypedDict):
    """Processed game statistics."""
    points: int
    fgp: str
    ftp: str
    tpm: int
    tpp: str
    totReb: int
    assists: int
    steals: int
    turnovers: int
    blocks: int
    plusMinus: int
    ast_to_tov: float
    stocks: int  # steals + blocks (defensive disruption)


def process_game_stats(raw: RawGameStats) -> ProcessedGameStats:
    """Process raw game stats into computed metrics."""
    turnovers = raw.get("turnovers", 1) or 1
    plus_minus_str = raw.get("plusMinus", "0")
    plus_minus = int(plus_minus_str) if plus_minus_str else 0
    assists = raw.get("assists", 0) or 0
    steals = raw.get("steals", 0) or 0
    blocks = raw.get("blocks", 0) or 0

    return {
        "points": raw.get("points", 0),
        "fgp": raw.get("fgp", "0"),
        "ftp": raw.get("ftp", "0"),
        "tpm": raw.get("tpm", 0),
        "tpp": raw.get("tpp", "0"),
        "totReb": raw.get("totReb", 0),
        "assists": assists,
        "steals": steals,
        "turnovers": raw.get("turnovers", 0),
        "blocks": blocks,
        "plusMinus": plus_minus,
        "ast_to_tov": round(assists / turnovers, 2),
        "stocks": steals + blocks,
    }


def process_h2h_results(games: Optional[List[Any]]) -> Optional[H2HResults]:
    """Process raw H2H games into organized results by season."""
    if not games:
        return None

    current_season_year = get_current_nba_season_year()
    if not current_season_year:
        return None

    cutoff_year = current_season_year - 5
    processed_results: H2HResults = {}

    for game in games:
        season_year = game.get("season")

        # Skip games that are older than the cutoff year
        if season_year < cutoff_year:
            continue

        teams = game.get("teams", {})
        scores = game.get("scores", {})
        home_team = teams.get("home", {})
        visitor_team = teams.get("visitors", {})

        home_points = scores.get("home", {}).get("points")
        visitor_points = scores.get("visitors", {}).get("points")

        # Skip games with null scores
        if home_points is None or visitor_points is None:
            continue

        # Initialize array for this season if it doesn't exist
        if season_year not in processed_results:
            processed_results[season_year] = []

        # Determine winner
        if home_points > visitor_points:
            winner = home_team.get("name")
        elif visitor_points > home_points:
            winner = visitor_team.get("name")
        else:
            winner = "tie"

        # Parse linescore (quarter scores as strings to numbers)
        home_linescore_raw = scores.get("home", {}).get("linescore") or []
        visitor_linescore_raw = scores.get("visitors", {}).get("linescore") or []

        home_linescore = [int(q) if q else 0 for q in home_linescore_raw]
        visitor_linescore = [int(q) if q else 0 for q in visitor_linescore_raw]

        processed_results[season_year].append({
            "id": game.get("id"),
            "home_team": home_team.get("name"),
            "visitor_team": visitor_team.get("name"),
            "home_points": home_points,
            "visitor_points": visitor_points,
            "winner": winner,
            "point_diff": home_points - visitor_points,
            "home_linescore": home_linescore,
            "visitor_linescore": visitor_linescore,
        })

    return processed_results


def compute_quarter_analysis(
    h2h_results: H2HResults,
    team1: str,
    team2: str
) -> Optional[QuarterAnalysis]:
    """Compute quarter-by-quarter analysis of H2H games."""
    # Flatten all games
    all_games = [g for games in h2h_results.values() for g in games]

    # Filter to games with quarter data
    games_with_quarters = [
        g for g in all_games
        if g.get("home_linescore") and len(g["home_linescore"]) >= 4
        and g.get("visitor_linescore") and len(g["visitor_linescore"]) >= 4
    ]

    if len(games_with_quarters) == 0:
        return None

    total_q1 = 0
    total_q2 = 0
    total_q3 = 0
    total_q4 = 0
    team1_q1 = 0
    team2_q1 = 0
    team1_q4 = 0
    team2_q4 = 0
    halftime_leader_wins = 0
    overtime_games = 0

    for game in games_with_quarters:
        home_qs = game["home_linescore"]
        visitor_qs = game["visitor_linescore"]

        # Quarter totals (combined)
        total_q1 += home_qs[0] + visitor_qs[0]
        total_q2 += home_qs[1] + visitor_qs[1]
        total_q3 += home_qs[2] + visitor_qs[2]
        total_q4 += home_qs[3] + visitor_qs[3]

        # Team-specific quarter scores
        is_team1_home = game["home_team"] == team1
        team1_qs = home_qs if is_team1_home else visitor_qs
        team2_qs = visitor_qs if is_team1_home else home_qs

        team1_q1 += team1_qs[0]
        team2_q1 += team2_qs[0]
        team1_q4 += team1_qs[3]
        team2_q4 += team2_qs[3]

        # Halftime leader analysis
        team1_halftime = team1_qs[0] + team1_qs[1]
        team2_halftime = team2_qs[0] + team2_qs[1]
        halftime_leader = team1 if team1_halftime > team2_halftime else team2
        if halftime_leader == game["winner"]:
            halftime_leader_wins += 1

        # Overtime detection (more than 4 quarters)
        if len(home_qs) > 4 or len(visitor_qs) > 4:
            overtime_games += 1

    n = len(games_with_quarters)

    return {
        "avg_q1_total": round(total_q1 / n, 1),
        "avg_q2_total": round(total_q2 / n, 1),
        "avg_q3_total": round(total_q3 / n, 1),
        "avg_q4_total": round(total_q4 / n, 1),
        "avg_first_half": round((total_q1 + total_q2) / n, 1),
        "avg_second_half": round((total_q3 + total_q4) / n, 1),
        "team1_q1_avg": round(team1_q1 / n, 1),
        "team2_q1_avg": round(team2_q1 / n, 1),
        "team1_q4_avg": round(team1_q4 / n, 1),
        "team2_q4_avg": round(team2_q4 / n, 1),
        "halftime_leader_wins_pct": round(halftime_leader_wins / n, 2),
        "overtime_games": overtime_games,
        "total_games_analyzed": n,
    }


async def h2h(team1_id: int, team2_id: int) -> Optional[H2HResults]:
    """Get head-to-head results between two teams."""
    if not team1_id or not team2_id:
        print(f"Invalid team IDs: team1_id={team1_id}, team2_id={team2_id}")
        return None

    resp = await get_head_to_head_games(team1_id, team2_id)
    return process_h2h_results(resp)


async def add_game_statistics_to_h2h_results(
    h2h_results: Optional[H2HResults]
) -> H2HResults:
    """Add detailed game statistics to H2H results."""
    if not h2h_results:
        return {}

    for year in h2h_results:
        games = h2h_results[year]

        for i, game in enumerate(games):
            game_id = game["id"]

            # Fetch statistics for the game
            statistics = await get_game_statistics(game_id)
            print(f"API request for game ID: {game_id}")

            if statistics and len(statistics) >= 2:
                # Process home team statistics
                home_team_stats = next(
                    (s for s in statistics if s.get("team", {}).get("name") == game["home_team"]),
                    None
                )
                if home_team_stats and home_team_stats.get("statistics"):
                    stats_list = home_team_stats["statistics"]
                    if stats_list and len(stats_list) > 0:
                        processed = process_game_stats(stats_list[0])
                        games[i]["home_statistics"] = processed

                # Process visitor team statistics
                visitor_team_stats = next(
                    (s for s in statistics if s.get("team", {}).get("name") == game["visitor_team"]),
                    None
                )
                if visitor_team_stats and visitor_team_stats.get("statistics"):
                    stats_list = visitor_team_stats["statistics"]
                    if stats_list and len(stats_list) > 0:
                        processed = process_game_stats(stats_list[0])
                        games[i]["visitor_statistics"] = processed

    return h2h_results


def compute_h2h_summary(h2h_results: H2HResults, team1: str, team2: str) -> H2HSummary:
    """Compute summary statistics for H2H matchup."""
    # Flatten all games across seasons with season info
    all_games: List[Dict[str, Any]] = []
    for season, games in h2h_results.items():
        for game in games:
            all_games.append({"game": game, "season": int(season)})

    # Sort by season descending (most recent first)
    all_games.sort(key=lambda x: x["season"], reverse=True)

    team1_wins = 0
    team2_wins = 0
    team1_home_wins = 0
    team1_home_losses = 0
    team1_away_wins = 0
    team1_away_losses = 0
    total_point_diff = 0
    team1_total_points = 0
    team2_total_points = 0
    close_games = 0
    blowouts = 0

    for item in all_games:
        game = item["game"]
        is_team1_home = game["home_team"] == team1
        team1_points = game["home_points"] if is_team1_home else game["visitor_points"]
        team2_points = game["visitor_points"] if is_team1_home else game["home_points"]
        diff = team1_points - team2_points

        team1_total_points += team1_points
        team2_total_points += team2_points
        total_point_diff += diff

        if abs(diff) <= 5:
            close_games += 1
        if abs(diff) >= 15:
            blowouts += 1

        if game["winner"] == team1:
            team1_wins += 1
            if is_team1_home:
                team1_home_wins += 1
            else:
                team1_away_wins += 1
        elif game["winner"] == team2:
            team2_wins += 1
            if is_team1_home:
                team1_home_losses += 1
            else:
                team1_away_losses += 1

    total_games = len(all_games)
    last_5 = [item["game"]["winner"] for item in all_games[:5]]

    # Determine recent trend based on last 5 games
    recent_team1_wins = sum(1 for w in last_5 if w == team1)
    if recent_team1_wins >= 4:
        recent_trend = "team1_hot"
    elif recent_team1_wins <= 1:
        recent_trend = "team2_hot"
    else:
        recent_trend = "balanced"

    return {
        "team1": team1,
        "team2": team2,
        "total_games": total_games,
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "team1_win_pct": round(team1_wins / total_games, 3) if total_games > 0 else 0.0,
        "team1_home_wins": team1_home_wins,
        "team1_home_losses": team1_home_losses,
        "team1_away_wins": team1_away_wins,
        "team1_away_losses": team1_away_losses,
        "avg_point_diff": round(total_point_diff / total_games, 1) if total_games > 0 else 0.0,
        "avg_total_points": round((team1_total_points + team2_total_points) / total_games, 1) if total_games > 0 else 0.0,
        "team1_avg_points": round(team1_total_points / total_games, 1) if total_games > 0 else 0.0,
        "team2_avg_points": round(team2_total_points / total_games, 1) if total_games > 0 else 0.0,
        "last_5_games": last_5,
        "recent_trend": recent_trend,
        "close_games": close_games,
        "blowouts": blowouts,
    }
