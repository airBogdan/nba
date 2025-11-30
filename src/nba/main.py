"""NBA Analytics main entry point."""

import asyncio
import json
import os
from pathlib import Path
from typing import List

from .helpers.api import (
    get_team_id_by_name,
    get_team_statistics_for_seasons,
    get_team_players_statistics,
    get_team_recent_games,
    process_player_statistics,
    ProcessedPlayerStats,
    RecentGame,
)
from .helpers.utils import get_current_nba_season_year
from .helpers.teams import get_teams_standings
from .helpers.games import h2h, compute_h2h_summary
from .helpers.matchup import build_matchup_analysis


# Output directory (relative to this file)
OUTPUT_DIR = Path(__file__).parent / "output"


def write_json(filename: str, data: dict) -> None:
    """Write data to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written: {filepath}")


async def main() -> None:
    """Main entry point for matchup analysis."""
    try:
        team1_name = "Atlanta Hawks"
        team2_name = "Philadelphia 76ers"
        home_team = team1_name  # Set which team is hosting

        # Get team IDs
        team1_id = await get_team_id_by_name(team1_name)
        team2_id = await get_team_id_by_name(team2_name)

        if not team1_id or not team2_id:
            print(f"Could not find one or both teams: {team1_name}={team1_id}, {team2_name}={team2_id}")
            return

        # Fetch standings and H2H data
        teams_standings = await get_teams_standings(team1_id, team1_name, team2_id, team2_name)
        print(f"Teams Standings: {len(teams_standings)} teams loaded")

        h2h_results = await h2h(team1_id, team2_id)
        h2h_game_count = sum(len(games) for games in h2h_results.values()) if h2h_results else 0
        print(f"H2H Results: {h2h_game_count} games loaded")

        h2h_summary = compute_h2h_summary(h2h_results, team1_name, team2_name) if h2h_results else None
        print("H2H Summary: computed")

        # Fetch team statistics
        team1_stats = await get_team_statistics_for_seasons(team1_id)
        print(f"{team1_name} Stats: {len(team1_stats) if team1_stats else 0} seasons")

        team2_stats = await get_team_statistics_for_seasons(team2_id)
        print(f"{team2_name} Stats: {len(team2_stats) if team2_stats else 0} seasons")

        # Fetch and process player statistics
        current_season = get_current_nba_season_year()
        team1_players: List[ProcessedPlayerStats] = []
        team2_players: List[ProcessedPlayerStats] = []
        team1_recent_games: List[RecentGame] = []
        team2_recent_games: List[RecentGame] = []

        if current_season:
            team1_raw_stats = await get_team_players_statistics(team1_id, current_season)
            team1_players = process_player_statistics(team1_raw_stats or [])
            print(f"{team1_name} Players: {len(team1_players)} rotation players")

            team2_raw_stats = await get_team_players_statistics(team2_id, current_season)
            team2_players = process_player_statistics(team2_raw_stats or [])
            print(f"{team2_name} Players: {len(team2_players)} rotation players")

            # Fetch recent games
            team1_recent_games = await get_team_recent_games(team1_id, current_season, 5)
            print(f"{team1_name} Recent Games: {len(team1_recent_games)} games")

            team2_recent_games = await get_team_recent_games(team2_id, current_season, 5)
            print(f"{team2_name} Recent Games: {len(team2_recent_games)} games")

        # Build unified matchup analysis
        matchup_analysis = build_matchup_analysis({
            "team1_name": team1_name,
            "team2_name": team2_name,
            "home_team": home_team,
            "team1_standings": teams_standings.get(team1_name, []),
            "team2_standings": teams_standings.get(team2_name, []),
            "team1_stats": team1_stats,
            "team2_stats": team2_stats,
            "team1_players": team1_players,
            "team2_players": team2_players,
            "team1_recent_games": team1_recent_games,
            "team2_recent_games": team2_recent_games,
            "h2h_summary": h2h_summary,
            "h2h_results": h2h_results,
        })

        write_json("matchup_analysis.json", matchup_analysis)
        print("Matchup Analysis: written")

    except Exception as error:
        print(f"Error: {error}")
        raise


def run() -> None:
    """Run the main function."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
