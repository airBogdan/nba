"""Tests for helpers/matchup.py."""

from datetime import datetime
from unittest.mock import patch

import pytest

from helpers.matchup import (
    build_team_snapshot,
    compute_edges,
    compute_days_rest,
    compute_streak,
    compute_games_last_n_days,
    compute_schedule_context,
    compute_h2h_patterns,
    compute_h2h_matchup_stats,
    compute_recent_h2h,
    compute_totals_analysis,
    build_team_players,
)


class TestBuildTeamSnapshot:
    """Tests for build_team_snapshot function."""

    @pytest.fixture
    def sample_standing(self):
        """Sample team standing."""
        return {
            "season": 2024,
            "conference_rank": 3,
            "wins": 25,
            "losses": 15,
            "win_pct": ".625",
            "home_wins": 15,
            "home_losses": 5,
            "away_wins": 10,
            "away_losses": 10,
            "last_ten_wins": 7,
            "last_ten_losses": 3,
            "home_win_pct": 0.75,
            "away_win_pct": 0.5,
            "last_ten_pct": 0.7,
            "home_court_advantage": 0.25,
        }

    @pytest.fixture
    def sample_stats(self):
        """Sample team stats."""
        return {
            "games": 40,
            "ppg": 112.5,
            "apg": 26.0,
            "rpg": 44.0,
            "topg": 13.5,
            "disruption": 10.0,
            "net_rating": 5.0,
            "tpp": 36.5,
            "fgp": 47.2,
            "pace": 100.0,
        }

    def test_builds_snapshot_with_all_data(self, sample_standing, sample_stats):
        """Builds complete snapshot with standing and stats."""
        result = build_team_snapshot("Hawks", sample_standing, sample_stats)

        assert result["name"] == "Hawks"
        assert result["record"] == "25-15"
        assert result["conf_rank"] == 3
        assert result["games"] == 40
        assert result["ppg"] == 112.5

    def test_computes_ortg_drtg(self, sample_standing, sample_stats):
        """ORTG and DRTG estimated from net rating."""
        result = build_team_snapshot("Hawks", sample_standing, sample_stats)

        # ORTG = 112 + net_rating/2 = 112 + 2.5 = 114.5
        assert result["ortg"] == 114.5
        # DRTG = 112 - net_rating/2 = 112 - 2.5 = 109.5
        assert result["drtg"] == 109.5

    def test_computes_opp_ppg(self, sample_standing, sample_stats):
        """Opponent PPG estimated from DRTG and pace."""
        result = build_team_snapshot("Hawks", sample_standing, sample_stats)
        # opp_ppg = DRTG * pace / 100 = 109.5 * 100 / 100 = 109.5
        assert result["opp_ppg"] == 109.5

    def test_handles_none_standing(self, sample_stats):
        """Handles None standing gracefully."""
        result = build_team_snapshot("Hawks", None, sample_stats)

        assert result["record"] == "N/A"
        assert result["conf_rank"] == 0
        assert result["last_ten"] == "N/A"
        assert result["home_record"] == "N/A"
        assert result["away_record"] == "N/A"

    def test_handles_none_stats(self, sample_standing):
        """Handles None stats gracefully."""
        result = build_team_snapshot("Hawks", sample_standing, None)

        assert result["games"] == 0
        assert result["ppg"] == 0.0
        assert result["pace"] == 100.0  # Default pace

    def test_handles_both_none(self):
        """Handles both None inputs."""
        result = build_team_snapshot("Hawks", None, None)

        assert result["name"] == "Hawks"
        assert result["record"] == "N/A"
        assert result["games"] == 0


class TestComputeEdges:
    """Tests for compute_edges function."""

    @pytest.fixture
    def team1_snapshot(self):
        """Team 1 snapshot."""
        return {
            "name": "Hawks",
            "ppg": 115.0,
            "net_rating": 5.0,
            "last_ten_pct": 0.7,
            "topg": 12.0,
            "rpg": 45.0,
            "fgp": 48.0,
            "tpp": 38.0,
            "pace": 102.0,
        }

    @pytest.fixture
    def team2_snapshot(self):
        """Team 2 snapshot."""
        return {
            "name": "76ers",
            "ppg": 110.0,
            "net_rating": 2.0,
            "last_ten_pct": 0.5,
            "topg": 14.0,
            "rpg": 42.0,
            "fgp": 46.0,
            "tpp": 35.0,
            "pace": 98.0,
        }

    def test_computes_ppg_edge(self, team1_snapshot, team2_snapshot):
        """PPG edge computed correctly."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["ppg"] == 5.0  # 115 - 110

    def test_computes_net_rating_edge(self, team1_snapshot, team2_snapshot):
        """Net rating edge computed correctly."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["net_rating"] == 3.0  # 5 - 2

    def test_computes_form_edge(self, team1_snapshot, team2_snapshot):
        """Form edge (last 10 pct difference) computed correctly."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["form"] == 0.2  # 0.7 - 0.5

    def test_computes_turnover_edge(self, team1_snapshot, team2_snapshot):
        """Turnover edge (positive = team1 turns over less)."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["turnovers"] == 2.0  # 14 - 12 (team2 - team1)

    def test_computes_rebound_edge(self, team1_snapshot, team2_snapshot):
        """Rebound edge computed correctly."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["rebounds"] == 3.0  # 45 - 42

    def test_computes_shooting_edges(self, team1_snapshot, team2_snapshot):
        """FG% and 3P% edges computed correctly."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["fgp"] == 2.0  # 48 - 46
        assert result["three_pt_pct"] == 3.0  # 38 - 35

    def test_computes_pace_metrics(self, team1_snapshot, team2_snapshot):
        """Pace difference and combined pace computed correctly."""
        result = compute_edges(team1_snapshot, team2_snapshot)
        assert result["pace"] == 4.0  # 102 - 98
        assert result["combined_pace"] == 100.0  # (102 + 98) / 2


class TestComputeDaysRest:
    """Tests for compute_days_rest function."""

    def test_returns_none_for_empty_list(self):
        """Returns None for empty recent games list."""
        assert compute_days_rest([]) is None

    @patch("helpers.matchup.datetime")
    def test_computes_days_since_last_game(self, mock_datetime):
        """Computes days since last game correctly."""
        mock_datetime.now.return_value = datetime(2024, 1, 15)
        mock_datetime.strptime = datetime.strptime

        recent = [{"date": "2024-01-13"}]
        result = compute_days_rest(recent)
        assert result == 2

    @patch("helpers.matchup.datetime")
    def test_same_day_returns_zero(self, mock_datetime):
        """Returns 0 for game on same day."""
        mock_datetime.now.return_value = datetime(2024, 1, 15)
        mock_datetime.strptime = datetime.strptime

        recent = [{"date": "2024-01-15"}]
        result = compute_days_rest(recent)
        assert result == 0


class TestComputeStreak:
    """Tests for compute_streak function."""

    def test_returns_no_streak_for_empty(self):
        """Returns no streak for empty games list."""
        result = compute_streak([])
        assert result["type"] is None
        assert result["count"] == 0

    def test_win_streak(self):
        """Computes win streak correctly."""
        recent = [
            {"result": "W"},
            {"result": "W"},
            {"result": "W"},
            {"result": "L"},
        ]
        result = compute_streak(recent)
        assert result["type"] == "W"
        assert result["count"] == 3

    def test_loss_streak(self):
        """Computes loss streak correctly."""
        recent = [
            {"result": "L"},
            {"result": "L"},
            {"result": "W"},
        ]
        result = compute_streak(recent)
        assert result["type"] == "L"
        assert result["count"] == 2

    def test_single_game(self):
        """Single game streak."""
        recent = [{"result": "W"}]
        result = compute_streak(recent)
        assert result["type"] == "W"
        assert result["count"] == 1


class TestComputeGamesLastNDays:
    """Tests for compute_games_last_n_days function."""

    def test_returns_zero_for_empty(self):
        """Returns 0 for empty games list."""
        assert compute_games_last_n_days([]) == 0

    @patch("helpers.matchup.datetime")
    def test_counts_games_in_window(self, mock_datetime):
        """Counts games within N day window."""
        mock_datetime.now.return_value = datetime(2024, 1, 15)
        mock_datetime.strptime = datetime.strptime

        recent = [
            {"date": "2024-01-14"},  # 1 day ago - in window
            {"date": "2024-01-12"},  # 3 days ago - in window
            {"date": "2024-01-10"},  # 5 days ago - in window
            {"date": "2024-01-05"},  # 10 days ago - outside 7 day window
        ]
        result = compute_games_last_n_days(recent, days=7)
        assert result == 3


class TestComputeScheduleContext:
    """Tests for compute_schedule_context function."""

    @patch("helpers.matchup.datetime")
    def test_computes_full_context(self, mock_datetime):
        """Computes complete schedule context."""
        mock_datetime.now.return_value = datetime(2024, 1, 15)
        mock_datetime.strptime = datetime.strptime

        recent = [
            {"date": "2024-01-14", "result": "W", "vs_win_pct": 0.6},
            {"date": "2024-01-12", "result": "W", "vs_win_pct": 0.55},
            {"date": "2024-01-10", "result": "L", "vs_win_pct": 0.7},
        ]
        result = compute_schedule_context(recent)

        assert result["days_rest"] == 1
        assert result["streak"] == "W2"
        assert result["games_last_7_days"] == 3
        # Quality wins: 2 (both Ws were vs .500+ teams)
        assert result["quality_wins"] == 2
        # Quality losses: 1 (L was vs .500+ team)
        assert result["quality_losses"] == 1

    def test_handles_empty_games(self):
        """Handles empty recent games."""
        result = compute_schedule_context([])
        assert result["days_rest"] is None
        assert result["streak"] == "N/A"
        assert result["games_last_7_days"] == 0


class TestComputeH2hPatterns:
    """Tests for compute_h2h_patterns function."""

    def test_returns_none_for_no_results(self):
        """Returns None for empty results."""
        assert compute_h2h_patterns(None) is None
        assert compute_h2h_patterns({}) is None

    def test_computes_avg_total(self):
        """Computes average combined score."""
        h2h = {
            2024: [
                {"home_team": "A", "home_points": 110, "visitor_points": 105,
                 "winner": "A", "point_diff": 5},
                {"home_team": "B", "home_points": 120, "visitor_points": 115,
                 "winner": "B", "point_diff": 5},
            ]
        }
        result = compute_h2h_patterns(h2h)
        # (215 + 235) / 2 = 225
        assert result["avg_total"] == 225.0

    def test_computes_home_win_pct(self):
        """Computes home team win percentage."""
        h2h = {
            2024: [
                {"home_team": "A", "home_points": 110, "visitor_points": 105,
                 "winner": "A", "point_diff": 5},
                {"home_team": "B", "home_points": 100, "visitor_points": 105,
                 "winner": "A", "point_diff": -5},  # Away team won
            ]
        }
        result = compute_h2h_patterns(h2h)
        assert result["home_win_pct"] == 0.5

    def test_computes_high_scoring_pct(self):
        """Computes percentage of games over 220."""
        h2h = {
            2024: [
                {"home_team": "A", "home_points": 115, "visitor_points": 110,
                 "winner": "A", "point_diff": 5},  # 225 - high scoring
                {"home_team": "B", "home_points": 100, "visitor_points": 95,
                 "winner": "B", "point_diff": 5},  # 195 - not high scoring
            ]
        }
        result = compute_h2h_patterns(h2h)
        assert result["high_scoring_pct"] == 0.5

    def test_computes_close_game_pct(self):
        """Computes percentage of close games (margin <= 5)."""
        h2h = {
            2024: [
                {"home_team": "A", "home_points": 110, "visitor_points": 108,
                 "winner": "A", "point_diff": 2},  # Close
                {"home_team": "B", "home_points": 120, "visitor_points": 100,
                 "winner": "B", "point_diff": 20},  # Not close
            ]
        }
        result = compute_h2h_patterns(h2h)
        assert result["close_game_pct"] == 0.5


class TestComputeH2hMatchupStats:
    """Tests for compute_h2h_matchup_stats function."""

    def test_returns_none_for_no_results(self):
        """Returns None for empty results."""
        assert compute_h2h_matchup_stats(None, "A", "B") is None

    def test_returns_none_for_no_box_scores(self):
        """Returns None when no games have box scores."""
        h2h = {2024: [{"home_team": "A", "visitor_team": "B"}]}
        result = compute_h2h_matchup_stats(h2h, "A", "B")
        assert result is None

    def test_aggregates_team_stats(self):
        """Aggregates stats for each team from box scores."""
        h2h = {
            2024: [
                {
                    "home_team": "Hawks", "visitor_team": "76ers",
                    "home_statistics": {
                        "fgp": "48.0", "tpp": "36.0", "totReb": 45,
                        "assists": 25, "turnovers": 12, "steals": 8, "blocks": 5
                    },
                    "visitor_statistics": {
                        "fgp": "45.0", "tpp": "34.0", "totReb": 42,
                        "assists": 22, "turnovers": 14, "steals": 6, "blocks": 4
                    },
                },
                {
                    "home_team": "76ers", "visitor_team": "Hawks",
                    "home_statistics": {
                        "fgp": "46.0", "tpp": "35.0", "totReb": 44,
                        "assists": 24, "turnovers": 13, "steals": 7, "blocks": 5
                    },
                    "visitor_statistics": {
                        "fgp": "50.0", "tpp": "38.0", "totReb": 46,
                        "assists": 26, "turnovers": 11, "steals": 9, "blocks": 6
                    },
                },
            ]
        }
        result = compute_h2h_matchup_stats(h2h, "Hawks", "76ers")

        # Hawks: game 1 home (48, 36), game 2 away (50, 38)
        assert result["team1"]["avg_fgp"] == 49.0  # (48 + 50) / 2
        # 76ers: game 1 away (45, 34), game 2 home (46, 35)
        assert result["team2"]["avg_fgp"] == 45.5  # (45 + 46) / 2


class TestComputeRecentH2h:
    """Tests for compute_recent_h2h function."""

    @patch("helpers.matchup.get_current_nba_season_year")
    def test_returns_none_in_offseason(self, mock_season):
        """Returns None when in off-season."""
        mock_season.return_value = None
        result = compute_recent_h2h({2024: []}, "A", "A")
        assert result is None

    @patch("helpers.matchup.get_current_nba_season_year")
    def test_filters_to_last_2_seasons(self, mock_season):
        """Only includes games from last 2 seasons."""
        mock_season.return_value = 2024
        h2h = {
            2024: [{"winner": "A", "home_team": "A"}],
            2023: [{"winner": "B", "home_team": "B"}],
            2022: [{"winner": "A", "home_team": "A"}],  # Should be excluded
        }
        result = compute_recent_h2h(h2h, "A", "A")
        assert result["games_last_2_seasons"] == 2

    @patch("helpers.matchup.get_current_nba_season_year")
    def test_computes_recent_wins(self, mock_season):
        """Computes wins for each team in recent games."""
        mock_season.return_value = 2024
        h2h = {
            2024: [
                {"winner": "Hawks", "home_team": "Hawks"},
                {"winner": "76ers", "home_team": "76ers"},
            ],
            2023: [
                {"winner": "Hawks", "home_team": "Hawks"},
            ],
        }
        result = compute_recent_h2h(h2h, "Hawks", "Hawks")
        assert result["team1_wins_last_2_seasons"] == 2
        assert result["team2_wins_last_2_seasons"] == 1


class TestComputeTotalsAnalysis:
    """Tests for compute_totals_analysis function."""

    @pytest.fixture
    def team_snapshots(self):
        """Team snapshots for totals analysis."""
        team1 = {
            "name": "Hawks", "ppg": 115.0, "opp_ppg": 110.0,
            "ortg": 114.0, "drtg": 110.0, "pace": 102.0
        }
        team2 = {
            "name": "76ers", "ppg": 112.0, "opp_ppg": 108.0,
            "ortg": 113.0, "drtg": 108.0, "pace": 100.0
        }
        return team1, team2

    def test_computes_expected_total_without_h2h(self, team_snapshots):
        """Expected total computed from current PPG when no H2H."""
        team1, team2 = team_snapshots
        result = compute_totals_analysis(team1, team2, None, None, [], [])

        # Current total = 115 + 112 = 227
        # H2H weight = 0.2 (no H2H), baseline = 225
        # Expected = 227 * 0.8 + 225 * 0.2 = 181.6 + 45 = 226.6
        assert result["expected_total"] == 226.6

    def test_computes_expected_total_with_h2h(self, team_snapshots):
        """Expected total weighted with H2H average."""
        team1, team2 = team_snapshots
        h2h_summary = {
            "avg_total_points": 220.0,
            "team1_avg_points": 108.0,
            "team2_avg_points": 112.0,
        }
        result = compute_totals_analysis(team1, team2, h2h_summary, None, [], [])

        # Current total = 227
        # H2H weight = 0.4
        # Expected = 227 * 0.6 + 220 * 0.4 = 136.2 + 88 = 224.2
        assert result["expected_total"] == 224.2

    def test_computes_pace_adjusted_total(self, team_snapshots):
        """Pace-adjusted total computed from combined pace and ORTG."""
        team1, team2 = team_snapshots
        result = compute_totals_analysis(team1, team2, None, None, [], [])

        # Combined pace = (102 + 100) / 2 = 101
        # Combined ORTG = (114 + 113) / 2 = 113.5
        # Pace adjusted = 101 * 113.5 / 100 = 114.635 ≈ 114.6
        assert result["pace_adjusted_total"] == 114.6

    def test_computes_defense_factor(self, team_snapshots):
        """Defense factor is average of both teams' DRTG."""
        team1, team2 = team_snapshots
        result = compute_totals_analysis(team1, team2, None, None, [], [])

        # (110 + 108) / 2 = 109
        assert result["defense_factor"] == 109.0


class TestBuildTeamPlayers:
    """Tests for build_team_players function."""

    @pytest.fixture
    def sample_players(self):
        """Sample processed player stats."""
        return [
            {"name": "Trae Young", "ppg": 28.0, "apg": 10.5, "mpg": 35.0,
             "plus_minus": 5.0, "games": 40},
            {"name": "Dejounte Murray", "ppg": 22.0, "apg": 6.5, "mpg": 34.0,
             "plus_minus": 3.0, "games": 38},
            {"name": "De'Andre Hunter", "ppg": 15.0, "apg": 2.0, "mpg": 30.0,
             "plus_minus": 1.0, "games": 35},
            {"name": "John Collins", "ppg": 13.0, "apg": 1.5, "mpg": 28.0,
             "plus_minus": -1.0, "games": 40},
            {"name": "Clint Capela", "ppg": 10.0, "apg": 1.0, "mpg": 26.0,
             "plus_minus": 2.0, "games": 40},
            {"name": "Bogdan Bogdanovic", "ppg": 12.0, "apg": 3.0, "mpg": 24.0,
             "plus_minus": 0.0, "games": 25},  # Limited availability
            {"name": "Onyeka Okongwu", "ppg": 8.0, "apg": 1.0, "mpg": 20.0,
             "plus_minus": 4.0, "games": 40},
            {"name": "Jalen Johnson", "ppg": 6.0, "apg": 1.5, "mpg": 18.0,
             "plus_minus": -2.0, "games": 40},
        ]

    def test_returns_none_for_empty_players(self):
        """Returns None for empty players list."""
        assert build_team_players([], 40, 110.0) is None

    def test_builds_rotation(self, sample_players):
        """Builds rotation with top N players by MPG."""
        result = build_team_players(sample_players, 40, 110.0, rotation_size=6)

        assert len(result["rotation"]) == 6
        assert result["rotation"][0]["name"] == "Trae Young"

    def test_identifies_availability_concerns(self, sample_players):
        """Identifies players with limited availability."""
        result = build_team_players(sample_players, 40, 110.0)

        # Bogdan: 25/40 = 62.5% < 70% threshold
        assert len(result["availability_concerns"]) > 0
        assert any("Bogdan" in c for c in result["availability_concerns"])

    def test_full_strength_when_no_concerns(self):
        """full_strength is True when no availability concerns."""
        players = [
            {"name": "Player 1", "ppg": 20.0, "apg": 5.0, "mpg": 30.0,
             "plus_minus": 2.0, "games": 40}
        ]
        result = build_team_players(players, 40, 100.0)
        assert result["full_strength"] is True

    def test_identifies_top_scorers(self, sample_players):
        """Identifies top 3 scorers."""
        result = build_team_players(sample_players, 40, 110.0)
        assert "Young 28.0" in result["top_scorers"]
        assert "Murray 22.0" in result["top_scorers"]

    def test_identifies_playmaker(self, sample_players):
        """Identifies top playmaker by APG."""
        result = build_team_players(sample_players, 40, 110.0)
        assert "Trae Young" in result["playmaker"]
        assert "10.5 APG" in result["playmaker"]

    def test_identifies_hot_hand(self, sample_players):
        """Identifies player with best plus/minus."""
        result = build_team_players(sample_players, 40, 110.0)
        # Trae Young has highest +/- at 5.0
        assert "Young" in result["hot_hand"]
        assert "+5.0" in result["hot_hand"]

    def test_computes_star_dependency(self, sample_players):
        """Computes star dependency as top scorer's share of team PPG."""
        result = build_team_players(sample_players, 40, 110.0)
        # Trae 28.0 / 110.0 = 25.45%
        assert result["star_dependency"] == 25.5

    def test_computes_bench_scoring(self, sample_players):
        """Computes bench scoring from players 6-8."""
        result = build_team_players(sample_players, 40, 110.0)
        # Players 6-8: Bogdan (12) + Onyeka (8) + Jalen (6) = 26
        assert result["bench_scoring"] == 26.0
