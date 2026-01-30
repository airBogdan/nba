# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Analytics tool that fetches data from the API-Sports NBA API and generates matchup analysis between two teams. Outputs JSON with team statistics, head-to-head history, player data, and contextual signals.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main analysis (outputs to output/matchup_analysis.json)
python main.py

# Run tests
pytest

# Run a single test
pytest tests/test_file.py::test_name -v
```

## Environment Setup

Requires `NBA_RAPID_API_KEY` environment variable for API access. Use a `.env` file or export directly.

## Architecture

The codebase is async Python using aiohttp for API calls.

**Data flow:**
1. `main.py` orchestrates the analysis pipeline
2. `helpers/api.py` handles all API requests to `v2.nba.api-sports.io` and contains data processing functions for player/team stats
3. `helpers/teams.py` fetches and processes team standings across seasons
4. `helpers/games.py` handles head-to-head game history and box score enrichment
5. `helpers/matchup.py` is the core analysis engine - builds team snapshots, computes comparison edges, and generates betting/analysis signals

**Key data structures (all TypedDict-based):**
- `ProcessedPlayerStats` / `ProcessedTeamStats` - aggregated per-game statistics
- `H2HResults` - maps season year to list of games with optional box score data
- `MatchupAnalysis` - final output containing snapshots, comparisons, totals analysis, and signals
- `TeamSnapshot` - current season state with computed ORTG/DRTG estimates

**Season logic:**
`helpers/utils.py::get_current_nba_season_year()` determines season year based on current month (Sep-Dec = current year, Jan-May = previous year, Jun-Aug = None).