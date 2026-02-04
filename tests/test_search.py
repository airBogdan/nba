"""Tests for web search enrichment strategies and compact_json."""

from unittest.mock import AsyncMock, patch

import pytest

from workflow.prompts import compact_json
from workflow.search import (
    _build_search_summary,
    search_enrich,
    search_strategy_a,
    search_strategy_b,
    search_strategy_c,
)


SAMPLE_GAME_DATA = {
    "matchup": {"home_team": "Lakers", "team1": "Lakers", "team2": "Celtics"},
    "current_season": {
        "team1": {"name": "Lakers", "record": "30-20", "conf_rank": 5, "ppg": 115.2, "ortg": 112.0, "drtg": 108.5},
        "team2": {"name": "Celtics", "record": "35-15", "conf_rank": 2, "ppg": 118.1, "ortg": 115.3, "drtg": 106.2},
    },
    "schedule": {
        "team1": {"streak": "W3", "days_rest": 2, "games_last_7_days": 3},
        "team2": {"streak": "L1", "days_rest": 1, "games_last_7_days": 4},
    },
    "players": {
        "team1": {
            "availability_concerns": ["LeBron James (50/60 games)"],
        },
        "team2": {
            "availability_concerns": [],
        },
    },
}

MATCHUP_STR = "Celtics @ Lakers"


class TestCompactJson:
    def test_removes_whitespace(self):
        result = compact_json({"key": "value", "num": 1})
        assert " " not in result
        assert "\n" not in result

    def test_no_space_after_colon(self):
        result = compact_json({"a": 1})
        assert ": " not in result
        assert '{"a":1}' == result

    def test_no_space_after_comma(self):
        result = compact_json({"a": 1, "b": 2})
        assert ", " not in result

    def test_nested_structures(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        result = compact_json(data)
        assert result == '{"outer":{"inner":[1,2,3]}}'

    def test_preserves_all_data(self):
        import json
        data = {"name": "Lakers", "record": "30-20", "stats": [1.5, 2.0]}
        result = compact_json(data)
        assert json.loads(result) == data

    def test_handles_empty_dict(self):
        assert compact_json({}) == "{}"

    def test_handles_strings_with_spaces(self):
        result = compact_json({"name": "Los Angeles Lakers"})
        assert "Los Angeles Lakers" in result


class TestBuildSearchSummary:
    def test_includes_team_names(self):
        result = _build_search_summary(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert "Lakers" in result
        assert "Celtics" in result

    def test_includes_records(self):
        result = _build_search_summary(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert "30-20" in result
        assert "35-15" in result

    def test_includes_matchup_string(self):
        result = _build_search_summary(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert MATCHUP_STR in result

    def test_includes_availability_concerns(self):
        result = _build_search_summary(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert "LeBron James" in result

    def test_handles_dict_availability_concerns(self):
        """Availability concerns can be dicts (with 'name' key) or plain strings."""
        data = {**SAMPLE_GAME_DATA, "players": {
            "team1": {"availability_concerns": [{"name": "AD", "status": "questionable"}]},
            "team2": {"availability_concerns": []},
        }}
        result = _build_search_summary(data, MATCHUP_STR)
        assert "AD" in result

    def test_includes_streak(self):
        result = _build_search_summary(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert "W3" in result

    def test_reasonable_length(self):
        result = _build_search_summary(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert len(result) < 1000


class TestSearchDispatcher:
    @pytest.mark.asyncio
    async def test_unknown_strategy_returns_none(self):
        result = await search_enrich("z", SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatches_to_strategy_a(self):
        mock_a = AsyncMock(return_value="search results")
        with patch.dict("workflow.search._STRATEGIES", {"a": mock_a}):
            result = await search_enrich("a", SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "search results"
        mock_a.assert_called_once_with(SAMPLE_GAME_DATA, MATCHUP_STR)

    @pytest.mark.asyncio
    async def test_dispatches_to_strategy_b(self):
        mock_b = AsyncMock(return_value="search results b")
        with patch.dict("workflow.search._STRATEGIES", {"b": mock_b}):
            result = await search_enrich("b", SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "search results b"
        mock_b.assert_called_once_with(SAMPLE_GAME_DATA, MATCHUP_STR)

    @pytest.mark.asyncio
    async def test_dispatches_to_strategy_c(self):
        mock_c = AsyncMock(return_value="search results c")
        with patch.dict("workflow.search._STRATEGIES", {"c": mock_c}):
            result = await search_enrich("c", SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "search results c"
        mock_c.assert_called_once_with(SAMPLE_GAME_DATA, MATCHUP_STR)

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        mock_a = AsyncMock(side_effect=Exception("API down"))
        with patch.dict("workflow.search._STRATEGIES", {"a": mock_a}):
            result = await search_enrich("a", SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result is None


class TestStrategyA:
    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_makes_two_calls(self, mock_complete):
        mock_complete.side_effect = [
            "Lakers Celtics injury report odds today",  # query gen
            "Injury: LeBron questionable. Spread: LAL -3.5",  # perplexity
        ]
        result = await search_strategy_a(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "Injury: LeBron questionable. Spread: LAL -3.5"
        assert mock_complete.call_count == 2

        # First call uses system prompt and query model
        first_call = mock_complete.call_args_list[0]
        assert first_call.kwargs["system"] is not None
        assert "haiku" in first_call.kwargs["model"]

        # Second call has no system prompt and uses perplexity model
        second_call = mock_complete.call_args_list[1]
        assert second_call.kwargs.get("system") is None
        assert "perplexity" in second_call.kwargs["model"]

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_none_if_query_gen_fails(self, mock_complete):
        mock_complete.return_value = None
        result = await search_strategy_a(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result is None
        assert mock_complete.call_count == 1


class TestStrategyB:
    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_makes_one_call(self, mock_complete):
        mock_complete.return_value = "Direct search results"
        result = await search_strategy_b(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "Direct search results"
        assert mock_complete.call_count == 1

        # No system prompt, perplexity model
        call = mock_complete.call_args_list[0]
        assert call.kwargs.get("system") is None
        assert "perplexity" in call.kwargs["model"]

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_none_on_failure(self, mock_complete):
        mock_complete.return_value = None
        result = await search_strategy_b(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result is None


class TestStrategyC:
    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_makes_three_calls_when_followup_needed(self, mock_complete):
        mock_complete.side_effect = [
            "Baseline: injuries and odds info",  # template search
            "Lakers Celtics line movement last 24 hours betting trends",  # followup gen
            "Line moved from -3 to -4.5 due to...",  # followup search
        ]
        result = await search_strategy_c(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert mock_complete.call_count == 3
        assert "Baseline: injuries and odds info" in result
        assert "### Additional Context" in result
        assert "Line moved from -3 to -4.5" in result

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_baseline_when_no_followup_needed(self, mock_complete):
        mock_complete.side_effect = [
            "Baseline: complete info",  # template search
            "No follow-up needed",  # followup gen
        ]
        result = await search_strategy_c(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "Baseline: complete info"
        assert mock_complete.call_count == 2

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_baseline_when_followup_gen_fails(self, mock_complete):
        mock_complete.side_effect = [
            "Baseline: complete info",  # template search
            None,  # followup gen fails
        ]
        result = await search_strategy_c(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "Baseline: complete info"
        assert mock_complete.call_count == 2

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_baseline_when_followup_short(self, mock_complete):
        mock_complete.side_effect = [
            "Baseline info",  # template search
            "ok",  # short followup - skip
        ]
        result = await search_strategy_c(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "Baseline info"
        assert mock_complete.call_count == 2

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_none_if_template_fails(self, mock_complete):
        mock_complete.return_value = None
        result = await search_strategy_c(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result is None
        assert mock_complete.call_count == 1

    @pytest.mark.asyncio
    @patch("workflow.search.complete", new_callable=AsyncMock)
    async def test_returns_baseline_when_followup_says_no_additional(self, mock_complete):
        mock_complete.side_effect = [
            "Baseline: all covered",  # template search
            "No additional search is necessary for this matchup",  # no additional
        ]
        result = await search_strategy_c(SAMPLE_GAME_DATA, MATCHUP_STR)
        assert result == "Baseline: all covered"
        assert mock_complete.call_count == 2
