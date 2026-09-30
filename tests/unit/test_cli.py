"""Unit tests for the command-line interface.

Only the argument parser and the pure helpers are exercised here; the
subcommands that need a live provider are covered by the integration suite.
"""

from __future__ import annotations

import pytest

from mcp_for_copilot.cli import _build_parser


class TestParser:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_chat_requires_a_prompt(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["chat"])

    def test_chat_defaults(self) -> None:
        args = _build_parser().parse_args(["chat", "hello"])
        assert args.prompt == "hello"
        assert args.system is None
        assert args.model is None
        assert args.json is False
        assert args.plan is False

    def test_chat_plan_flag(self) -> None:
        args = _build_parser().parse_args(["chat", "hello", "--plan"])
        assert args.plan is True

    def test_chat_json_flag(self) -> None:
        args = _build_parser().parse_args(["chat", "hello", "--json"])
        assert args.json is True

    def test_marathon_requires_a_question(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["marathon"])

    def test_marathon_accepts_several_questions(self) -> None:
        args = _build_parser().parse_args(["marathon", "one", "two", "three"])
        assert args.questions == ["one", "two", "three"]

    def test_marathon_defaults(self) -> None:
        args = _build_parser().parse_args(["marathon", "one"])
        assert args.interval == 2.0
        assert args.timeout == 600.0
        assert args.json is False

    def test_marathon_overrides(self) -> None:
        args = _build_parser().parse_args(
            ["marathon", "one", "--interval", "0.5", "--timeout", "30", "--json"]
        )
        assert args.interval == 0.5
        assert args.timeout == 30.0
        assert args.json is True

    def test_known_subcommands(self) -> None:
        parser = _build_parser()
        required = {"chat": ["x"], "marathon": ["x"]}
        for name in ("serve", "mcp", "chat", "marathon", "models", "status"):
            args = parser.parse_args([name, *required.get(name, [])])
            assert args.command == name
