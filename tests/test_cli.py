from click.testing import CliRunner

from agentfuse.cli import main


def test_policies_lists_all_four() -> None:
    result = CliRunner().invoke(main, ["policies"])
    assert result.exit_code == 0
    for name in ("loop", "budget", "stall", "rate"):
        assert name in result.output


def test_serve_is_registered() -> None:
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0 and "--port" in result.output
