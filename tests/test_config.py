from pathlib import Path

from agentfuse.config import FuseConfig, load_config


def test_defaults_when_no_file() -> None:
    cfg = load_config(None)
    assert cfg == FuseConfig()
    assert cfg.budget_per_run == 5.0
    assert cfg.loop_threshold == 4


def test_partial_toml_overrides(tmp_path: Path) -> None:
    p = tmp_path / "fuse.toml"
    p.write_text('[budget]\nper_run = 2.5\n[policies.loop]\nthreshold = 7\n')
    cfg = load_config(p)
    assert cfg.budget_per_run == 2.5
    assert cfg.loop_threshold == 7
    assert cfg.rate_calls_per_minute == 30  # untouched default


def test_loop_volatile_keys_default_and_override(tmp_path: Path) -> None:
    assert "timestamp" in FuseConfig().loop_volatile_keys
    p = tmp_path / "fuse.toml"
    p.write_text('[policies.loop]\nvolatile_keys = ["session_id"]\n')
    assert load_config(p).loop_volatile_keys == ("session_id",)
