"""Offline coverage for the hand-rolled .env loader in config.py.

config.py deliberately parses .env itself rather than take a dependency, and it's
the single place all configuration enters the app — so a quiet failure here
misconfigures everything downstream. The regression that prompted these: the path
defaulted to a bare ".env", making it CWD-relative, so launching from anywhere but
the repo root silently skipped the file. That isn't a partial config, it's a
*wrong* one — MODEL falls back to "" (auto-detect whatever model is loaded) and
NO_THINK to False (reasoning back ON), the opposite of the production setup.

Run:  python tests/test_config.py
"""
import os

from _harness import case, run, temp_dir  # also puts the repo root on sys.path

import config


def _write(text: str) -> str:
    path = os.path.join(temp_dir(), ".env")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _clean(*keys):
    for k in keys:
        os.environ.pop(k, None)


@case
async def loads_keys_and_strips_quotes():
    _clean("T_PLAIN", "T_DQ", "T_SQ", "T_SPACED")
    config._load_dotenv(_write(
        'T_PLAIN=hello\nT_DQ="quoted"\nT_SQ=\'single\'\nT_SPACED =  padded  \n'))
    assert os.environ["T_PLAIN"] == "hello"
    assert os.environ["T_DQ"] == "quoted", "double quotes should be stripped"
    assert os.environ["T_SQ"] == "single", "single quotes should be stripped"
    assert os.environ["T_SPACED"] == "padded", "whitespace around key/value trimmed"
    _clean("T_PLAIN", "T_DQ", "T_SQ", "T_SPACED")


@case
async def skips_comments_blanks_and_malformed_lines():
    _clean("T_REAL", "T_HASH")
    config._load_dotenv(_write(
        "\n# a comment\n   \n#T_HASH=nope\nnot_a_pair_line\nT_REAL=yes\n"))
    assert os.environ["T_REAL"] == "yes"
    assert "T_HASH" not in os.environ, "commented lines must not be set"
    assert "not_a_pair_line" not in os.environ
    _clean("T_REAL")


@case
async def existing_environment_wins():
    # setdefault semantics, matching real dotenv libraries: a value already exported
    # by the shell must beat the file, or a deliberate override gets silently ignored.
    _clean("T_PRECEDENCE")
    os.environ["T_PRECEDENCE"] = "from_shell"
    config._load_dotenv(_write("T_PRECEDENCE=from_file\n"))
    assert os.environ["T_PRECEDENCE"] == "from_shell"
    _clean("T_PRECEDENCE")


@case
async def missing_file_is_a_noop():
    config._load_dotenv(os.path.join(temp_dir(), "definitely-absent"))  # must not raise


@case
async def flag_accepts_the_usual_truthy_spellings():
    for raw, want in [("1", True), ("true", True), ("TRUE", True), ("Yes", True),
                      ("on", True), (" true ", True),
                      ("0", False), ("false", False), ("no", False), ("", False),
                      ("maybe", False)]:
        os.environ["T_FLAG"] = raw
        assert config._flag("T_FLAG", False) is want, f"{raw!r} -> expected {want}"
    _clean("T_FLAG")
    assert config._flag("T_FLAG", True) is True, "unset falls back to the default"
    assert config._flag("T_FLAG", False) is False


@case
async def numeric_knobs_parse_and_fall_back():
    _clean("T_NUM")
    assert config._f("T_NUM", 0.8) == 0.8, "unset -> default"
    assert config._i("T_NUM", 3) == 3
    os.environ["T_NUM"] = " 0.25 "
    assert config._f("T_NUM", 0.8) == 0.25, "whitespace tolerated"
    os.environ["T_NUM"] = "7"
    assert config._i("T_NUM", 3) == 7
    _clean("T_NUM")


@case
async def a_malformed_number_names_the_offending_key():
    # It used to surface as a bare "could not convert string to float: 'abc'" at
    # import, with no clue which of ~40 knobs was wrong.
    os.environ["TEMPERATURE_TEST"] = "abc"
    try:
        config._f("TEMPERATURE_TEST", 0.8)
        raise AssertionError("expected a ConfigError")
    except config.ConfigError as e:
        assert "TEMPERATURE_TEST" in str(e), f"error must name the key: {e}"
        assert "abc" in str(e), f"error must show the bad value: {e}"
    finally:
        _clean("TEMPERATURE_TEST")


@case
async def default_path_is_anchored_to_the_repo_not_the_cwd():
    """The actual regression: a bare ".env" default made this CWD-dependent."""
    import inspect

    default = inspect.signature(config._load_dotenv).parameters["path"].default
    assert os.path.isabs(default), f"default .env path must be absolute, got {default!r}"
    assert default.endswith(".env")
    # ...and it must sit next to config.py, which lives at the repo root.
    assert os.path.dirname(default) == os.path.dirname(os.path.abspath(config.__file__))


if __name__ == "__main__":
    raise SystemExit(run())
