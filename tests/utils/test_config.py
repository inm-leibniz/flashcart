import pytest

from flashcart.utils.config import parse_config_args, parse_override


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        pytest.param("n=5", ("n", 5), id="int"),
        pytest.param("lr=1e-3", ("lr", 1.0e-3), id="float"),
        pytest.param("flag=true", ("flag", True), id="bool_true"),
        pytest.param("flag=False", ("flag", False), id="bool_false"),
        pytest.param("x=none", ("x", None), id="none"),
        pytest.param("x=null", ("x", None), id="null"),
        pytest.param("elements=[H, O]", ("elements", ["H", "O"]), id="yaml_list"),
        pytest.param("name=foo", ("name", "foo"), id="string"),
        pytest.param("path=models/0001", ("path", "models/0001"), id="path_string"),
    ],
)
def test_parse_override_matches_python_literals(override, expected):
    key, value = parse_override(override)
    assert (key, value) == expected
    assert type(value) is type(expected[1])


def test_parse_override_raises_on_missing_equals():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_override("no_equals_sign")


def test_parse_config_args_overrides_yaml_file(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("lr: 1.0e-2\nmax_epochs: 100\nelements: [H]\n")

    merged = parse_config_args([str(config), "lr=5e-3", "n_radial=8"])

    assert merged["lr"] == 5.0e-3
    assert merged["max_epochs"] == 100
    assert merged["elements"] == ["H"]
    assert merged["n_radial"] == 8
