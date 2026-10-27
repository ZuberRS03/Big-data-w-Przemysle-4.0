import pytest

from src.pklot import read_annotations


CONTOUR = '<contour><point x="0" y="0"/><point x="10" y="0"/><point x="10" y="20"/><point x="0" y="20"/></contour>'
RECTANGLE = '<rotatedRect><center x="50" y="60"/><size w="20" h="10"/><angle d="90"/></rotatedRect>'


def parse(tmp_path, content):
    path = tmp_path / "sample.xml"
    path.write_text(f"<parking>{content}</parking>", encoding="utf-8")
    return read_annotations(path)


@pytest.mark.parametrize("attribute,expected,issue", [
    ('occupied="0"', 0, None), ('occupied="1"', 1, None),
    ("", None, "missing_occupancy"), ('occupied="2"', None, "invalid_occupancy"),
])
def test_occupancy_is_never_guessed(tmp_path, attribute, expected, issue):
    space = parse(tmp_path, f'<space id="1" {attribute}>{CONTOUR}</space>')[0]
    assert space.occupied == expected
    assert space.issues == (() if issue is None else (issue,))


def test_contour_takes_precedence(tmp_path):
    space = parse(tmp_path, f'<space id="1" occupied="0">{RECTANGLE}{CONTOUR}</space>')[0]
    assert space.geometry_source == "contour"
    assert space.polygon == ((0, 0), (10, 0), (10, 20), (0, 20))


def test_rotated_rectangle_fallback(tmp_path):
    space = parse(tmp_path, f'<space id="1" occupied="1">{RECTANGLE}</space>')[0]
    assert space.geometry_source == "rotatedRect"
    for actual, expected in zip(space.polygon, ((55, 50), (55, 70), (45, 70), (45, 50))):
        assert actual == pytest.approx(expected)


def test_uppercase_points_in_original_ufpr04(tmp_path):
    geometry = CONTOUR.replace("point", "Point")
    space = parse(tmp_path, f'<space id="1" occupied="0">{geometry}</space>')[0]
    assert space.geometry_source == "contour"
    assert space.polygon == ((0, 0), (10, 0), (10, 20), (0, 20))
    assert not space.issues


@pytest.mark.parametrize("geometry", ["", '<contour><point x="1" y="1"/></contour>', CONTOUR.replace('x="10"', 'x="nan"'), RECTANGLE.replace('w="20"', 'w="-20"')])
def test_invalid_geometry_reported(tmp_path, geometry):
    space = parse(tmp_path, f'<space id="1" occupied="1">{geometry}</space>')[0]
    assert space.occupied == 1
    assert space.polygon == ()
    assert "invalid_geometry" in space.issues


def test_duplicate_identifiers_reported(tmp_path):
    spaces = parse(tmp_path, f'<space id="1" occupied="0">{CONTOUR}</space>' * 2)
    assert "duplicate_id" in spaces[1].issues
