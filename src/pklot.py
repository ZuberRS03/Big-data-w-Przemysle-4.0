"""Read PKLot annotations without guessing missing occupancy labels."""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ParkingSpace:
    identifier: str
    occupied: int | None
    polygon: tuple[tuple[float, float], ...]
    geometry_source: str
    issues: tuple[str, ...]


def _polygon(space: ET.Element) -> tuple[tuple[tuple[float, float], ...], str]:
    contour = space.find("contour")
    if contour is not None and len(contour):
        # Both spellings occur in the original dataset (including UFPR04).
        points = tuple(
            (float(p.attrib["x"]), float(p.attrib["y"]))
            for p in contour if p.tag in {"point", "Point"}
        )
        source = "contour"
    else:
        rect = space.find("rotatedRect")
        if rect is None:
            raise ValueError("Missing contour and rotatedRect")
        center, size, angle = rect.find("center"), rect.find("size"), rect.find("angle")
        if center is None or size is None or angle is None:
            raise ValueError("Incomplete rotatedRect")
        cx, cy = float(center.attrib["x"]), float(center.attrib["y"])
        w, h = float(size.attrib["w"]), float(size.attrib["h"])
        degrees = float(angle.attrib["d"])
        if not all(math.isfinite(v) for v in (cx, cy, w, h, degrees)) or w <= 0 or h <= 0:
            raise ValueError("Invalid rotatedRect dimensions or coordinates")
        a = math.radians(degrees)
        points = tuple(
            (cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a))
            for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
        )
        source = "rotatedRect"
    if len(points) < 3 or not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("Invalid polygon coordinates")
    twice_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )
    if abs(twice_area) < 1e-6:
        raise ValueError("Zero-area polygon")
    return points, source


def read_annotations(path: Path) -> list[ParkingSpace]:
    root = ET.parse(path).getroot()
    if root.tag != "parking":
        raise ValueError(f"Unexpected XML root: {root.tag}")
    result = []
    identifiers = set()
    for element in root.findall("space"):
        issues = []
        identifier = element.get("id", "")
        if not identifier:
            issues.append("missing_id")
        if identifier in identifiers:
            issues.append("duplicate_id")
        identifiers.add(identifier)
        raw = element.get("occupied")
        occupied = int(raw) if raw in {"0", "1"} else None
        if occupied is None:
            issues.append("missing_occupancy" if raw is None else "invalid_occupancy")
        try:
            polygon, source = _polygon(element)
        except (KeyError, ValueError, OverflowError):
            polygon, source = (), "invalid"
            issues.append("invalid_geometry")
        result.append(ParkingSpace(identifier, occupied, polygon, source, tuple(issues)))
    return result
