import json
import logging
import re
from argparse import ArgumentParser
from base64 import b64encode
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree

from filelock import FileLock
from openai import APIError, APIStatusError, APITimeoutError
from resvg_py import svg_to_bytes

from llm_gateway import LLMConfigurationError, ModelClient, complete_json, llm_client


LOGGER = logging.getLogger("uvicorn.error")
SVG_TAG = "{http://www.w3.org/2000/svg}svg"
REFERENCE_VARIANT = "L_main"
DESCRIPTION_PROMPT = (
    "Заполни description коротким описанием объекта на русском, в одном или двух "
    "предложениях. Начни сразу с предмета, без фраз вроде «На изображении изображена». "
    "Опиши его форму, ракурс и заметные детали, включая символы на поверхности. "
    "Палитра в редакторе меняется. Опускай цвета, фон и возможные ассоциации. "
    "Укажи 1–4 смысловые точки anchors внутри видимого предмета для размещения "
    "других объектов поверх него. Выбирай центры узнаваемых частей или свободных "
    "поверхностей, избегай падающей тени и пустых полей. Дай каждой точке короткое "
    "английское имя в snake_case. Для раскрытой книги используй on_left_page "
    "и on_right_page в центрах видимых страниц. Если выделяется только одна "
    "поверхность, достаточно одной точки. x — горизонталь, y — вертикаль. "
    "Координаты целые от 0 до 1000 относительно всего изображения, с началом "
    "в левом верхнем углу. x растёт вправо, y растёт вниз."
)
VISION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "svg_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["description", "anchors"],
            "additionalProperties": False,
            "properties": {
                "description": {"type": "string"},
                "anchors": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "required": ["name", "x", "y"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                        },
                    },
                },
            },
        },
    },
}


class Size(StrEnum):
    L = "L"
    S = "S"


class Role(StrEnum):
    MAIN = "main"
    BACKGROUND = "bg"


@dataclass(frozen=True)
class SvgAsset:
    path: Path
    size: Size
    role: Role
    view_box: tuple[float, float, float, float]
    content: bytes


@dataclass(frozen=True)
class Anchor:
    x: float
    y: float


@dataclass(frozen=True)
class VariantFile:
    file: str
    width: float
    height: float


@dataclass(frozen=True)
class CatalogEntry:
    description: str | None
    anchors: dict[str, Anchor]
    variants: dict[str, VariantFile]
    source_sha256: str | None


def scan_assets(root: Path) -> dict[str, dict[str, SvgAsset]]:
    assets: dict[str, dict[str, SvgAsset]] = {}
    for size in Size:
        for role in Role:
            variant = f"{size}_{role}"
            for path in sorted((root / size / role).glob("*")):
                if path.suffix.lower() != ".svg" or not path.is_file():
                    continue
                try:
                    content = path.read_bytes()
                    svg = ElementTree.fromstring(content)
                    if svg.tag != SVG_TAG:
                        raise ValueError("expected an SVG document")
                    x, y, width, height = map(float, svg.attrib["viewBox"].replace(",", " ").split())
                    if not all(map(isfinite, (x, y, width, height))) or min(width, height) <= 0:
                        raise ValueError("invalid viewBox")
                except (OSError, ElementTree.ParseError, KeyError, ValueError) as error:
                    LOGGER.warning("Skipping SVG %s: %s", path, error)
                    continue
                group = assets.setdefault(path.stem, {})
                if variant in group:
                    raise ValueError(f"Duplicate SVG for {size}/{role}/{path.stem}")
                group[variant] = SvgAsset(path, size, role, (x, y, width, height), content)
    return assets


def save_catalog(path: Path, catalog: dict[str, CatalogEntry]) -> None:
    with NamedTemporaryFile(
        dir=path.parent, prefix=".catalog-", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        records = {asset_id: asdict(entry) for asset_id, entry in catalog.items()}
        content = json.dumps(records, ensure_ascii=False, indent=2)
        temporary.write_text(content + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def svg_preview(content: bytes, width: float, height: float, pixels: int) -> str:
    image_size = {"width": pixels} if width >= height else {"height": pixels}
    png = svg_to_bytes(svg_string=content.decode("utf-8-sig"), **image_size)
    return "data:image/png;base64," + b64encode(png).decode("ascii")


def describe_svg(client: ModelClient, asset: SvgAsset) -> tuple[str, dict[str, Anchor]]:
    image_url = svg_preview(asset.content, *asset.view_box[2:], pixels=768)
    content = complete_json(client, [{
        "role": "user",
        "content": [
            {"type": "text", "text": DESCRIPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }], VISION_RESPONSE_FORMAT)
    try:
        analysis = json.loads(content)
        description = analysis["description"]
        points = analysis["anchors"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Vision returned an empty description")
        if not isinstance(points, list) or not 1 <= len(points) <= 4:
            raise ValueError("Expected one to four anchors")
        anchors = {}
        for point in points:
            name, x, y = point["name"], point["x"], point["y"]
            if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) or name in anchors:
                raise ValueError("Expected unique anchor names in snake_case")
            if not all(type(coord) is int and 0 <= coord <= 1000 for coord in (x, y)):
                raise ValueError("Expected integer anchor coordinates between 0 and 1000")
            anchors[name] = Anchor(x / 1000, y / 1000)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Vision returned an unexpected response") from error
    return description.strip(), anchors


def refresh_catalog(
    assets: dict[str, dict[str, SvgAsset]], asset_root: Path, path: Path, *, rebuild: bool = False
) -> dict[str, CatalogEntry]:
    with FileLock(f"{path}.lock"):
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not rebuild else {}
        previous = {}
        analyses: dict[str, tuple[str, dict[str, Anchor]]] = {}
        for asset_id, record in records.items():
            anchors = {name: Anchor(**point) for name, point in record.get("anchors", {}).items()}
            variants = {
                key: VariantFile(**value)
                for key, value in record["variants"].items() if isinstance(value, dict)
            }
            entry = CatalogEntry(record["description"], anchors, variants, record["source_sha256"])
            previous[asset_id] = entry
            if entry.source_sha256 and entry.description and entry.anchors:
                analyses[entry.source_sha256] = (entry.description, entry.anchors)
        catalog = {}
        pending: dict[str, tuple[SvgAsset, str]] = {}
        for asset_id, group in sorted(assets.items()):
            entry = previous.get(asset_id, CatalogEntry(None, {}, {}, None))
            variants = {
                key: VariantFile(asset.path.relative_to(asset_root).as_posix(), *asset.view_box[2:])
                for key, asset in group.items()
            }
            entry = replace(entry, variants=variants)
            reference = group.get(REFERENCE_VARIANT)
            if reference is not None:
                fingerprint = sha256(reference.content).hexdigest()
                if fingerprint != entry.source_sha256 or not entry.description or not entry.anchors:
                    if fingerprint in analyses:
                        description, anchors = analyses[fingerprint]
                        entry = replace(entry, description=description, anchors=anchors, source_sha256=fingerprint)
                    else:
                        pending[asset_id] = (reference, fingerprint)
            catalog[asset_id] = entry

        if rebuild or catalog != previous or not path.exists():
            save_catalog(path, catalog)
        if not pending:
            return catalog

        LOGGER.info("Analyzing %d objects with Vision", len(pending))
        consecutive_unavailable = 0
        saved = 0
        with ExitStack() as stack:
            try:
                client = stack.enter_context(llm_client())
            except LLMConfigurationError as error:
                LOGGER.warning("Vision update deferred: %s", error)
                return catalog
            for asset_id, (asset, fingerprint) in pending.items():
                try:
                    description, anchors = analyses.get(fingerprint) or describe_svg(client, asset)
                except APIError as error:
                    if isinstance(error, APIStatusError):
                        status = error.status_code
                        reason = f"HTTP {status}"
                        temporary = status in (502, 503, 504)
                    else:
                        reason = type(error).__name__
                        temporary = isinstance(error, APITimeoutError)
                    consecutive_unavailable += 1
                    if temporary and consecutive_unavailable < 3:
                        LOGGER.warning("Vision deferred %s (%s)", asset.path.name, reason)
                        continue
                    LOGGER.warning("Vision update paused at %s (%s)", asset.path.name, reason)
                    break
                except ValueError as error:
                    LOGGER.warning("Skipping Vision analysis for %s: %s", asset.path.name, error)
                    continue
                consecutive_unavailable = 0
                analyses[fingerprint] = (description, anchors)
                catalog[asset_id] = replace(
                    catalog[asset_id], description=description, anchors=anchors, source_sha256=fingerprint
                )
                save_catalog(path, catalog)
                saved += 1
                LOGGER.info("Saved %s with %d anchors", asset_id, len(anchors))
        LOGGER.info("Saved %d of %d requested Vision analyses", saved, len(pending))
        return catalog


if __name__ == "__main__":
    parser = ArgumentParser(description="Build the SVG catalog from raw_assets.")
    parser.add_argument("--rebuild", action="store_true", help="Replace the catalog and analyze all L/main references.")
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    project = Path(__file__).resolve().parent
    raw = project / "raw_assets"
    groups = scan_assets(raw)
    entries = refresh_catalog(groups, raw, project / "catalog.json", rebuild=options.rebuild)
    pending = [
        asset_id for asset_id, entry in entries.items()
        if REFERENCE_VARIANT in groups[asset_id] and (
            entry.source_sha256 != sha256(groups[asset_id][REFERENCE_VARIANT].content).hexdigest()
            or not entry.description or not entry.anchors
        )
    ]
    if pending:
        LOGGER.warning("Pending Vision analyses: %s", ", ".join(pending))
        raise SystemExit(1)
