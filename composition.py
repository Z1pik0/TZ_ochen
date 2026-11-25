from __future__ import annotations

import json
import logging
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import (
    BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, StringConstraints, ValidationInfo, field_validator,
)

from asset_catalog import Role, Size, svg_preview
from llm_gateway import ModelClient, complete_json, llm_client

if TYPE_CHECKING:
    from main import Asset, Variant


LOGGER = logging.getLogger("uvicorn.error")
SELECT_PROMPT = (
    "Ты подбираешь объекты из каталога векторных иллюстраций к запросу пользователя. "
    "Для заголовка или метафоры подбирай объекты по смыслу. Обычно достаточно двух-трёх, "
    "для простой идеи может хватить одного. Если подходящих объектов нет, верни пустые списки. "
    "Считай названием конкретного предмета и одиночное слово, если оно обозначает предмет. "
    "Для явного запроса сохрани все названные предметы и количество каждого. "
    "Для повторов указывай тот же ID один раз на каждый экземпляр. Учитывай синонимы "
    "и формы слов. Если явно запрошенного предмета нет в каталоге, запиши его название "
    "в missing_objects. При любом missing_objects верни asset_ids пустым. "
    "Используй только ID из каталога и верни missing_objects пустым при полном совпадении."
)
PLACE_PROMPT = (
    "Собери из выбранных объектов физически правдоподобную композицию по запросу. "
    "Сначала заполни arrangement: кратко опиши, что стоит отдельно, что лежит на опоре "
    "и где задумано перекрытие в проекции. Реализуй явно заданные отношения, а для "
    "остальных пар предпочитай размещение с видимым зазором. Фоновая роль задаёт "
    "монохромный вариант. По умолчанию размести такой предмет на заднем плане с зазором "
    "от основной группы, чтобы его силуэт, включая выступающие детали, читался целиком. "
    "«Рядом» означает небольшой видимый зазор, а «отдельно» — заметное расстояние. "
    "Если пользователь указал зазор, считай его между краями объектов. "
    "Для намеренного перекрытия в проекции оцени смещение и читаемость обоих силуэтов. "
    "Затем рассчитай layers. По превью оцени реальные контуры объектов. Для предмета "
    "на опоре используй нужный anchor как приблизительный ориентир. Отдельно проверь дальний и ближний "
    "концы корпуса в выбранном центре. Если один конец оказывается у границы опоры, "
    "сдвинь предмет глубже на её поверхность, чтобы по краям оставалось видимое поле. "
    "Собери предметы с общей опорой как одну группу и реши, где на холсте будут остальные. "
    "Перед ответом проверь каждую пару независимых объектов по рассчитанным координатам. "
    "Сравни диапазоны x ± width/2 и y ± height/2. Если они пересекаются по обеим осям, "
    "оцени видимые контуры по превью. Если силуэты касаются или перекрываются хотя бы "
    "краями, раздвинь предметы до видимого зазора и повтори проверку. Для задуманных "
    "наложений проверь опору или читаемость перекрытия. Убедись, что сцена помещается на холсте. "
    "В instances перечислены экземпляры. Их id уникальны, asset_id ссылается на objects. "
    "Размести каждый экземпляр, в том числе повторяющиеся объекты. Верни его id ровно один раз "
    "и перечисли слои от нижнего к верхнему. "
    "Выбери role из variants с учётом запроса: main — цветной объект, bg — монохромный. "
    "x и y — координаты центра всего SVG в долях холста, начало в левом верхнем углу. "
    "width и height выбранного варианта — размеры SVG в долях холста. Координаты anchors "
    "заданы в долях ширины и высоты самого SVG. Чтобы найти точку на холсте, прибавь к x "
    "и y опоры смещения (anchor.x - 0.5) * width и (anchor.y - 0.5) * height."
)


class CompositionRequest(BaseModel):
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    size: Size
    width: PositiveInt
    height: PositiveInt


class ModelReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class AssetSelection(ModelReply):
    asset_ids: list[str]
    missing_objects: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]]

    @field_validator("asset_ids")
    @classmethod
    def validate_ids(cls, ids: list[str], info: ValidationInfo) -> list[str]:
        allowed = (info.context or {}).get("asset_ids")
        if allowed is not None and not set(ids) <= allowed:
            raise ValueError("Expected IDs from the catalog")
        return ids


class NoMatchingAssets(Exception):
    pass


class Position(ModelReply):
    id: str
    role: Role
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class LayerPlan(ModelReply):
    arrangement: str = Field(min_length=1, description="Short spatial plan checked against the object shapes")
    layers: list[Position] = Field(min_length=1)

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, layers: list[Position], info: ValidationInfo) -> list[Position]:
        ids = [layer.id for layer in layers]
        selected = (info.context or {}).get("instance_ids")
        if len(ids) != len(set(ids)) or (selected is not None and set(ids) != selected):
            raise ValueError("Expected exactly one layer for each selected instance")
        roles = (info.context or {}).get("roles")
        if roles is not None and any(layer.role not in roles.get(layer.id, ()) for layer in layers):
            raise ValueError("Expected an available role for each layer")
        return layers


class GeneratedComposition(BaseModel):
    scale: PositiveFloat
    layers: list[Position]


def request_plan[Reply: ModelReply](
    client: ModelClient,
    schema: type[Reply],
    instructions: str,
    payload: dict,
    context: dict,
    previews: dict[str, str] | None = None,
) -> Reply:
    payload_text = json.dumps(payload, ensure_ascii=False)
    user_content = payload_text
    if previews:
        user_content = [{"type": "text", "text": payload_text}]
        for asset_id, image_url in previews.items():
            user_content.extend([
                {"type": "text", "text": f"Превью SVG для объекта {asset_id}"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ])
    content = complete_json(
        client,
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ],
        {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True, "schema": schema.model_json_schema()},
        },
    )
    return schema.model_validate_json(content, context=context)


def variants_for_size(catalog: list[Asset], size: Size) -> dict[str, tuple[Asset, dict[Role, Variant]]]:
    available = {}
    for asset in catalog:
        matching = {variant.role: variant for variant in asset.variants if variant.size == size}
        if matching:
            available[asset.id] = (asset, matching)
    return available


def fit_canvas(layers: list[Position], footprints: dict[str, dict[Role, dict[str, float]]]) -> tuple[float, list[Position]]:
    margin = 0.04
    left, right, top, bottom = 1.0, 0.0, 1.0, 0.0
    for layer in layers:
        size = footprints[layer.id][layer.role]
        half_width, half_height = size["width"] / 2, size["height"] / 2
        left, right = min(left, layer.x - half_width), max(right, layer.x + half_width)
        top, bottom = min(top, layer.y - half_height), max(bottom, layer.y + half_height)
    available = 1 - 2 * margin
    factor = min(1, available / (right - left), available / (bottom - top))
    dx = min(max(0, margin - left * factor), 1 - margin - right * factor)
    dy = min(max(0, margin - top * factor), 1 - margin - bottom * factor)
    return factor, [layer.model_copy(update={"x": layer.x * factor + dx, "y": layer.y * factor + dy}) for layer in layers]


def describe_objects(
    available: dict[str, tuple[Asset, dict[Role, Variant]]], request: CompositionRequest, scale: float,
) -> list[dict]:
    return [
        {
            "id": asset.id,
            "name": asset.name,
            "description": asset.description,
            "variants": {
                role: {"width": variant.width * scale / request.width, "height": variant.height * scale / request.height}
                for role, variant in variants.items()
            },
            "anchors": {name: asdict(point) for name, point in asset.anchors.items()},
        }
        for asset, variants in available.values()
    ]


def object_previews(
    available: dict[str, tuple[Asset, dict[Role, Variant]]],
    asset_paths: dict[tuple[Size, Role, str], Path],
) -> dict[str, str]:
    previews = {}
    for asset, variants in available.values():
        variant = variants.get(Role.MAIN) or variants[Role.BACKGROUND]
        path = asset_paths.get((variant.size, variant.role, asset.id))
        if path is not None:
            try:
                previews[asset.id] = svg_preview(path.read_bytes(), variant.width, variant.height, pixels=320)
            except (OSError, ValueError) as error:
                LOGGER.warning("Skipping composition preview for %s: %s", asset.id, error)
    return previews


def generate_composition(
    request: CompositionRequest,
    available: dict[str, tuple[Asset, dict[Role, Variant]]],
    asset_paths: dict[tuple[Size, Role, str], Path],
) -> GeneratedComposition:
    catalog = [
        {"id": asset.id, "name": asset.name, "description": asset.description}
        for asset, _ in available.values()
    ]
    with llm_client() as client:
        choice = request_plan(client, AssetSelection, SELECT_PROMPT, {
            "prompt": request.prompt,
            "catalog": catalog,
        }, {"asset_ids": set(available)})
        if choice.missing_objects:
            names = ", ".join(f"«{name}»" for name in dict.fromkeys(choice.missing_objects))
            raise NoMatchingAssets(f"В каталоге отсутствуют запрошенные объекты — {names}.")
        if not choice.asset_ids:
            raise NoMatchingAssets("В каталоге нет подходящих объектов для этого запроса.")
        selected = {asset_id: available[asset_id] for asset_id in choice.asset_ids}
        instances = {f"l{index}": asset_id for index, asset_id in enumerate(choice.asset_ids, start=1)}
        all_variants = [variant for _, variants in selected.values() for variant in variants.values()]
        areas = {
            asset_id: max(variant.width * variant.height for variant in variants.values())
            for asset_id, (_, variants) in selected.items()
        }
        area = sum(areas[asset_id] for asset_id in choice.asset_ids)
        scale = min(
            0.72 * request.width / max(variant.width for variant in all_variants),
            0.72 * request.height / max(variant.height for variant in all_variants),
            sqrt(0.55 * request.width * request.height / area),
        )
        objects = describe_objects(selected, request, scale)
        footprints = {item["id"]: item["variants"] for item in objects}
        plan = request_plan(client, LayerPlan, PLACE_PROMPT, {
            "prompt": request.prompt,
            "canvas": {"width": request.width, "height": request.height, "size": request.size},
            "objects": objects,
            "instances": [{"id": key, "asset_id": asset_id} for key, asset_id in instances.items()],
        }, {
            "instance_ids": set(instances),
            "roles": {key: footprints[asset_id] for key, asset_id in instances.items()},
        }, previews=object_previews(selected, asset_paths))

    resolved = [layer.model_copy(update={"id": instances[layer.id]}) for layer in plan.layers]
    factor, layers = fit_canvas(resolved, footprints)
    return GeneratedComposition(scale=scale * factor, layers=layers)
