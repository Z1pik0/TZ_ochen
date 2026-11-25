from asyncio import to_thread
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIError, APIStatusError

from asset_catalog import Anchor, Role, Size, refresh_catalog, scan_assets
from composition import CompositionRequest, GeneratedComposition, NoMatchingAssets, generate_composition, variants_for_size
from llm_gateway import LLMConfigurationError, RETRYABLE_STATUSES


PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = PROJECT_ROOT / "raw_assets"
CATALOG_PATH = PROJECT_ROOT / "catalog.json"
EDITOR_DIST = PROJECT_ROOT / "frontend" / "dist"
ASSET_URL = "/api/assets/{size}/{role}/{asset_id}.svg"
DISPLAY_NAMES = {
    "book": "Книга",
    "phone": "Телефон",
    "puzzle": "Пазл",
    "coin": "Монета",
    "headphones": "Наушники",
    "lamp": "Лампа",
    "vase": "Ваза",
    "wallet": "Кошелёк",
    "folder": "Папка",
    "pen": "Ручка",
}


@dataclass(frozen=True)
class Variant:
    key: str
    size: Size
    role: Role
    width: float
    height: float
    view_box: tuple[float, float, float, float]
    url: str


@dataclass(frozen=True)
class Asset:
    id: str
    name: str
    description: str | None
    anchors: dict[str, Anchor]
    variants: list[Variant]


def load_assets() -> tuple[list[Asset], dict[tuple[Size, Role, str], Path]]:
    sources = scan_assets(ASSET_ROOT)
    entries = refresh_catalog(sources, ASSET_ROOT, CATALOG_PATH)
    catalog = []
    paths = {}
    for asset_id, group in sorted(sources.items()):
        entry = entries[asset_id]
        variants = []
        for variant_key, source in group.items():
            paths[source.size, source.role, asset_id] = source.path
            url = ASSET_URL.format(size=source.size, role=source.role, asset_id=quote(asset_id, safe=""))
            dimensions = entry.variants[variant_key]
            variants.append(Variant(
                variant_key, source.size, source.role, dimensions.width, dimensions.height, source.view_box, url
            ))
        display_name = DISPLAY_NAMES.get(asset_id, asset_id.replace("-", " ").replace("_", " "))
        catalog.append(Asset(asset_id, display_name, entry.description, entry.anchors, variants))
    return catalog, paths


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.catalog, app.state.asset_paths = await to_thread(load_assets)
    yield


app = FastAPI(title="Illustration Builder", lifespan=lifespan)


@app.get("/api/catalog")
def get_catalog(request: Request) -> list[Asset]:
    return request.app.state.catalog


@app.post("/api/compose")
def compose(request: Request, data: CompositionRequest) -> GeneratedComposition:
    available = variants_for_size(request.app.state.catalog, data.size)
    if not available:
        raise HTTPException(status_code=422, detail="В каталоге нет объектов выбранного размера")
    try:
        return generate_composition(data, available, request.app.state.asset_paths)
    except NoMatchingAssets as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (APIError, ValueError) as error:
        raise model_error(error) from error


def model_error(error: APIError | ValueError) -> HTTPException:
    if isinstance(error, LLMConfigurationError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, APIConnectionError):
        return HTTPException(status_code=503, detail="Сервис генерации временно недоступен")
    if isinstance(error, APIStatusError):
        status = error.status_code
        if status in (401, 403):
            return HTTPException(status_code=502, detail="Провайдер модели отклонил ключ. Проверьте LLM_API_KEY.")
        if status in RETRYABLE_STATUSES:
            return HTTPException(status_code=503, detail="Модель временно недоступна. Попробуйте снова.")
        if status == 404:
            return HTTPException(status_code=502, detail="Проверьте LLM_BASE_URL и LLM_MODEL в .env.")
        return HTTPException(status_code=502, detail="Провайдер отклонил запрос. Проверьте совместимость модели и настройки LLM_REASONING_EFFORT.")
    return HTTPException(status_code=502, detail="Модель вернула неполную или некорректную композицию.")


@app.get(
    ASSET_URL,
    response_class=FileResponse,
    responses={200: {"content": {"image/svg+xml": {}}}},
)
def get_asset(request: Request, size: Size, role: Role, asset_id: str) -> FileResponse:
    path = request.app.state.asset_paths.get((size, role, asset_id))
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown asset variant")
    return FileResponse(path, media_type="image/svg+xml")


if (EDITOR_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=EDITOR_DIST, html=True), name="editor")
