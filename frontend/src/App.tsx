import { useEffect, useId, useMemo, useRef, useState, type ChangeEvent, type FormEvent, type KeyboardEvent, type PointerEvent } from 'react';
import {
  ArrowDown, ArrowUp, Copy, Download, Layers3, Plus, RectangleHorizontal, RectangleVertical, RotateCcw, RotateCw, Square, Trash2, Upload,
} from 'lucide-react';
import { paletteColors, parsePalette, presetPalettes, type Palette } from './palettes';
import {
  arrange, createItem, dimensions, example, formats, initialScene, reformat, transformFor, variantFor,
  type Asset, type Format, type Point, type Role, type Scene, type SceneItem, type Size, type Variant,
} from './scene';
import { buildSvg, downloadSvg, inlineArtwork, parseArtwork } from './svg';

interface Library {
  assets: Asset[];
  sources: Map<string, Element>;
}

interface GeneratedComposition {
  scale: number;
  layers: (Point & { id: string; role: Role })[];
}

const formatIcons = { square: Square, wide: RectangleHorizontal, portrait: RectangleVertical };

async function loadLibrary(signal: AbortSignal): Promise<Library> {
  const response = await fetch('/api/catalog', { signal });
  if (!response.ok) throw new Error('Каталог недоступен');
  const assets: Asset[] = await response.json();
  const files = await Promise.all(assets.flatMap((asset) => asset.variants.map(async ({ url }) => {
    const svg = await fetch(url, { signal });
    if (!svg.ok) throw new Error(`Не удалось загрузить объект «${asset.name}»`);
    return [url, parseArtwork(await svg.text())] as const;
  })));
  return { assets, sources: new Map(files) };
}

function ArtworkThumbnail({ variant, source, colors }: {
  variant: Variant;
  source: Element;
  colors: ReadonlyMap<string, string>;
}) {
  const id = useId().replace(/[^a-z0-9_-]/gi, '-');
  const markup = useMemo(() => inlineArtwork(source, `thumbnail-${id}`, colors), [source, id, colors]);
  return <svg className="artwork-thumb" viewBox={variant.view_box.join(' ')} aria-hidden="true" dangerouslySetInnerHTML={{ __html: markup }} />;
}

function CanvasLayer({ item, scene, variant, source, colors, selected, onPointerDown }: {
  item: SceneItem;
  scene: Scene;
  variant: Variant;
  source: Element;
  colors: ReadonlyMap<string, string>;
  selected: boolean;
  onPointerDown: (event: PointerEvent<SVGGElement>, item: SceneItem) => void;
}) {
  const markup = useMemo(() => inlineArtwork(source, `preview-${item.id}`, colors), [source, item.id, colors]);
  const [x, y, width, height] = variant.view_box;
  return (
    <g transform={transformFor(item, variant, scene)} className="scene-object" onPointerDown={(event) => onPointerDown(event, item)}>
      <g dangerouslySetInnerHTML={{ __html: markup }} />
      {selected && (
        <rect x={x} y={y} width={width} height={height} className="selection-outline" vectorEffect="non-scaling-stroke" />
      )}
    </g>
  );
}

export function App() {
  const [library, setLibrary] = useState<Library | null>(null);
  const [layout, setLayout] = useState(initialScene);
  const [format, setFormat] = useState<Format>(initialScene.format);
  const scene = useMemo(() => reformat(layout, format), [layout, format]);
  const [customPalette, setCustomPalette] = useState<Palette | null>(null);
  const [paletteError, setPaletteError] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [message, setMessage] = useState('');
  const [loadError, setLoadError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [space, setSpace] = useState({ width: 720, height: 560 });
  const stage = useRef<HTMLDivElement>(null);
  const artboard = useRef<SVGSVGElement>(null);
  const paletteInput = useRef<HTMLInputElement>(null);
  const drag = useRef<{ pointerId: number; id: string; offset: Point } | null>(null);
  const assets = useMemo(() => new Map(library?.assets.map((asset) => [asset.id, asset])), [library]);
  const colors = useMemo(() => paletteColors(scene.palette), [scene.palette]);
  const palettes = customPalette ? [...presetPalettes, customPalette] : presetPalettes;
  const { width, height } = dimensions(scene);
  const selectedIndex = scene.items.findIndex((item) => item.id === selectedId);
  const selected = scene.items[selectedIndex];
  const selectedAsset = selected && assets.get(selected.assetId);
  const stageWidth = Math.max(120, Math.min(space.width - 48, (space.height - 54) * width / height, 760));

  useEffect(() => {
    const controller = new AbortController();
    loadLibrary(controller.signal).then((data) => {
      if (controller.signal.aborted) return;
      setLibrary(data);
      const index = new Map(data.assets.map((asset) => [asset.id, asset]));
      const items = example.flatMap((id) => {
        const asset = index.get(id);
        const item = asset && createItem(asset, initialScene.size);
        return item ? [item] : [];
      });
      setLayout(arrange({ ...initialScene, items }, index));
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setLoadError(error instanceof Error ? error.message : 'Не удалось открыть редактор');
      }
    });
    return () => controller.abort();
  }, [attempt]);

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      setSpace({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    if (stage.current) observer.observe(stage.current);
    return () => observer.disconnect();
  }, []);

  function replaceScene(items: SceneItem[]) {
    setLayout(arrange({ ...scene, items }, assets));
    setSelectedId(null);
    setMessage('');
  }

  async function generate(event: FormEvent) {
    event.preventDefault();
    const text = prompt.trim();
    if (!library || generating || !text) return;
    setGenerating(true);
    setMessage('');
    try {
      const response = await fetch('/api/compose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, size: scene.size, width, height }),
      });
      const result = await response.json().catch(() => null);
      if (!response.ok || !result) {
        throw new Error(typeof result?.detail === 'string' ? result.detail : 'Не удалось собрать сцену');
      }
      const { scale, layers } = result as GeneratedComposition;
      const items = layers.map(({ id, role, x, y }) => ({ id: crypto.randomUUID(), assetId: id, role, x, y, angle: 0 }));
      setLayout((current) => ({ ...current, format, scale, items }));
      setSelectedId(null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Не удалось собрать сцену');
    } finally {
      setGenerating(false);
    }
  }

  function loadExample() {
    const items = example.flatMap((id) => {
      const asset = assets.get(id);
      const item = asset && createItem(asset, scene.size);
      return item ? [item] : [];
    });
    replaceScene(items);
    setPrompt('');
  }

  function addAsset(asset: Asset) {
    const item = createItem(asset, scene.size);
    if (!item) return;
    const variant = variantFor(asset, scene.size, item.role)!;
    const offset = 0.035 * ((scene.items.length % 5) - 2);
    item.x += offset;
    item.y += offset;
    const fit = Math.min(width * 0.76 / variant.width, height * 0.76 / variant.height);
    setLayout({ ...scene, scale: scene.items.length ? Math.min(scene.scale, fit) : fit, items: [...scene.items, item] });
    setSelectedId(item.id);
    setMessage('');
  }

  function updateItem(id: string, updates: Partial<SceneItem>) {
    setLayout((current) => {
      const view = reformat(current, format);
      const items = view.items.map((item) => item.id === id ? { ...item, ...updates } : item);
      return { ...view, items };
    });
  }

  function changePalette(palette: Palette) {
    setLayout((current) => ({ ...current, palette }));
    setPaletteError('');
  }

  async function importPalette(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = '';
    if (!file) return;
    try {
      const palette = parsePalette(await file.text());
      setCustomPalette(palette);
      changePalette(palette);
    } catch (error) {
      setPaletteError(error instanceof Error ? error.message : 'Не удалось загрузить палитру.');
    }
  }

  function sizeAvailable(size: Size) {
    return scene.items.every((item) => !!variantFor(assets.get(item.assetId), size, item.role));
  }

  function changeRole(role: Role) {
    if (selected && variantFor(selectedAsset, scene.size, role)) updateItem(selected.id, { role });
  }

  function duplicate() {
    if (!selected) return;
    const item = { ...selected, id: crypto.randomUUID(), x: selected.x + 18 / width, y: selected.y + 18 / height };
    const items = [...scene.items];
    items.splice(selectedIndex + 1, 0, item);
    setLayout({ ...scene, items });
    setSelectedId(item.id);
  }

  function remove() {
    if (!selected) return;
    setLayout({ ...scene, items: scene.items.filter((item) => item.id !== selected.id) });
    setSelectedId(null);
  }

  function moveLayer(step: number) {
    const nextIndex = selectedIndex + step;
    if (!selected || nextIndex < 0 || nextIndex >= scene.items.length) return;
    const items = [...scene.items];
    [items[selectedIndex], items[nextIndex]] = [items[nextIndex], items[selectedIndex]];
    setLayout({ ...scene, items });
  }

  function pointFor(event: PointerEvent): Point | undefined {
    const svg = artboard.current;
    const matrix = svg?.getScreenCTM();
    if (!matrix) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return { x: point.x / width, y: point.y / height };
  }

  function startDrag(event: PointerEvent<SVGGElement>, item: SceneItem) {
    if (generating || event.button !== 0) return;
    const point = pointFor(event);
    if (!point) return;
    event.stopPropagation();
    artboard.current?.focus({ preventScroll: true });
    artboard.current?.setPointerCapture(event.pointerId);
    drag.current = { pointerId: event.pointerId, id: item.id, offset: { x: item.x - point.x, y: item.y - point.y } };
    setSelectedId(item.id);
  }

  function moveDrag(event: PointerEvent<SVGSVGElement>) {
    const active = drag.current;
    if (generating || !active || event.pointerId !== active.pointerId) return;
    const point = pointFor(event);
    if (point) updateItem(active.id, { x: point.x + active.offset.x, y: point.y + active.offset.y });
  }

  function endDrag() { drag.current = null; }

  function handleKeyboard(event: KeyboardEvent<HTMLDivElement>) {
    if (!selected || generating) return;
    if (event.target instanceof Element && event.target.closest('input, textarea, select, [contenteditable="true"]')) return;
    if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault();
      remove();
      return;
    }
    const offsets: Record<string, Point> = {
      ArrowLeft: { x: -1, y: 0 }, ArrowRight: { x: 1, y: 0 },
      ArrowUp: { x: 0, y: -1 }, ArrowDown: { x: 0, y: 1 },
    };
    const offset = offsets[event.key];
    if (offset) {
      event.preventDefault();
      const step = event.shiftKey ? 10 : 1;
      updateItem(selected.id, { x: selected.x + offset.x * step / width, y: selected.y + offset.y * step / height });
    }
  }

  function exportScene() {
    if (!library || !scene.items.length) return;
    downloadSvg(buildSvg(scene, assets, library.sources), `composition-${scene.format}-${scene.size}.svg`);
  }

  return (
    <div className="editor" onKeyDown={handleKeyboard}>
      <header className="app-header">
        <div className="brand"><span className="brand-icon"><Layers3 size={21} /></span><span>Сборка</span></div>
        <div className="header-actions">
          <button className="button secondary" onClick={loadExample} disabled={!library || generating}><RotateCcw size={16} /> Пример</button>
          <button className="button primary" onClick={exportScene} disabled={!library || generating || !scene.items.length}><Download size={17} /> Скачать SVG</button>
        </div>
      </header>

      <main className="workspace">
        <aside className="library panel" aria-label="Библиотека объектов">
          <div className="panel-head"><h2>Библиотека</h2>{library && <span className="count">{library.assets.length}</span>}</div>
          <form className="name-form" onSubmit={generate} aria-busy={generating}>
            <label htmlFor="composition-prompt">Идея или список объектов</label>
            <textarea id="composition-prompt" rows={3} value={prompt}
              onChange={(event) => { setPrompt(event.target.value); setMessage(''); }}
              placeholder="Защита данных на смартфоне" disabled={!library || generating} />
            <button className="button assemble" disabled={!library || generating || !prompt.trim()} type="submit">
              {generating ? 'Собираем сцену…' : 'Собрать сцену'}
            </button>
            {message && <p className="feedback" role="alert">{message}</p>}
          </form>
          <div className="asset-grid">
            {library?.assets.map((asset) => {
              const preview = variantFor(asset, scene.size, 'main') ?? variantFor(asset, scene.size, 'bg');
              const source = preview && library.sources.get(preview.url);
              return (
                <button key={asset.id} type="button" className="asset-card" onClick={() => addAsset(asset)} disabled={!preview || generating}
                  aria-label={`Добавить ${asset.name}`} title={asset.description ?? asset.name}>
                  {preview && source && <ArtworkThumbnail variant={preview} source={source} colors={colors} />}
                  <span>{asset.name}</span><Plus size={13} className="asset-plus" aria-hidden="true" />
                </button>
              );
            })}
          </div>
        </aside>

        <section className="canvas-panel" aria-label="Рабочая область">
          <div className="canvas-toolbar">
            <div className="canvas-heading"><h1>Композиция</h1></div>
            <div className="canvas-controls">
              <div className="segmented" role="group" aria-label="Формат холста">
                {formats.map(({ id, label }) => {
                  const Icon = formatIcons[id];
                  return <button key={id} aria-pressed={scene.format === id} aria-label={label} title={label} onClick={() => setFormat(id)} disabled={generating}><Icon size={17} /></button>;
                })}
              </div>
              <div className="segmented size-switch" role="group" aria-label="Размер иллюстрации">
                {(['L', 'S'] as const).map((size) => (
                  <button key={size} aria-pressed={scene.size === size} onClick={() => setLayout((current) => ({ ...current, size }))} disabled={generating || !sizeAvailable(size)}>{size}</button>
                ))}
              </div>
            </div>
          </div>
          <div className="palette-toolbar">
            <label className="palette-picker">
              Палитра
              <select value={palettes.indexOf(scene.palette)} disabled={!library}
                onChange={(event) => changePalette(palettes[Number(event.currentTarget.value)])}>
                {palettes.map((palette, index) => <option key={index} value={index}>{palette.name}</option>)}
              </select>
            </label>
            <div className="palette-swatches">
              <div className="swatch-group" role="group" aria-label="Цвета фона">
                {scene.palette.background.map((color, index) => <span key={index} title={color} style={{ backgroundColor: color }} />)}
              </div>
              <div className="swatch-group" role="group" aria-label="Цвета объектов">
                {scene.palette.objects.map((color, index) => <span key={index} title={color} style={{ backgroundColor: color }} />)}
              </div>
            </div>
            <input ref={paletteInput} type="file" accept=".json,application/json" hidden onChange={importPalette} />
            <button className="button secondary palette-upload" type="button" disabled={!library} onClick={() => paletteInput.current?.click()}>
              <Upload size={14} /> Загрузить JSON
            </button>
            <a className="palette-template" href="/palette-template.json" download>Шаблон JSON</a>
            {paletteError && <p className="feedback" role="alert">{paletteError}</p>}
          </div>
          <div className="stage" ref={stage}>
            {loadError ? <div className="stage-message" role="alert"><strong>{loadError}</strong><button className="button secondary" onClick={() => { setLoadError(''); setAttempt(attempt + 1); }}>Повторить</button></div>
              : !library ? <div className="stage-message" role="status">Загружаем SVG</div>
                : <svg ref={artboard} className="artboard" viewBox={`0 0 ${width} ${height}`} width={stageWidth} height={stageWidth * height / width}
                  style={{ background: scene.palette.background[0] }} aria-label="Холст композиции" tabIndex={0} onPointerDown={() => setSelectedId(null)}
                  onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={endDrag} onLostPointerCapture={endDrag}>
                  {scene.items.map((item) => {
                    const variant = variantFor(assets.get(item.assetId), scene.size, item.role);
                    const source = variant && library.sources.get(variant.url);
                    return variant && source && <CanvasLayer key={item.id} item={item} scene={scene} variant={variant} source={source} colors={colors}
                      selected={item.id === selectedId} onPointerDown={startDrag} />;
                  })}
                </svg>}
            {library && !scene.items.length && <p className="empty-canvas">Добавьте объект из библиотеки</p>}
          </div>
          <div className="canvas-footer">
            <span>{formats.find((format) => format.id === scene.format)?.label}</span>
            <span>{width} × {height} px</span>
          </div>
        </section>

        <aside className="inspector panel" aria-label="Слои и свойства">
          <div className="panel-head"><h2>Слои</h2><span className="count">{scene.items.length}</span></div>
          <div className="layer-list">
            {[...scene.items].reverse().map((item) => {
              const asset = assets.get(item.assetId);
              const variant = variantFor(asset, scene.size, item.role);
              const source = variant && library?.sources.get(variant.url);
              return (
                <button key={item.id} className="layer" aria-pressed={item.id === selectedId} onClick={() => setSelectedId(item.id)}>
                  <span className="layer-thumb">{variant && source && <ArtworkThumbnail variant={variant} source={source} colors={colors} />}</span>
                  <span className="layer-name">{asset?.name ?? item.assetId}</span>
                  <span className="layer-role">{item.role === 'main' ? 'Главный' : 'Фоновый'}</span>
                </button>
              );
            })}
          </div>
          <div className="properties">
            <div className="panel-head"><h2>Свойства</h2></div>
            {selected ? (
              <fieldset className="property-controls" disabled={generating}>
                <div className="position-fields">
                  <label>X<input aria-label="Координата X" type="number" value={Math.round(selected.x * width)} onChange={(event) => {
                    const value = event.target.valueAsNumber;
                    if (Number.isFinite(value)) updateItem(selected.id, { x: value / width });
                  }} /></label>
                  <label>Y<input aria-label="Координата Y" type="number" value={Math.round(selected.y * height)} onChange={(event) => {
                    const value = event.target.valueAsNumber;
                    if (Number.isFinite(value)) updateItem(selected.id, { y: value / height });
                  }} /></label>
                </div>
                <div className="angle-field">
                  <label htmlFor="layer-angle">Поворот</label>
                  <div className="angle-controls">
                    <button className="icon-button" type="button" aria-label="Повернуть на 15° против часовой" title="−15°"
                      onClick={() => updateItem(selected.id, { angle: selected.angle - 15 })}><RotateCcw size={14} /></button>
                    <input id="layer-angle" type="number" value={selected.angle} onChange={(event) => {
                      const value = event.target.valueAsNumber;
                      if (Number.isFinite(value)) updateItem(selected.id, { angle: value });
                    }} />
                    <button className="icon-button" type="button" aria-label="Повернуть на 15° по часовой" title="+15°"
                      onClick={() => updateItem(selected.id, { angle: selected.angle + 15 })}><RotateCw size={14} /></button>
                  </div>
                </div>
                <div className="segmented roles" role="group" aria-label="Роль объекта">
                  <button aria-pressed={selected.role === 'main'} disabled={!variantFor(selectedAsset, scene.size, 'main')} onClick={() => changeRole('main')}>Главный</button>
                  <button aria-pressed={selected.role === 'bg'} disabled={!variantFor(selectedAsset, scene.size, 'bg')} onClick={() => changeRole('bg')}>Фоновый</button>
                </div>
                <div className="layer-tools">
                  <button className="icon-button" title="Поднять слой" aria-label="Поднять слой" onClick={() => moveLayer(1)} disabled={selectedIndex === scene.items.length - 1}><ArrowUp size={17} /></button>
                  <button className="icon-button" title="Опустить слой" aria-label="Опустить слой" onClick={() => moveLayer(-1)} disabled={selectedIndex === 0}><ArrowDown size={17} /></button>
                  <button className="icon-button" title="Дублировать" aria-label="Дублировать" onClick={duplicate}><Copy size={16} /></button>
                  <button className="icon-button remove" title="Удалить" aria-label="Удалить" onClick={remove}><Trash2 size={17} /></button>
                </div>
              </fieldset>
            ) : <p className="panel-hint">Выберите объект на холсте или в слоях</p>}
          </div>
        </aside>
      </main>
    </div>
  );
}
