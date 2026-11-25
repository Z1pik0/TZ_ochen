export type Size = 'L' | 'S';
export type Role = 'main' | 'bg';
export type Format = 'square' | 'wide' | 'portrait';

export interface Point {
  x: number;
  y: number;
}

export interface Variant {
  size: Size;
  role: Role;
  width: number;
  height: number;
  view_box: [number, number, number, number];
  url: string;
}

export interface Asset {
  id: string;
  name: string;
  description: string | null;
  anchors: Record<string, Point>;
  variants: Variant[];
}

export interface SceneItem extends Point {
  id: string;
  assetId: string;
  role: Role;
  angle: number;
}

export interface Scene {
  size: Size;
  format: Format;
  palette: Palette;
  scale: number;
  items: SceneItem[];
}

export const formats = [
  { id: 'square', label: 'Квадрат', width: 1000, height: 1000 },
  { id: 'wide', label: 'Горизонтальный', width: 1200, height: 640 },
  { id: 'portrait', label: 'Вертикальный', width: 800, height: 1200 },
] as const;

export const initialScene: Scene = { size: 'L', format: 'square', palette: presetPalettes[0], items: [], scale: 1 };
export const example = ['book', 'phone', 'puzzle'];

export function dimensions(scene: Scene) {
  const format = formats.find((item) => item.id === scene.format)!;
  const scale = scene.size === 'S' ? 0.5 : 1;
  return { width: format.width * scale, height: format.height * scale };
}

export function variantFor(asset: Asset | undefined, size: Size, role: Role) {
  return asset?.variants.find((variant) => variant.size === size && variant.role === role);
}

export function createItem(asset: Asset, size: Size): SceneItem | undefined {
  const variant = variantFor(asset, size, 'main') ?? variantFor(asset, size, 'bg');
  if (variant) return { id: crypto.randomUUID(), assetId: asset.id, role: variant.role, x: 0.5, y: 0.5, angle: 0 };
}

export function arrange(scene: Scene, assets: ReadonlyMap<string, Asset>): Scene {
  if (!scene.items.length) return scene;
  const { width, height } = dimensions(scene);
  const book = scene.items.find((item) => item.assetId === 'book');
  const phone = scene.items.find((item) => item.assetId === 'phone');
  const booksAndPhone = book && phone && (scene.items.length === 2 ||
    (scene.items.length === 3 && scene.items.some((item) => item.assetId === 'puzzle')));
  const bookVariant = book && variantFor(assets.get(book.assetId), scene.size, book.role);
  const page = assets.get('book')?.anchors.on_right_page;

  if (booksAndPhone && bookVariant && page) {
    const formatPositions = {
      square: { book: { x: 0.48, y: 0.59 }, puzzle: { x: 0.72, y: 0.28 }, fit: [0.67, 0.47] },
      wide: { book: { x: 0.40, y: 0.56 }, puzzle: { x: 0.76, y: 0.35 }, fit: [0.60, 0.72] },
      portrait: { book: { x: 0.47, y: 0.60 }, puzzle: { x: 0.66, y: 0.30 }, fit: [0.76, 0.52] },
    };
    const positions = formatPositions[scene.format];
    const scale = Math.min(width * positions.fit[0] / bookVariant.width, height * positions.fit[1] / bookVariant.height);
    const phonePosition = {
      x: positions.book.x + (page.x - 0.5) * bookVariant.width * scale / width,
      y: positions.book.y + (page.y - 0.5) * bookVariant.height * scale / height,
    };
    const items = scene.items.map((item) => ({
      ...item,
      ...(item.id === book.id ? positions.book : item.id === phone.id ? phonePosition : positions.puzzle),
      angle: 0,
    }));
    const bookIndex = items.findIndex((item) => item.id === book.id);
    const phoneIndex = items.findIndex((item) => item.id === phone.id);
    if (bookIndex > phoneIndex) items.splice(phoneIndex, 0, items.splice(bookIndex, 1)[0]);
    return { ...scene, items, scale };
  }

  const columns = Math.min(scene.items.length, Math.max(1, Math.round(Math.sqrt(scene.items.length * width / height))));
  const rows = Math.ceil(scene.items.length / columns);
  const cellWidth = 0.86 * width / columns;
  const cellHeight = 0.82 * height / rows;
  const scale = Math.min(...scene.items.map((item) => {
    const variant = variantFor(assets.get(item.assetId), scene.size, item.role)!;
    return Math.min(cellWidth * 0.86 / variant.width, cellHeight * 0.86 / variant.height);
  }));
  const items = scene.items.map((item, index) => {
    const row = Math.floor(index / columns);
    const countInRow = Math.min(columns, scene.items.length - row * columns);
    return {
      ...item,
      x: 0.5 + (index % columns + 0.5 - countInRow / 2) * cellWidth / width,
      y: 0.09 + (row + 0.5) * cellHeight / height,
      angle: 0,
    };
  });
  return { ...scene, items, scale };
}

export function reformat(scene: Scene, format: Format): Scene {
  if (scene.format === format) return scene;
  const from = dimensions(scene);
  const to = dimensions({ ...scene, format });
  const fit = Math.min(to.width / from.width, to.height / from.height);
  return {
    ...scene,
    format,
    scale: scene.scale * fit,
    items: scene.items.map((item) => ({
      ...item,
      x: 0.5 + (item.x - 0.5) * from.width * fit / to.width,
      y: 0.5 + (item.y - 0.5) * from.height * fit / to.height,
    })),
  };
}

export function transformFor(item: SceneItem, variant: Variant, scene: Scene) {
  const { width, height } = dimensions(scene);
  const [minX, minY, artWidth, artHeight] = variant.view_box;
  return `translate(${item.x * width} ${item.y * height}) rotate(${item.angle}) scale(${scene.scale}) translate(${-minX - artWidth / 2} ${-minY - artHeight / 2})`;
}
import { presetPalettes, type Palette } from './palettes.ts';
