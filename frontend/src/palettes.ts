type Colors = readonly [string, string, string, string];

export interface Palette {
  name: string;
  background: Colors;
  objects: Colors;
}

// The two columns of the palette swatches in the Figma SVG archives, top to bottom.
export const presetPalettes: readonly [Palette, Palette, Palette] = [
  {
    name: 'Сиреневая',
    background: ['#D7CDF5', '#AA9BF0', '#7864D7', '#181817'],
    objects: ['#F9ECB0', '#F9A3AA', '#FFE3E1', '#FFFFFF'],
  },
  {
    name: 'Персиковая',
    background: ['#FDD7B9', '#FAC896', '#D27532', '#181817'],
    objects: ['#F9ECB0', '#80DADA', '#BEEBEB', '#FFFFFF'],
  },
  {
    name: 'Бирюзовая',
    background: ['#BEEBEB', '#80DADA', '#5ABEBE', '#181817'],
    objects: ['#FFD4F2', '#6D7BFC', '#B6C6FE', '#FFFFFF'],
  },
];

function normalizeHex(value: string): string | undefined {
  const color = value.trim().toUpperCase();
  if (/^#[0-9A-F]{6}$/.test(color)) return color;
  if (/^#[0-9A-F]{3}$/.test(color)) return '#' + [...color.slice(1)].map((digit) => digit.repeat(2)).join('');
}

function readColors(value: unknown, field: string): Colors {
  const colors = Array.isArray(value) && value.length === 4
    ? value.map((color: unknown) => typeof color === 'string' ? normalizeHex(color) : undefined)
    : [];
  const [first, second, third, fourth] = colors;
  if (!first || !second || !third || !fourth) {
    throw new Error(`В поле ${field} должно быть четыре цвета HEX, например #D7CDF5.`);
  }
  return [first, second, third, fourth];
}

export function parsePalette(text: string): Palette {
  let input: unknown;
  try {
    input = JSON.parse(text.replace(/^\uFEFF/, ''));
  } catch {
    throw new Error('Не удалось прочитать JSON палитры.');
  }
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error('Палитра должна быть объектом JSON с полями background и objects.');
  }
  const { name, background, objects } = input as Record<string, unknown>;
  return {
    name: typeof name === 'string' && name.trim() ? name.trim() : 'Своя палитра',
    background: readColors(background, 'background'),
    objects: readColors(objects, 'objects'),
  };
}

export function paletteColors(palette: Palette): ReadonlyMap<string, string> {
  const reference = presetPalettes[0];
  const replacements = new Map<string, string>();
  for (const group of ['background', 'objects'] as const) {
    reference[group].forEach((color, index) => replacements.set(color, palette[group][index]));
  }
  return replacements;
}

export function recolor(value: string, colors: ReadonlyMap<string, string>): string {
  const key = value.trim().toLowerCase() === 'white' ? '#FFFFFF' : normalizeHex(value);
  const replacement = key && colors.get(key);
  return replacement && replacement !== key ? replacement : value;
}
