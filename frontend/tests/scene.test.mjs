import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { paletteColors, parsePalette, recolor } from '../src/palettes.ts';
import { dimensions, initialScene, reformat } from '../src/scene.ts';

test('format change fits the entire scene while preserving rotations and layer order', () => {
  const layout = {
    ...initialScene,
    scale: 0.5,
    items: [
      { id: '1', assetId: 'book', role: 'bg', x: 0.25, y: 0.4, angle: 30 },
      { id: '2', assetId: 'pen', role: 'main', x: 0.6, y: 0.7, angle: -15 },
    ],
  };

  for (const format of ['wide', 'portrait']) {
    const view = reformat(layout, format);
    const { width, height } = dimensions(view);
    const fit = view.scale / layout.scale;
    assert.deepEqual(view.items.map(({ id, role, angle }) => ({ id, role, angle })),
      layout.items.map(({ id, role, angle }) => ({ id, role, angle })));
    assert.ok(Math.abs((view.items[1].x - view.items[0].x) * width - 350 * fit) < 1e-9);
    assert.ok(Math.abs((view.items[1].y - view.items[0].y) * height - 300 * fit) < 1e-9);
  }

  assert.equal(reformat(layout, 'square'), layout);
  assert.equal(layout.items[0].angle, 30);
});

test('the downloadable palette template is valid and recolors only known slots', () => {
  const template = readFileSync(new URL('../public/palette-template.json', import.meta.url), 'utf8');
  const palette = parsePalette(template);
  const colors = paletteColors(palette);
  assert.equal(recolor('#D7CDF5', colors), palette.background[0]);
  assert.equal(recolor('white', colors), palette.objects[3]);
  assert.equal(recolor('none', colors), 'none');
  assert.throws(() => parsePalette('{"background":[]}'));
});
