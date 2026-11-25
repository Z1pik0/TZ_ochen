import { paletteColors, recolor } from './palettes';
import { dimensions, transformFor, variantFor, type Asset, type Scene } from './scene';

const svgNamespace = 'http://www.w3.org/2000/svg';
const tags = new Set([
  'svg', 'g', 'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon',
  'defs', 'clipPath', 'mask', 'linearGradient', 'radialGradient', 'stop', 'pattern', 'use',
]);
const presentation = new Set([
  'fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-linecap',
  'stroke-linejoin', 'stroke-miterlimit', 'stroke-dasharray', 'stroke-dashoffset', 'opacity',
  'clip-path', 'clip-rule', 'mask', 'color', 'paint-order', 'visibility', 'display',
]);
const attributes = new Set([
  ...presentation, 'id', 'd', 'points', 'x', 'y', 'width', 'height', 'viewBox', 'preserveAspectRatio',
  'rx', 'ry', 'r', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'fx', 'fy', 'fr', 'transform',
  'offset', 'stop-color', 'stop-opacity', 'gradientUnits', 'gradientTransform', 'spreadMethod',
  'patternTransform', 'patternUnits', 'patternContentUnits', 'clipPathUnits', 'maskUnits', 'maskContentUnits',
  'href', 'vector-effect',
]);
const paintAttributes = new Set(['fill', 'stroke', 'color', 'stop-color']);

export function parseArtwork(source: string): Element {
  const root = new DOMParser().parseFromString(source, 'image/svg+xml').documentElement;
  if (root.namespaceURI !== svgNamespace || root.localName !== 'svg' || /<!\s*(?:DOCTYPE|ENTITY)/i.test(source)) {
    throw new Error('Некорректный SVG');
  }
  for (const node of [root, ...root.querySelectorAll('*')]) {
    if (node.namespaceURI !== svgNamespace || !tags.has(node.localName)) {
      node.remove();
      continue;
    }
    for (const attribute of [...node.attributes]) {
      if (!attributes.has(attribute.localName)) node.removeAttributeNode(attribute);
    }
  }
  return root;
}

export function inlineArtwork(source: Element, scope: string, colors: ReadonlyMap<string, string>): string {
  const root = source.cloneNode(true) as Element;
  const nodes = [root, ...root.querySelectorAll('*')];
  const ids = new Map<string, string>();
  for (const node of nodes) {
    const id = node.getAttribute('id');
    if (id) ids.set(id, `${scope}-${ids.size}`);
  }
  for (const node of nodes) {
    const isMask = node.closest('mask, clipPath') !== null;
    for (const attribute of [...node.attributes]) {
      if (attribute.localName === 'id') attribute.value = ids.get(attribute.value) ?? '';
      else if (attribute.localName === 'href') {
        const id = attribute.value.startsWith('#') && ids.get(attribute.value.slice(1));
        if (!id) node.removeAttributeNode(attribute);
        else {
          node.removeAttributeNode(attribute);
          node.setAttribute('href', `#${id}`);
        }
      } else if (presentation.has(attribute.localName) && attribute.value.includes('(')) {
        const reference = attribute.value.match(/^url\(\s*['"]?#([^\s'"()]+)['"]?\s*\)$/i);
        const id = reference && ids.get(reference[1]);
        if (id) attribute.value = `url(#${id})`;
        else if (!/^(?:rgba?|hsla?)\([\d.,%\s+-]+\)$/i.test(attribute.value)) node.removeAttributeNode(attribute);
      } else if (!isMask && paintAttributes.has(attribute.localName)) {
        attribute.value = recolor(attribute.value, colors);
      }
    }
  }
  const group = document.createElementNS(svgNamespace, 'g');
  for (const attribute of [...root.attributes]) {
    if (presentation.has(attribute.localName) || attribute.name === 'transform' || attribute.name === 'id') {
      group.setAttribute(attribute.name, attribute.value);
    }
  }
  group.append(...root.children);
  return new XMLSerializer().serializeToString(group);
}

export function buildSvg(scene: Scene, assets: ReadonlyMap<string, Asset>, sources: ReadonlyMap<string, Element>) {
  const { width, height } = dimensions(scene);
  const colors = paletteColors(scene.palette);
  const layers = scene.items.map((item, index) => {
    const variant = variantFor(assets.get(item.assetId), scene.size, item.role);
    const source = variant && sources.get(variant.url);
    if (!variant || !source) throw new Error('SVG слоя не загружен');
    const id = `object-${index + 1}-${item.assetId.replace(/[^a-z0-9_-]/gi, '-')}`;
    return `    <g id="${id}" transform="${transformFor(item, variant, scene)}">${inlineArtwork(source, id, colors)}</g>`;
  });
  return [
    `<svg xmlns="${svgNamespace}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    '  <defs>',
    `    <clipPath id="artboard-clip"><rect width="${width}" height="${height}"/></clipPath>`,
    '  </defs>',
    `  <rect id="background" width="${width}" height="${height}" fill="${scene.palette.background[0]}"/>`,
    '  <g clip-path="url(#artboard-clip)">',
    ...layers,
    '  </g>',
    '</svg>',
    '',
  ].join('\n');
}

export function downloadSvg(svg: string, name: string) {
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
