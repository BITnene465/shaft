const TILE_ZOOM_THRESHOLD = 2.1;
const MAX_RENDERED_TILES = 24;
const IMAGE_PREVIEW_MAX_DISPLAY_SIDE = 1800;

export function PyramidTileLayer({
  width,
  height,
  tileSize,
  level,
  urlTemplate
}: {
  width: number;
  height: number;
  tileSize: number;
  level: number;
  urlTemplate: string;
}) {
  const scale = 2 ** level;
  const levelWidth = Math.ceil(width / scale);
  const levelHeight = Math.ceil(height / scale);
  const columns = Math.ceil(levelWidth / tileSize);
  const rows = Math.ceil(levelHeight / tileSize);
  if (columns * rows > MAX_RENDERED_TILES) {
    return null;
  }
  const tiles = [];
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const originalLeft = x * tileSize * scale;
      const originalTop = y * tileSize * scale;
      const originalRight = Math.min(width, (x + 1) * tileSize * scale);
      const originalBottom = Math.min(height, (y + 1) * tileSize * scale);
      tiles.push({
        key: `${level}-${x}-${y}`,
        src: urlTemplate
          .replace("{level}", String(level))
          .replace("{x}", String(x))
          .replace("{y}", String(y)),
        left: `${(originalLeft / width) * 100}%`,
        top: `${(originalTop / height) * 100}%`,
        width: `${((originalRight - originalLeft) / width) * 100}%`,
        height: `${((originalBottom - originalTop) / height) * 100}%`
      });
    }
  }
  return (
    <div className="pyramid-tile-layer" aria-hidden="true">
      {tiles.map((tile) => (
        <img
          key={tile.key}
          className="pyramid-tile"
          src={tile.src}
          alt=""
          draggable={false}
          loading="lazy"
          decoding="async"
          style={{
            left: tile.left,
            top: tile.top,
            width: tile.width,
            height: tile.height
          }}
        />
      ))}
    </div>
  );
}

export function tileLevelForZoom({
  width,
  height,
  zoom,
  tileSize
}: {
  width: number;
  height: number;
  zoom: number;
  tileSize: number;
}) {
  if (zoom < TILE_ZOOM_THRESHOLD || Math.max(width, height) <= IMAGE_PREVIEW_MAX_DISPLAY_SIDE) {
    return null;
  }
  const preferredLevel = zoom >= 3.5 ? 0 : 1;
  for (let level = preferredLevel; level <= 8; level += 1) {
    const scale = 2 ** level;
    const columns = Math.ceil(Math.ceil(width / scale) / tileSize);
    const rows = Math.ceil(Math.ceil(height / scale) / tileSize);
    if (columns * rows <= MAX_RENDERED_TILES) {
      return level;
    }
  }
  return null;
}
