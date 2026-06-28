// Shared class -> color map. Kept in sync with backend/classes.py CLASS_COLORS.
// Values are [r, g, b] 0-255. The backend also sends this via /api/config, but
// we keep a local copy as a fallback and for the legend.
export const CLASS_COLORS = {
  person: [239, 68, 68],
  bicycle: [245, 158, 11],
  car: [59, 130, 246],
  motorcycle: [168, 85, 247],
  bus: [16, 185, 129],
  truck: [14, 165, 233],
  bird: [250, 204, 21],
  cat: [244, 114, 182],
  dog: [251, 146, 60],
  horse: [132, 204, 22],
  sheep: [148, 163, 184],
  cow: [217, 119, 6],
  elephant: [100, 116, 139],
  bear: [120, 53, 15],
  zebra: [226, 232, 240],
  giraffe: [202, 138, 4],
  "trash bin": [74, 222, 128],
  scooter: [34, 211, 238],
  skates: [232, 121, 249],
  drone: [250, 250, 250],
};

export const DEFAULT_COLOR = [148, 163, 184];

export function colorFor(name) {
  return CLASS_COLORS[name] || DEFAULT_COLOR;
}

export function cssColor(name, alpha = 1) {
  const [r, g, b] = colorFor(name);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function hexColor(name) {
  const [r, g, b] = colorFor(name);
  return (r << 16) | (g << 8) | b;
}
