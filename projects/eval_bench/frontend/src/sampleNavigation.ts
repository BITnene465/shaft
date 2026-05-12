export function sampleIndexFromLocation() {
  if (typeof window === "undefined") {
    return 0;
  }
  const value = new URLSearchParams(window.location.search).get("sample");
  const index = Number(value);
  return Number.isInteger(index) && index >= 0 ? index : 0;
}

export function samplePageOffsetFromLocation(pageSize: number) {
  const index = sampleIndexFromLocation();
  return Math.floor(index / pageSize) * pageSize;
}

export function updateSampleIndexInLocation(index: number) {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("sample", String(index));
  window.history.replaceState(null, "", url);
}
