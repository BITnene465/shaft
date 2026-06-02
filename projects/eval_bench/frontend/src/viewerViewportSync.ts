export type SyncedViewportSnapshot = {
  zoom: number;
  panRatio: { x: number; y: number };
  dirty: boolean;
  sourceId: string;
};

const syncedViewports = new Map<string, SyncedViewportSnapshot>();
const syncedViewportSubscribers = new Map<string, Set<(snapshot: SyncedViewportSnapshot) => void>>();

export function currentSyncedViewport(key: string) {
  return syncedViewports.get(key);
}

export function subscribeSyncedViewport(
  key: string,
  listener: (snapshot: SyncedViewportSnapshot) => void
) {
  const listeners = syncedViewportSubscribers.get(key) ?? new Set();
  listeners.add(listener);
  syncedViewportSubscribers.set(key, listeners);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      syncedViewportSubscribers.delete(key);
    }
  };
}

export function publishSyncedViewport(key: string, snapshot: SyncedViewportSnapshot) {
  syncedViewports.set(key, snapshot);
  syncedViewportSubscribers.get(key)?.forEach((listener) => listener(snapshot));
}
