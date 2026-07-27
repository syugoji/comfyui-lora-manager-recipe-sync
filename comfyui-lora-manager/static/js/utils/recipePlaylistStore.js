// Recipe playlist ("実行リスト") storage and pure list operations.
// The playlist is an ordered list of recipe references persisted in
// localStorage — deliberately independent from recipe tags so the tag UI
// stays untouched.

import { getStorageItem, setStorageItem } from './storageHelpers.js';

export const PLAYLIST_STORAGE_KEY = 'recipes_playlist';

export function normalizePlaylistEntry(recipe) {
    if (!recipe || typeof recipe !== 'object') return null;
    const id = recipe.id || recipe.recipe_id;
    if (!id) return null;
    const checkpoint = recipe.checkpoint;
    return {
        id: String(id),
        title: String(recipe.title || id),
        checkpointName:
            checkpoint && typeof checkpoint === 'object'
                ? String(checkpoint.name || checkpoint.file_name || '')
                : '',
    };
}

export function addPlaylistEntries(entries, recipes) {
    const next = Array.isArray(entries) ? [...entries] : [];
    const known = new Set(next.map(entry => entry.id));
    let added = 0;
    let skipped = 0;
    for (const recipe of Array.isArray(recipes) ? recipes : []) {
        const entry = normalizePlaylistEntry(recipe);
        if (!entry || known.has(entry.id)) {
            skipped += 1;
            continue;
        }
        known.add(entry.id);
        next.push(entry);
        added += 1;
    }
    return { entries: next, added, skipped };
}

export function removePlaylistEntry(entries, id) {
    return (Array.isArray(entries) ? entries : []).filter(
        entry => entry.id !== id
    );
}

export function movePlaylistEntry(entries, id, offset) {
    const next = Array.isArray(entries) ? [...entries] : [];
    const index = next.findIndex(entry => entry.id === id);
    if (index < 0) return next;
    const target = index + offset;
    if (target < 0 || target >= next.length) return next;
    const [entry] = next.splice(index, 1);
    next.splice(target, 0, entry);
    return next;
}

export function loadPlaylist() {
    const stored = getStorageItem(PLAYLIST_STORAGE_KEY, []);
    if (!Array.isArray(stored)) return [];
    return stored.filter(
        entry => entry && typeof entry === 'object' && entry.id
    );
}

export function savePlaylist(entries) {
    setStorageItem(PLAYLIST_STORAGE_KEY, Array.isArray(entries) ? entries : []);
}
