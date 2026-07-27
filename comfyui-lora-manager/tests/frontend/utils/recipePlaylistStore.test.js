import { beforeEach, describe, expect, it } from 'vitest';
import {
    addPlaylistEntries,
    loadPlaylist,
    movePlaylistEntry,
    normalizePlaylistEntry,
    removePlaylistEntry,
    savePlaylist,
} from '../../../static/js/utils/recipePlaylistStore.js';

describe('recipePlaylistStore', () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it('normalizes recipes into playlist entries', () => {
        expect(
            normalizePlaylistEntry({
                id: 'r1',
                title: 'Recipe One',
                checkpoint: { name: 'WAI v14' },
            })
        ).toEqual({ id: 'r1', title: 'Recipe One', checkpointName: 'WAI v14' });

        expect(normalizePlaylistEntry({ title: 'no id' })).toBeNull();
        expect(normalizePlaylistEntry(null)).toBeNull();
    });

    it('adds entries with dedupe by recipe id', () => {
        const first = addPlaylistEntries([], [
            { id: 'r1', title: 'One' },
            { id: 'r2', title: 'Two' },
        ]);
        expect(first.added).toBe(2);

        const second = addPlaylistEntries(first.entries, [
            { id: 'r2', title: 'Two' },
            { id: 'r3', title: 'Three' },
            { title: 'invalid' },
        ]);
        expect(second.added).toBe(1);
        expect(second.skipped).toBe(2);
        expect(second.entries.map(entry => entry.id)).toEqual(['r1', 'r2', 'r3']);
    });

    it('moves entries up and down with clamping', () => {
        const entries = [
            { id: 'r1', title: '1', checkpointName: '' },
            { id: 'r2', title: '2', checkpointName: '' },
            { id: 'r3', title: '3', checkpointName: '' },
        ];

        expect(movePlaylistEntry(entries, 'r3', -1).map(e => e.id)).toEqual([
            'r1',
            'r3',
            'r2',
        ]);
        expect(movePlaylistEntry(entries, 'r1', -1).map(e => e.id)).toEqual([
            'r1',
            'r2',
            'r3',
        ]);
        expect(movePlaylistEntry(entries, 'missing', 1).map(e => e.id)).toEqual([
            'r1',
            'r2',
            'r3',
        ]);
    });

    it('removes entries and persists across save/load', () => {
        const entries = [
            { id: 'r1', title: '1', checkpointName: '' },
            { id: 'r2', title: '2', checkpointName: '' },
        ];
        savePlaylist(removePlaylistEntry(entries, 'r1'));

        expect(loadPlaylist()).toEqual([
            { id: 'r2', title: '2', checkpointName: '' },
        ]);
    });
});
