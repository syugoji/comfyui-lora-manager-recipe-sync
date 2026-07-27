import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../static/js/api/recipeApi.js', () => ({
    fetchRecipeDetails: vi.fn(),
}));
vi.mock('../../../static/js/utils/uiHelpers.js', () => ({ showToast: vi.fn() }));
vi.mock('../../../static/js/utils/recipeReplayCapability.js', () => ({
    analyzeRecipeReplayCapability: vi.fn(),
}));

import { queuePlaylistEntries } from '../../../static/js/managers/RecipePlaylistManager.js';

function makeDeps(overrides = {}) {
    let uuidCount = 0;
    return {
        fetchRecipe: vi.fn(async id => ({ id, title: `Recipe ${id}` })),
        analyze: vi.fn(async () => ({
            level: 'compatible',
            built: {
                prompt: { 1: { class_type: 'SaveImage', inputs: {} } },
                source: 'standard',
                a1111Parameters: null,
                a1111Checkpoint: null,
                replayManifest: null,
            },
        })),
        jsonFetch: vi.fn(async url => {
            if (url === '/api/lm/load-recipe-workflow') {
                return { success: true, prompt: { queued: true } };
            }
            return { prompt_id: `pid-${(uuidCount += 1)}` };
        }),
        uuid: vi.fn(() => `pid-${uuidCount + 1}`),
        ...overrides,
    };
}

describe('queuePlaylistEntries', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('queues every entry in order', async () => {
        const deps = makeDeps();
        const results = await queuePlaylistEntries(
            [
                { id: 'r1', title: 'One' },
                { id: 'r2', title: 'Two' },
            ],
            deps
        );

        expect(results.map(item => item.status)).toEqual(['queued', 'queued']);
        expect(deps.fetchRecipe.mock.calls.map(call => call[0])).toEqual([
            'r1',
            'r2',
        ]);
        // 2 calls per entry: prepare + queue
        expect(deps.jsonFetch).toHaveBeenCalledTimes(4);
        const queueCall = deps.jsonFetch.mock.calls.find(
            call => call[0] === '/prompt'
        );
        const body = JSON.parse(queueCall[1].body);
        expect(body.extra_data.lora_manager_recipe_playlist.recipe_id).toBe('r1');
    });

    it('skips unavailable recipes and continues with the rest', async () => {
        const deps = makeDeps({
            analyze: vi
                .fn()
                .mockResolvedValueOnce({
                    level: 'unavailable',
                    reasons: ['元画像にプロンプトがありません'],
                    built: null,
                })
                .mockResolvedValue({
                    level: 'compatible',
                    built: {
                        prompt: { 1: { class_type: 'SaveImage', inputs: {} } },
                        source: 'standard',
                        a1111Parameters: null,
                        a1111Checkpoint: null,
                        replayManifest: null,
                    },
                }),
        });

        const results = await queuePlaylistEntries(
            [
                { id: 'bad', title: 'Bad' },
                { id: 'good', title: 'Good' },
            ],
            deps
        );

        expect(results[0].status).toBe('skipped');
        expect(results[0].reason).toContain('プロンプトがありません');
        expect(results[1].status).toBe('queued');
    });

    it('skips entries whose queue submission fails without stopping the run', async () => {
        let promptCalls = 0;
        const deps = makeDeps({
            jsonFetch: vi.fn(async url => {
                if (url === '/api/lm/load-recipe-workflow') {
                    return { success: true, prompt: {} };
                }
                promptCalls += 1;
                if (promptCalls === 1) {
                    throw new Error('queue is down');
                }
                return { prompt_id: 'pid-ok' };
            }),
        });

        const results = await queuePlaylistEntries(
            [
                { id: 'r1', title: 'One' },
                { id: 'r2', title: 'Two' },
            ],
            deps
        );

        expect(results[0].status).toBe('skipped');
        expect(results[0].reason).toContain('queue is down');
        expect(results[1].status).toBe('queued');
        expect(results[1].promptId).toBe('pid-ok');
    });

    it('reports progress transitions', async () => {
        const events = [];
        const deps = makeDeps({
            onProgress: (id, status) => events.push(`${id}:${status}`),
        });

        await queuePlaylistEntries([{ id: 'r1', title: 'One' }], deps);

        expect(events).toEqual(['r1:preparing', 'r1:queued']);
    });
});
