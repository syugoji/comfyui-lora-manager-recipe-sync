import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
    RecipeTrialManager,
    createTrialSeeds,
} from '../../../static/js/managers/RecipeTrialManager.js';

function jsonResponse(payload, status = 200) {
    return {
        ok: status >= 200 && status < 300,
        status,
        json: vi.fn().mockResolvedValue(payload),
    };
}

function recipe(seed = 42) {
    return {
        id: 'recipe-trial',
        title: 'Civitai_Recipe_123',
        replay_manifest: {
            schema: 'lora-manager.replay-manifest',
            version: 1,
            manifest_hash: 'manifest-current',
            required_resources: [],
            advisory_resources: [],
            errors: [],
        },
        source_etag: 'e'.repeat(64),
        gen_params: { prompt: 'old', negative_prompt: 'bad hands', seed },
    };
}

function draft() {
    return {
        draft_hash: 'draft-current',
        manifest_hash: 'manifest-current',
        proposed_prompt: '<lora:style:0.9>, protected, generated scene',
        negative_prompt: 'bad hands',
    };
}

function successfulAnalysis(trialRecipe) {
    return {
        level: 'compatible',
        reasons: [],
        audit: { ok: true, failures: [], required_model_inputs: [] },
        built: {
            prompt: {
                '1': {
                    class_type: 'KSampler',
                    inputs: { seed: trialRecipe.gen_params.seed },
                },
            },
            source: 'standard',
            a1111Parameters: null,
            a1111Checkpoint: null,
            replayManifest: trialRecipe.replay_manifest,
        },
    };
}

describe('RecipeTrialManager', () => {
    beforeEach(() => {
        localStorage.clear();
        vi.clearAllMocks();
    });

    it('uses original plus three unique random seeds, or four random seeds for -1', () => {
        const randomValues = [7, 8, 9, 10];
        let index = 0;
        expect(createTrialSeeds(42, () => randomValues[index++])).toEqual([
            { seed: 42, origin: 'original' },
            { seed: 7, origin: 'random' },
            { seed: 8, origin: 'random' },
            { seed: 9, origin: 'random' },
        ]);
        index = 0;
        expect(createTrialSeeds(-1, () => randomValues[index++])).toEqual([
            { seed: 7, origin: 'random' },
            { seed: 8, origin: 'random' },
            { seed: 9, origin: 'random' },
            { seed: 10, origin: 'random' },
        ]);
    });

    it('submits exactly four candidates sequentially and persists UUID before each POST', async () => {
        const events = [];
        const persistedBeforePost = [];
        const uuidValues = [
            '00000000-0000-4000-8000-000000000000',
            '00000000-0000-4000-8000-000000000001',
            '00000000-0000-4000-8000-000000000002',
            '00000000-0000-4000-8000-000000000003',
            '00000000-0000-4000-8000-000000000004',
        ];
        let uuidIndex = 0;
        const fetchImpl = vi.fn(async (url, options = {}) => {
            if (url === '/queue') return jsonResponse({ queue_running: [], queue_pending: [] });
            if (url === '/api/lm/load-recipe-workflow') {
                const body = JSON.parse(options.body);
                return jsonResponse({ success: true, prompt: body.prompt });
            }
            if (url === '/prompt') {
                const body = JSON.parse(options.body);
                events.push(`prompt:${body.prompt_id}`);
                const stored = JSON.parse(localStorage.getItem('lm_recipe_trial_v1:recipe-trial'));
                const candidate = stored.candidates.find(item => item.prompt_id === body.prompt_id);
                persistedBeforePost.push(Boolean(candidate?.attempted_at));
                return jsonResponse({ prompt_id: body.prompt_id, number: 1, node_errors: {} });
            }
            if (url.startsWith('/history/')) {
                const promptId = decodeURIComponent(url.split('/').pop());
                events.push(`history:${promptId}`);
                return jsonResponse({
                    [promptId]: {
                        status: { completed: true, status_str: 'success' },
                        outputs: {
                            '9': {
                                images: [{
                                    filename: `${promptId}.png`,
                                    subfolder: 'recipe trials',
                                    type: 'output',
                                }],
                            },
                        },
                    },
                });
            }
            throw new Error(`unexpected URL: ${url}`);
        });
        const randomValues = [101, 102, 103];
        let randomIndex = 0;
        const analyze = vi.fn(async value => successfulAnalysis(value));
        const manager = new RecipeTrialManager({
            fetchImpl,
            analyze,
            storage: localStorage,
            now: () => 1000,
            randomSeed: () => randomValues[randomIndex++],
            uuid: () => uuidValues[uuidIndex++],
            sleep: vi.fn().mockResolvedValue(undefined),
        });

        const job = await manager.start({ recipe: recipe(42), draft: draft() });

        expect(job.status).toBe('completed');
        expect(job.draft_snapshot).toEqual(draft());
        expect(job.candidates.map(item => item.seed)).toEqual([42, 101, 102, 103]);
        expect(job.candidates.every(item => item.status === 'succeeded')).toBe(true);
        expect(job.candidates[0].images[0]).toMatchObject({
            output_node_id: '9', image_index: 0,
        });
        expect(job.candidates[0].images[0].url).toContain('subfolder=recipe+trials');
        expect(persistedBeforePost).toEqual([true, true, true, true]);
        expect(events).toEqual([
            `prompt:${uuidValues[1]}`, `history:${uuidValues[1]}`,
            `prompt:${uuidValues[2]}`, `history:${uuidValues[2]}`,
            `prompt:${uuidValues[3]}`, `history:${uuidValues[3]}`,
            `prompt:${uuidValues[4]}`, `history:${uuidValues[4]}`,
        ]);
        expect(analyze.mock.calls.map(call => call[0].gen_params.seed)).toEqual([42, 101, 102, 103]);
    });

    it('does not submit when the ComfyUI queue is busy', async () => {
        const fetchImpl = vi.fn(async url => {
            if (url === '/queue') {
                return jsonResponse({ queue_running: [[1, 'foreign-job']], queue_pending: [] });
            }
            throw new Error(`unexpected URL: ${url}`);
        });
        const manager = new RecipeTrialManager({
            fetchImpl,
            analyze: vi.fn(async value => successfulAnalysis(value)),
            storage: localStorage,
            now: () => 1000,
            randomSeed: () => 5,
            uuid: () => '00000000-0000-4000-8000-000000000000',
        });

        await expect(manager.start({ recipe: recipe(), draft: draft() }))
            .rejects.toThrow('キューが空ではありません');

        expect(fetchImpl.mock.calls.some(call => call[0] === '/prompt')).toBe(false);
        const stored = manager.readStoredJob('recipe-trial');
        expect(stored.status).toBe('failed');
        expect(stored.candidates.every(item => item.status === 'not_submitted')).toBe(true);
    });

    it('does not create an unadoptable job when the source ETag is missing', async () => {
        const fetchImpl = vi.fn();
        const manager = new RecipeTrialManager({ fetchImpl, storage: localStorage });
        const withoutEtag = recipe();
        delete withoutEtag.source_etag;

        await expect(manager.start({ recipe: withoutEtag, draft: draft() }))
            .rejects.toThrow('保存用ETag');

        expect(fetchImpl).not.toHaveBeenCalled();
        expect(manager.readStoredJob('recipe-trial')).toBeNull();
    });

    it('stops before prompt submission when strict replay audit fails', async () => {
        const fetchImpl = vi.fn(async url => {
            if (url === '/queue') return jsonResponse({ queue_running: [], queue_pending: [] });
            throw new Error(`unexpected URL: ${url}`);
        });
        const manager = new RecipeTrialManager({
            fetchImpl,
            analyze: vi.fn().mockResolvedValue({
                level: 'unavailable',
                reasons: ['必須LoRAがありません'],
                audit: { ok: false, failures: [{ message: '必須LoRAがありません' }] },
                built: null,
            }),
            storage: localStorage,
            now: () => 1000,
            randomSeed: () => 5,
            uuid: () => '00000000-0000-4000-8000-000000000000',
        });

        const job = await manager.start({ recipe: recipe(), draft: draft() });

        expect(job.status).toBe('failed');
        expect(job.candidates[0].error).toContain('必須LoRA');
        expect(fetchImpl.mock.calls.some(call => call[0] === '/prompt')).toBe(false);
    });

    it('recovers known prompt history without resubmitting pending candidates', async () => {
        const stored = {
            schema: 'lora-manager.recipe-trial', version: 1,
            job_id: 'job', recipe_id: 'recipe-trial', recipe_title: 'Recipe',
            draft_hash: 'draft-current', manifest_hash: 'manifest-current',
            prompt_source: 'lm_studio', created_at: 1000, expires_at: 90000,
            status: 'running', active_index: 0, error: null,
            candidates: [
                { index: 0, seed: 42, seed_origin: 'original', status: 'running', prompt_id: 'known-id', attempted_at: 1000, images: [], error: null },
                ...[1, 2, 3].map(index => ({ index, seed: 50 + index, seed_origin: 'random', status: 'pending', prompt_id: null, attempted_at: null, images: [], error: null })),
            ],
        };
        localStorage.setItem('lm_recipe_trial_v1:recipe-trial', JSON.stringify(stored));
        const fetchImpl = vi.fn(async url => {
            if (url === '/history/known-id') {
                return jsonResponse({
                    'known-id': {
                        status: { completed: true, status_str: 'success' },
                        outputs: { '9': { images: [{ filename: 'done.png', type: 'output' }] } },
                    },
                });
            }
            if (url === '/queue') return jsonResponse({ queue_running: [], queue_pending: [] });
            throw new Error(`unexpected URL: ${url}`);
        });
        const manager = new RecipeTrialManager({
            fetchImpl, storage: localStorage, now: () => 2000,
        });

        const job = await manager.recover('recipe-trial');

        expect(job.status).toBe('partial');
        expect(job.candidates[0].status).toBe('succeeded');
        expect(job.candidates.slice(1).every(item => item.status === 'not_submitted')).toBe(true);
        expect(fetchImpl.mock.calls.some(call => call[0] === '/prompt')).toBe(false);
    });
});
