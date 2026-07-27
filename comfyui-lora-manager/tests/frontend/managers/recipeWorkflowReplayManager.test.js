import { beforeEach, describe, expect, it, vi } from 'vitest';

const stubs = vi.hoisted(() => ({
    fetchRecipeDetails: vi.fn(),
    analyze: vi.fn(),
    collectMissingResources: vi.fn(),
    downloadMissingResources: vi.fn(),
    showToast: vi.fn(),
}));

vi.mock('../../../static/js/api/recipeApi.js', () => ({
    fetchRecipeDetails: stubs.fetchRecipeDetails,
}));
vi.mock('../../../static/js/utils/recipeReplayCapability.js', () => ({
    analyzeRecipeReplayCapability: stubs.analyze,
}));
vi.mock('../../../static/js/managers/BulkMissingLoraDownloadManager.js', () => ({
    bulkMissingLoraDownloadManager: {
        collectMissingResources: stubs.collectMissingResources,
        downloadMissingResources: stubs.downloadMissingResources,
    },
}));
vi.mock('../../../static/js/utils/uiHelpers.js', () => ({ showToast: stubs.showToast }));

import { recipeWorkflowReplayManager } from '../../../static/js/managers/RecipeWorkflowReplayManager.js';

describe('RecipeWorkflowReplayManager', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        const recipe = {
            id: 'b1221d96-eb42-41d1-9107-337703c7f029',
            title: 'Civitai_Recipe_51644312',
        };
        stubs.fetchRecipeDetails.mockResolvedValue(recipe);
        stubs.collectMissingResources.mockReturnValue({ uniqueCount: 0 });
        stubs.analyze.mockResolvedValue({
            level: 'compatible',
            reasons: [],
            built: {
                prompt: { '1': { inputs: {}, class_type: 'SaveImage' } },
                source: 'standard', warnings: [], a1111Parameters: null, a1111Checkpoint: null,
            },
        });
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: vi.fn().mockResolvedValue({ success: true }),
        });
    });

    it('sends the Civitai recipe number as the ComfyUI workflow name', async () => {
        await recipeWorkflowReplayManager.replay('b1221d96-eb42-41d1-9107-337703c7f029');

        expect(stubs.fetchRecipeDetails).toHaveBeenCalledTimes(2);
        expect(stubs.fetchRecipeDetails).toHaveBeenNthCalledWith(
            1,
            'b1221d96-eb42-41d1-9107-337703c7f029',
            { variant: 'active' }
        );
        expect(stubs.fetchRecipeDetails).toHaveBeenNthCalledWith(
            2,
            'b1221d96-eb42-41d1-9107-337703c7f029',
            { variant: 'active' }
        );
        const request = global.fetch.mock.calls[0][1];
        expect(JSON.parse(request.body).workflow_name).toBe('Civitai_Recipe_51644312');
    });

    it('does not auto-download models for an unavailable recipe', async () => {
        stubs.analyze.mockResolvedValue({
            level: 'unavailable',
            reasons: ['元画像にプロンプトがありません'],
            built: null,
        });
        stubs.collectMissingResources.mockReturnValue({ uniqueCount: 2 });

        await expect(recipeWorkflowReplayManager.replay('b1221d96-eb42-41d1-9107-337703c7f029'))
            .rejects.toThrow('元画像にプロンプトがありません');

        expect(stubs.downloadMissingResources).not.toHaveBeenCalled();
        expect(global.fetch).not.toHaveBeenCalled();
    });

    it('labels the a1111 native import toast as a compatible rebuild', async () => {
        stubs.analyze.mockResolvedValue({
            level: 'compatible',
            reasons: [],
            built: {
                prompt: { '1': { inputs: {}, class_type: 'SaveImage' } },
                source: 'a1111', warnings: [], a1111Parameters: 'Steps: 20',
                a1111Checkpoint: 'model.safetensors',
            },
        });

        await recipeWorkflowReplayManager.replay('b1221d96-eb42-41d1-9107-337703c7f029');

        const [key, params] = stubs.showToast.mock.calls[0];
        expect(key).toBe('toast.recipes.workflowReplayed');
        expect(params.source).toBe('元画像のA1111生成データから互換再構築したワークフロー');
    });

    it('does not send a manifest-backed workflow when strict audit is missing or failed', async () => {
        stubs.fetchRecipeDetails.mockResolvedValue({
            id: 'b1221d96-eb42-41d1-9107-337703c7f029',
            title: 'Civitai_Recipe_51644312',
            loras: [{ file_name: 'catalog-only.safetensors', inLibrary: false }],
            replay_manifest: {
                schema: 'lora-manager.replay-manifest', version: 1,
                manifest_hash: 'strict-failed', required_resources: [],
                advisory_resources: [], errors: [],
            },
        });
        stubs.analyze.mockResolvedValue({
            level: 'compatible', reasons: [], audit: { ok: false, failures: [] },
            built: {
                prompt: { '1': { inputs: {}, class_type: 'SaveImage' } },
                source: 'standard', warnings: [], a1111Parameters: null,
                a1111Checkpoint: null, replayManifest: {},
            },
        });

        await expect(recipeWorkflowReplayManager.replay('b1221d96-eb42-41d1-9107-337703c7f029'))
            .rejects.toThrow(/再現監査.*送信していません.*変更していません/);

        expect(global.fetch).not.toHaveBeenCalled();
        expect(stubs.collectMissingResources.mock.calls[0][0][0].loras).toEqual([]);
    });
});
