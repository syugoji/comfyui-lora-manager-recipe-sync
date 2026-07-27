import { describe, expect, it } from 'vitest';

import { buildCompleteRecipeMetadata } from '../../../static/js/utils/recipeMetadata.js';

describe('complete recipe metadata', () => {
    it('preserves every replay representation and resource type', () => {
        const metadata = buildCompleteRecipeMetadata({
            base_model: 'NoobAI',
            loras: [{ file_name: 'style.safetensors' }],
            embeddings: [{ file_name: 'negative.pt' }],
            gen_params: { scheduler: 'Exponential', hires_upscale: 1.5 },
            generation_metadata: { Version: 'f1.3.0', changed: [null] },
            a1111_parameters: 'prompt\nNegative prompt: \nSteps: 20',
            comfy_prompt: { 1: { class_type: 'KSampler' } },
            comfy_workflow: { nodes: [{ id: 1 }] },
            generation_source: 'embedded_a1111',
            generation_source_policy: 'embedded-first-v1',
            checkpoint: { file_name: 'model.safetensors' },
            preview_nsfw_level: 2,
        }, 'https://civitai.red/images/115941302');

        expect(metadata).toMatchObject({
            embeddings: [{ file_name: 'negative.pt' }],
            generation_metadata: { Version: 'f1.3.0', changed: [null] },
            generation_source: 'embedded_a1111',
            generation_source_policy: 'embedded-first-v1',
            source_path: 'https://civitai.red/images/115941302',
            preview_nsfw_level: 2,
        });
        expect(metadata.a1111_parameters).toContain('Steps: 20');
        expect(metadata.comfy_prompt).toBeTruthy();
        expect(metadata.comfy_workflow).toBeTruthy();
    });
});
