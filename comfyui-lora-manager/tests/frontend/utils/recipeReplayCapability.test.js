import { describe, expect, it } from 'vitest';

import {
    analyzeRecipeReplayCapability,
    getRecipePromptStatus,
} from '../../../static/js/utils/recipeReplayCapability.js';

function standardObjectInfo({ checkpoints = ['model.safetensors'], loras = [], upscalers = [] } = {}) {
    return {
        CheckpointLoaderSimple: {
            input: { required: { ckpt_name: [checkpoints] } }, output_node: false,
        },
        CLIPTextEncode: { input: { required: { text: ['STRING'], clip: ['CLIP'] } }, output_node: false },
        EmptyLatentImage: {
            input: { required: { width: ['INT'], height: ['INT'], batch_size: ['INT'] } }, output_node: false,
        },
        KSampler: {
            input: { required: {
                seed: ['INT'], steps: ['INT'], cfg: ['FLOAT'], sampler_name: ['STRING'],
                scheduler: ['STRING'], denoise: ['FLOAT'], model: ['MODEL'], positive: ['CONDITIONING'],
                negative: ['CONDITIONING'], latent_image: ['LATENT'],
            } },
            output_node: false,
        },
        VAEDecode: { input: { required: { samples: ['LATENT'], vae: ['VAE'] } }, output_node: false },
        SaveImage: {
            input: { required: { filename_prefix: ['STRING'], images: ['IMAGE'] } }, output_node: true,
        },
        LoraLoader: {
            input: { required: {
                model: ['MODEL'], clip: ['CLIP'], lora_name: [loras],
                strength_model: ['FLOAT'], strength_clip: ['FLOAT'],
            } },
            output_node: false,
        },
        UpscaleModelLoader: {
            input: { required: { model_name: [upscalers] } }, output_node: false,
        },
        ImageUpscaleWithModel: {
            input: { required: { upscale_model: ['UPSCALE_MODEL'], image: ['IMAGE'] } }, output_node: false,
        },
        ImageScale: {
            input: { required: {
                image: ['IMAGE'], upscale_method: ['STRING'], width: ['INT'], height: ['INT'], crop: ['STRING'],
            } },
            output_node: false,
        },
        VAEEncode: {
            input: { required: { pixels: ['IMAGE'], vae: ['VAE'] } }, output_node: false,
        },
    };
}

describe('recipe replay capability', () => {
    it('treats an active LM Studio revision as generated even when the source prompt is missing', () => {
        expect(getRecipePromptStatus({
            gen_params: {},
            revision_summary: { active: true, prompt_source: 'LM_STUDIO' },
        })).toBe('generated');
        expect(getRecipePromptStatus({
            gen_params: {},
            revision_summary: { active: true, prompt_source: 'source' },
        })).toBe('missing');
    });

    it('marks a complete embedded graph as exact', async () => {
        const objectInfo = {
            LoadImage: { input: { required: { image: ['STRING'] } }, output_node: false },
            SaveImage: { input: { required: { images: ['IMAGE'] } }, output_node: true },
        };
        const capability = await analyzeRecipeReplayCapability({
            comfy_prompt: {
                '1': { inputs: { image: 'source.png' }, class_type: 'LoadImage' },
                '2': {
                    inputs: { filename_prefix: 'embedded-test', images: ['1', 0] },
                    class_type: 'SaveImage',
                },
            },
            gen_params: {},
        }, { objectInfo });

        expect(capability.level).toBe('exact');
    });

    it('marks a reconstructed workflow with an unavailable checkpoint as unavailable', async () => {
        const capability = await analyzeRecipeReplayCapability({
            checkpoint: { file_name: 'missing.safetensors' },
            gen_params: { prompt: 'portrait' },
        }, { objectInfo: standardObjectInfo() });

        expect(capability.level).toBe('unavailable');
        expect(capability.reasons.join(' ')).toMatch(/missing\.safetensors/);
    });

    it('marks a missing LoRA as unavailable because the source image cannot be fully reproduced', async () => {
        const capability = await analyzeRecipeReplayCapability({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{ file_name: 'missing-lora.safetensors' }],
            gen_params: { prompt: 'portrait' },
        }, { objectInfo: standardObjectInfo() });

        expect(capability.level).toBe('unavailable');
        expect(capability.reasons.join(' ')).toMatch(/missing-lora/);
    });

    it('marks a structurally invalid cached LoRA as unavailable even if ComfyUI lists its name', async () => {
        const capability = await analyzeRecipeReplayCapability({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{ file_name: 'html-not-a-model.safetensors', inLibrary: false }],
            gen_params: { prompt: 'portrait' },
        }, { objectInfo: standardObjectInfo({ loras: ['html-not-a-model.safetensors'] }) });

        expect(capability.level).toBe('unavailable');
        expect(capability.reasons.join(' ')).toMatch(/未導入または破損.*html-not-a-model/);
    });

    it('marks a missing hires upscaler as unavailable rather than compatible', async () => {
        const capability = await analyzeRecipeReplayCapability({
            checkpoint: { file_name: 'model.safetensors' },
            a1111_parameters: 'portrait\nSteps: 20, Size: 768x1024, Hires upscale: 2, Hires upscaler: 4xNomos8kDAT',
            gen_params: {
                prompt: 'portrait', size: '768x1024', hires_upscale: 2,
                hires_upscaler: '4xNomos8kDAT',
            },
        }, { objectInfo: standardObjectInfo({ upscalers: ['remacri_original.pth'] }) });

        expect(capability.level).toBe('unavailable');
        expect(capability.reasons.join(' ')).toMatch(/4xNomos8kDAT/);
    });

    it('rejects a different installed checkpoint when embedded metadata names another hash', async () => {
        const capability = await analyzeRecipeReplayCapability({
            checkpoint: {
                file_name: 'model.safetensors',
                hash: '490d8efa4dd220eda4d12d16b1da95ffa8efb9bbf34366d47c1855f2cbc9369a',
            },
            a1111_parameters: 'portrait\nSteps: 20, Model hash: 3a0dacebb3, Model: model',
            gen_params: { prompt: 'portrait' },
        }, { objectInfo: standardObjectInfo() });

        expect(capability.level).toBe('unavailable');
        expect(capability.reasons.join(' ')).toMatch(/チェックポイントSHA.*3a0dacebb3.*490d8efa4dd2/);
    });

    it('marks missing generation metadata as unavailable', async () => {
        const capability = await analyzeRecipeReplayCapability({
            generation_source: 'reconstructed',
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: {},
        }, { objectInfo: standardObjectInfo() });

        expect(capability.level).toBe('unavailable');
        expect(capability.reasons.join(' ')).toMatch(/プロンプト／生成パラメータ/);
    });

    it('strictly audits an evidence-backed LoRA path before replay', async () => {
        const recipe = {
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{
                file_name: 'required-style.safetensors',
                inLibrary: true,
                localPath: 'D:/models/loras/required-style.safetensors',
            }],
            gen_params: { prompt: 'portrait' },
            replay_manifest: {
                schema: 'lora-manager.replay-manifest', version: 1,
                manifest_hash: 'strict-one', source_kind: 'standard', errors: [],
                advisory_resources: [],
                required_resources: [{
                    requirement_id: 'recipe:0', kind: 'lora', required: true,
                    resource: {
                        file_name: 'required-style.safetensors', inLibrary: true,
                        localPath: 'D:/models/loras/required-style.safetensors',
                    },
                    resolution: { status: 'recipe_match', match: 'exact_name' },
                    expected: { strength_model: 0.75, strength_clip: 0.5 },
                    evidence: [{ source: 'inline_lora_tag' }],
                }],
            },
        };

        const capability = await analyzeRecipeReplayCapability(recipe, {
            objectInfo: standardObjectInfo({ loras: ['required-style.safetensors'] }),
        });

        expect(capability.level).toBe('compatible');
        expect(capability.audit).toMatchObject({ ok: true, mode: 'strict' });
        expect(capability.audit.required_model_inputs).toHaveLength(1);
    });

    it('does not fail strict replay for a missing advisory catalog LoRA', async () => {
        const recipe = {
            checkpoint: { file_name: 'model.safetensors' },
            loras: [
                { file_name: 'required-style.safetensors', inLibrary: true },
                { file_name: 'catalog-only.safetensors', inLibrary: false },
            ],
            gen_params: { prompt: 'portrait' },
            replay_manifest: {
                schema: 'lora-manager.replay-manifest', version: 1,
                manifest_hash: 'strict-advisory', source_kind: 'standard', errors: [],
                required_resources: [{
                    requirement_id: 'recipe:0', kind: 'lora', required: true,
                    resource: { file_name: 'required-style.safetensors', inLibrary: true },
                    resolution: { status: 'recipe_match', match: 'exact_name' },
                    expected: { strength_model: 1, strength_clip: 1 },
                    evidence: [{ source: 'inline_lora_tag' }],
                }],
                advisory_resources: [{
                    kind: 'lora', required: false, reason: 'recipe_catalog_only',
                    resource: { file_name: 'catalog-only.safetensors', inLibrary: false },
                }],
            },
        };

        const capability = await analyzeRecipeReplayCapability(recipe, {
            objectInfo: standardObjectInfo({ loras: ['required-style.safetensors'] }),
        });

        expect(capability.level).toBe('compatible');
        expect(capability.audit.ok).toBe(true);
        expect(capability.reasons.join(' ')).not.toMatch(/catalog-only/);
    });

    it('does not inject a missing required LoRA into an embedded graph', async () => {
        const objectInfo = {
            ...standardObjectInfo({ loras: ['required-style.safetensors'] }),
            LoadImage: { input: { required: { image: ['STRING'] } }, output_node: false },
        };
        const capability = await analyzeRecipeReplayCapability({
            comfy_prompt: {
                '1': { inputs: { image: 'source.png' }, class_type: 'LoadImage' },
                '2': {
                    inputs: { filename_prefix: 'strict-embedded', images: ['1', 0] },
                    class_type: 'SaveImage',
                },
            },
            loras: [{ file_name: 'required-style.safetensors', inLibrary: true }],
            gen_params: {},
            replay_manifest: {
                schema: 'lora-manager.replay-manifest', version: 1,
                manifest_hash: 'strict-embedded', source_kind: 'embedded', errors: [],
                advisory_resources: [],
                required_resources: [{
                    requirement_id: 'recipe:0', kind: 'lora', required: true,
                    resource: { file_name: 'required-style.safetensors', inLibrary: true },
                    resolution: { status: 'recipe_match', match: 'exact_name' },
                    expected: { strength_model: 1, strength_clip: 1 },
                    evidence: [{ source: 'a1111_civitai_resources' }],
                }],
            },
        }, { objectInfo });

        expect(capability.level).toBe('unavailable');
        expect(capability.audit, capability.reasons.join(' / ')).not.toBeNull();
        expect(capability.audit.ok).toBe(false);
        expect(capability.audit.failures.map(failure => failure.code)).toContain('REQUIRED_LORA_MISSING');
        expect(Object.values(capability.built.prompt).filter(node => node.class_type === 'LoraLoader')).toHaveLength(0);
    });
});
