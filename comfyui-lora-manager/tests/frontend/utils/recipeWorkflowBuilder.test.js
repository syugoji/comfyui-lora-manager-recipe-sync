import { describe, expect, it } from 'vitest';

import {
    buildRecipeWorkflow,
    getResourceFilename,
} from '../../../static/js/utils/recipeWorkflowBuilder.js';

describe('recipeWorkflowBuilder', () => {
    it('builds a standard workflow and inserts every recipe LoRA', () => {
        const recipe = {
            id: 'recipe-1',
            title: 'Portrait',
            checkpoint: {
                file_name: 'dream.safetensors',
            },
            loras: [
                { name: 'Detail', file_name: 'detail', weight: 0.7 },
                { name: 'Style', file_name: 'style.safetensors', weight: 1.1 },
            ],
            gen_params: {
                prompt: 'portrait prompt',
                negative_prompt: 'bad hands',
                steps: 24,
                cfg_scale: 5.5,
                seed: 42,
                sampler: 'Euler a Karras',
                size: '768x1024',
            },
        };

        const result = buildRecipeWorkflow(recipe);
        const loraNodes = Object.entries(result.prompt)
            .filter(([, node]) => node.class_type === 'LoraLoader');

        expect(result.source).toBe('standard');
        expect(loraNodes).toHaveLength(2);
        expect(loraNodes[0][1].inputs.lora_name).toBe('detail.safetensors');
        expect(loraNodes[1][1].inputs.model).toEqual([loraNodes[0][0], 0]);
        expect(result.prompt['2'].inputs.clip).toEqual([loraNodes[1][0], 1]);
        expect(result.prompt['5'].inputs.model).toEqual([loraNodes[1][0], 0]);
        expect(result.prompt['5'].inputs.sampler_name).toBe('euler_ancestral');
        expect(result.prompt['5'].inputs.scheduler).toBe('karras');
        expect(result.prompt['4'].inputs).toMatchObject({ width: 768, height: 1024 });
    });

    it('forces full denoise when a reconstructed txt2img workflow starts from an empty latent', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'portrait', denoising_strength: 0.3 },
        });

        expect(result.prompt['5'].inputs.denoise).toBe(1);
    });

    it('keeps recipe denoise for a real img2img latent', () => {
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': {
                    inputs: { denoise: 1, latent_image: ['2', 0] },
                    class_type: 'KSampler',
                },
                '2': {
                    inputs: { pixels: ['3', 0], vae: ['4', 2] },
                    class_type: 'VAEEncode',
                },
            },
            gen_params: { denoising_strength: 0.3 },
        });

        expect(result.prompt['1'].inputs.denoise).toBe(0.3);
    });

    it('uses a checkpoint Comfy prompt as a compatibility template', () => {
        const template = {
            prompt: {
                '3': {
                    inputs: {
                        seed: 1,
                        steps: 9,
                        cfg: 1,
                        sampler_name: 'euler',
                        scheduler: 'simple',
                        model: ['16', 0],
                        positive: ['6', 0],
                        negative: ['7', 0],
                        latent_image: ['13', 0],
                    },
                    class_type: 'KSampler',
                    _meta: { title: 'KSampler' },
                },
                '6': {
                    inputs: { text: 'old', clip: ['18', 0] },
                    class_type: 'CLIPTextEncode',
                    _meta: { title: 'CLIP Text Encode (Positive Prompt)' },
                },
                '7': {
                    inputs: { text: 'old negative', clip: ['18', 0] },
                    class_type: 'CLIPTextEncode',
                    _meta: { title: 'CLIP Text Encode (Negative Prompt)' },
                },
                '13': {
                    inputs: { width: 832, height: 1216, batch_size: 1 },
                    class_type: 'EmptySD3LatentImage',
                },
                '16': {
                    inputs: { unet_name: 'template.safetensors', weight_dtype: 'default' },
                    class_type: 'UNETLoader',
                },
                '18': {
                    inputs: { clip_name: 'qwen.safetensors', type: 'lumina2' },
                    class_type: 'CLIPLoader',
                },
            },
        };
        const recipe = {
            checkpoint: {
                civitai: {
                    files: [{ name: 'zImageTurbo_turbo.safetensors', type: 'Diffusion Model', primary: true }],
                    images: [{ meta: { comfy: JSON.stringify(template) } }],
                },
            },
            loras: [{ name: 'Z Detail', file_name: 'z_detail.safetensors' }],
            gen_params: { prompt: 'new prompt', steps: 11, seed: 99, cfg_scale: 1 },
        };

        const result = buildRecipeWorkflow(recipe);
        const loraEntry = Object.entries(result.prompt).find(([, node]) => node.class_type === 'LoraLoader');

        expect(result.source).toBe('checkpoint-template');
        expect(result.prompt['16'].inputs.unet_name).toBe('zImageTurbo_turbo.safetensors');
        expect(result.prompt['6'].inputs.text).toBe('new prompt');
        expect(result.prompt['3'].inputs).toMatchObject({ seed: 99, steps: 11, cfg: 1 });
        expect(result.prompt['3'].inputs.model).toEqual([loraEntry[0], 0]);
        expect(result.prompt['6'].inputs.clip).toEqual([loraEntry[0], 1]);
    });

    it('prefers an embedded Comfy API prompt over a generic template', () => {
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': { inputs: { ckpt_name: 'old.safetensors' }, class_type: 'CheckpointLoaderSimple' },
            },
            checkpoint: { file_name: 'new.safetensors' },
            gen_params: {},
        });

        expect(result.source).toBe('embedded');
        expect(result.prompt['1'].inputs.ckpt_name).toBe('new.safetensors');
        expect(result.warnings).toEqual([]);
    });

    it('prefers preserved A1111 parameters over generic workflow templates', () => {
        const parameters = 'portrait\nNegative prompt: bad\nSteps: 24, Sampler: Euler a, CFG scale: 7, Seed: 42, Size: 768x1024';
        const result = buildRecipeWorkflow({
            a1111_parameters: parameters,
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'fallback' },
        });

        expect(result.source).toBe('a1111');
        expect(result.a1111Parameters).toBe(parameters);
        expect(result.a1111Checkpoint).toBe('model.safetensors');
        expect(result.warnings).toEqual([]);
    });

    it('rebuilds unsafe A1111 metadata with an external VAE and pixel-space model hires pass', () => {
        const parameters = [
            'portrait',
            'Negative prompt: bad',
            'Steps: 24, Sampler: DPM++ 2M, Size: 768x1024,',
            'VAE: sdxl_vae_fixed.safetensors, Hires upscale: 1.5, Hires upscaler: 4x-UltraSharp',
        ].join('\n');
        const result = buildRecipeWorkflow({
            a1111_parameters: parameters,
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: {
                prompt: 'portrait', size: '768x1024', vae: 'sdxl_vae_fixed.safetensors',
                hires_upscale: 1.5, hires_steps: 10, denoising_strength: 0.35,
            },
        });

        expect(result.source).toBe('standard');
        expect(result.a1111Parameters).toBeNull();
        expect(result.prompt['8']).toMatchObject({
            class_type: 'VAELoader',
            inputs: { vae_name: 'sdxl_vae_fixed.safetensors' },
        });
        const upscaler = Object.values(result.prompt).find(node => node.class_type === 'UpscaleModelLoader');
        const imageUpscale = Object.values(result.prompt).find(node => node.class_type === 'ImageUpscaleWithModel');
        const vaeEncode = Object.values(result.prompt).find(node => node.class_type === 'VAEEncode');
        const hiresSampler = Object.values(result.prompt).find(
            node => node._meta?.title === 'KSampler (Hires pass)'
        );
        expect(upscaler.inputs.model_name).toBe('4x-UltraSharp');
        expect(imageUpscale).toBeDefined();
        expect(vaeEncode.inputs.vae).toEqual(['8', 0]);
        expect(hiresSampler.inputs).toMatchObject({ steps: 10, denoise: 0.35 });
        expect(result.prompt['6'].inputs.samples).toEqual(['5', 0]);
        expect(result.prompt['7'].inputs.images).not.toEqual(['6', 0]);
    });

    it('keeps latent hires for A1111 latent/None upscalers', () => {
        const result = buildRecipeWorkflow({
            a1111_parameters: 'portrait\nSteps: 20, Size: 768x1024, Hires upscale: 1.5, Hires upscaler: None',
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'portrait', size: '768x1024', hires_upscale: 1.5 },
        });

        expect(Object.values(result.prompt).some(node => node.class_type === 'LatentUpscaleBy')).toBe(true);
        expect(Object.values(result.prompt).some(node => node.class_type === 'ImageUpscaleWithModel')).toBe(false);
    });

    it('preserves an A1111 scheduler stored separately from the sampler name', () => {
        const result = buildRecipeWorkflow({
            a1111_parameters: 'portrait\nSteps: 20, Sampler: DPM++ 2M, Schedule type: Karras, Size: 768x1024',
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'portrait', sampler: 'DPM++ 2M', scheduler: 'Karras' },
        });

        expect(result.prompt['5'].inputs).toMatchObject({
            sampler_name: 'dpmpp_2m', scheduler: 'karras',
        });
    });

    it('removes zero-width mojibake from saved prompt text', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'portrait', negative_prompt: 'anime, âbad quality' },
        });

        expect(result.prompt['3'].inputs.text).toBe('anime, bad quality');
    });

    it('restores ordered LoRA weights from A1111 extension metadata', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [
                { file_name: 'one.safetensors', strength: 1 },
                { file_name: 'two.safetensors', strength: 1 },
            ],
            a1111_parameters: 'portrait\nSteps: 20, Size: 832x1216, loraweights: "0.4,0.7"',
            gen_params: { prompt: 'portrait', size: '832x1216' },
        });

        const loaders = Object.values(result.prompt).filter(node => node.class_type === 'LoraLoader');
        expect(loaders.map(node => node.inputs.strength_model)).toEqual([0.4, 0.7]);
        expect(loaders.map(node => node.inputs.strength_clip)).toEqual([0.4, 0.7]);
        expect(result.warnings.join(' ')).toMatch(/LoRA weights/);
    });

    it('uses tiled VAE decode for multi-megapixel reconstructed outputs', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'portrait', size: '1792x2432' },
        });

        expect(result.prompt['6']).toMatchObject({
            class_type: 'VAEDecodeTiled',
            inputs: { tile_size: 512, overlap: 64 },
        });
    });

    it('treats extension-heavy A1111 metadata as a compatibility reconstruction', () => {
        const parameters = [
            '<segment:face> portrait',
            'Steps: 20, Sampler: Euler, Size: 768x1024, Segment enabled: True, Refiner: model',
        ].join('\n');
        const result = buildRecipeWorkflow({
            a1111_parameters: parameters,
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: '<segment:face> portrait', size: '768x1024' },
        });

        expect(result.source).toBe('standard');
        expect(result.warnings[0]).toMatch(/A1111固有情報/);
    });

    it('builds a native Flux loader graph for diffusion-model resources', () => {
        const result = buildRecipeWorkflow({
            generation_source: 'reconstructed',
            base_model: 'Flux.1 D',
            checkpoint: {
                baseModel: 'Flux.1 D',
                inLibrary: true,
                localPath: 'D:/ComfyUI/models/diffusion_models/Flux.1 D/flux_dev.safetensors',
            },
            gen_params: { prompt: 'a portrait', size: '1024x1024', cfg_scale: 3.5 },
        });

        expect(result.prompt['1']).toMatchObject({
            class_type: 'UNETLoader',
            inputs: { unet_name: 'Flux.1 D/flux_dev.safetensors' },
        });
        expect(result.prompt['2']).toMatchObject({ class_type: 'DualCLIPLoader' });
        expect(result.prompt['3']).toMatchObject({
            class_type: 'VAELoader', inputs: { vae_name: 'ae.safetensors' },
        });
        expect(result.prompt['8'].inputs.cfg).toBe(1);
    });

    it('prefers the declared SDXL family over a mixed SDXL and Flux checkpoint title', () => {
        const result = buildRecipeWorkflow({
            generation_source: 'reconstructed',
            checkpoint: {
                name: 'The Araminta Experiment (SDXL+Flux)',
                baseModel: 'SDXL 1.0',
                inLibrary: true,
                localPath: 'D:/models/Stable-diffusion/SDXL 1.0/theAramintaExperiment_cv5.safetensors',
            },
            gen_params: { prompt: 'two opalescent spheres', size: '1024x1024' },
        });

        expect(result.prompt['1']).toMatchObject({
            class_type: 'CheckpointLoaderSimple',
            inputs: { ckpt_name: 'SDXL 1.0/theAramintaExperiment_cv5.safetensors' },
        });
        expect(Object.values(result.prompt).some(node => node.class_type === 'DualCLIPLoader')).toBe(false);
        expect(Object.values(result.prompt).some(node => node.class_type === 'FluxGuidance')).toBe(false);
    });

    it('replaces ambiguous Flux ae VAE with the checkpoint VAE for SDXL-family embedded workflows', () => {
        const result = buildRecipeWorkflow({
            base_model: 'Illustrious',
            comfy_prompt: {
                '1': { inputs: { ckpt_name: 'old.safetensors' }, class_type: 'CheckpointLoaderSimple' },
                '2': { inputs: { vae_name: 'ae.safetensors' }, class_type: 'VAELoader' },
                '3': { inputs: { samples: ['4', 0], vae: ['2', 0] }, class_type: 'VAEDecode' },
                '4': { inputs: {}, class_type: 'KSampler' },
                '5': { inputs: { images: ['3', 0] }, class_type: 'SaveImage' },
            },
            checkpoint: { file_name: 'illustrious.safetensors' },
            gen_params: { prompt: 'portrait' },
        });

        expect(result.prompt['2']).toBeUndefined();
        expect(result.prompt['3'].inputs.vae).toEqual(['1', 2]);
        expect(result.warnings.join(' ')).toMatch(/16ch.*ae\.safetensors/);
    });

    it('replaces ambiguous Flux ae VAE after SDXL-family A1111 reconstruction', () => {
        const result = buildRecipeWorkflow({
            base_model: 'Illustrious',
            a1111_parameters: [
                'portrait',
                'Steps: 24, Sampler: Euler a, Size: 832x1216, VAE: ae',
            ].join('\n'),
            checkpoint: { file_name: 'illustrious.safetensors' },
            gen_params: {
                prompt: 'portrait', size: '832x1216', vae: 'ae',
                steps: 24, cfg_scale: 6,
            },
        });

        const vaeLoader = Object.values(result.prompt).find(
            node => node.class_type === 'VAELoader'
        );
        expect(result.source).toBe('standard');
        expect(vaeLoader).toBeUndefined();
        expect(result.prompt['6'].inputs.vae).toEqual(['1', 2]);
        expect(result.warnings.join(' ')).toMatch(/16ch.*ae\.safetensors/);
    });

    it('refuses a reconstructed recipe that has no generation metadata', () => {
        expect(() => buildRecipeWorkflow({
            generation_source: 'reconstructed',
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: {},
        })).toThrow(/プロンプト／生成パラメータ/);
    });

    it('inlines legacy primitive constants used by embedded workflows', () => {
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': { inputs: { width: ['2', 0], height: ['3', 0], batch_size: 1 }, class_type: 'EmptyLatentImage' },
                '2': { inputs: { Number: '1024' }, class_type: 'Int' },
                '3': { inputs: { Number: '1536' }, class_type: 'Int' },
            },
            gen_params: {},
        });

        expect(result.prompt['1'].inputs).toMatchObject({ width: 1024, height: 1536 });
        expect(result.prompt['2']).toBeUndefined();
        expect(result.prompt['3']).toBeUndefined();
    });

    it('repairs a non-output image saver with a core SaveImage node', () => {
        const objectInfo = {
            KSampler: { input: { required: {} }, output_node: false },
            SDPromptSaver: { input: { required: { images: ['IMAGE'] } }, output_node: false },
            SaveImage: { input: { required: { images: ['IMAGE'] } }, output_node: true },
        };
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': { inputs: {}, class_type: 'KSampler' },
                '2': { inputs: { images: ['1', 0] }, class_type: 'SDPromptSaver' },
            },
            gen_params: {},
        }, { objectInfo });

        expect(result.source).toBe('embedded');
        expect(Object.values(result.prompt).some(node => node.class_type === 'SaveImage')).toBe(true);
        expect(result.warnings.join(' ')).toMatch(/SaveImage/);
    });

    it('rebuilds a reachable missing custom node when generation metadata is complete', () => {
        const objectInfo = {
            SaveImage: { input: { required: { images: ['IMAGE'] } }, output_node: true },
        };
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': { inputs: {}, class_type: 'MissingCustomSampler' },
                '2': { inputs: { images: ['1', 0] }, class_type: 'SaveImage' },
            },
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { prompt: 'portrait' },
        }, { objectInfo });

        expect(result.source).toBe('standard');
        expect(result.warnings.join(' ')).toMatch(/不足ノード/);
    });

    it('reduces a single selected batch item without changing its seed sequence', () => {
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': { inputs: { width: 512, height: 512, batch_size: 4 }, class_type: 'EmptyLatentImage' },
                '2': { inputs: { samples: ['1', 0], batch_index: 1, length: 1 }, class_type: 'LatentFromBatch' },
                '3': { inputs: { seed: 100, latent_image: ['2', 0] }, class_type: 'KSampler' },
            },
            gen_params: {},
        });

        expect(result.prompt['1'].inputs.batch_size).toBe(1);
        expect(result.prompt['2']).toBeUndefined();
        expect(result.prompt['3'].inputs).toMatchObject({ seed: 101, latent_image: ['1', 0] });
    });

    it('reuses an existing embedded LoRA loader and deduplicates recipe resources', () => {
        const result = buildRecipeWorkflow({
            comfy_prompt: {
                '1': { inputs: { ckpt_name: 'model.safetensors' }, class_type: 'Checkpoint Loader (Simple)' },
                '2': {
                    inputs: {
                        model: ['1', 0], clip: ['1', 1], lora_name: 'detail',
                        strength_model: 1, strength_clip: 1,
                    },
                    class_type: 'Load LoRA (Model and CLIP)',
                },
            },
            checkpoint: { file_name: 'model.safetensors' },
            loras: [
                { file_name: 'folder/detail.safetensors', strength: 0.6 },
                { file_name: 'detail.safetensors', strength: 0.6 },
            ],
            gen_params: {},
        });
        const loraNodes = Object.values(result.prompt).filter(
            node => String(node.class_type).toLowerCase().includes('lora')
        );

        expect(loraNodes).toHaveLength(1);
        expect(loraNodes[0].inputs).toMatchObject({
            lora_name: 'detail.safetensors', strength_model: 0.6, strength_clip: 0.6,
        });
    });

    it('converts a negative external seed into a ComfyUI-valid seed', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            gen_params: { seed: -1 },
        });

        expect(result.prompt['5'].inputs.seed).toBe(0);
    });

    it('converts inline LoRA tags into loader strengths and removes them from the prompt', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{ file_name: 'sampleStyleLora_v15.safetensors', strength: 1 }],
            gen_params: { prompt: '<lora:sampleStyleLora_v15:0.5> portrait' },
        });
        const loraNode = Object.values(result.prompt).find(node => node.class_type === 'LoraLoader');

        expect(result.prompt['2'].inputs.text).toBe('portrait');
        expect(loraNode.inputs).toMatchObject({
            lora_name: 'sampleStyleLora_v15.safetensors',
            strength_model: 0.5,
            strength_clip: 0.5,
        });
    });

    it('uses the recipe strength field when no inline LoRA tag is present', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{ file_name: 'detail.safetensors', strength: 0.35 }],
            gen_params: { prompt: 'portrait' },
        });
        const loraNode = Object.values(result.prompt).find(node => node.class_type === 'LoraLoader');

        expect(loraNode.inputs).toMatchObject({
            lora_name: 'detail.safetensors',
            strength_model: 0.35,
            strength_clip: 0.35,
        });
    });

    it('preserves CLIP Skip in a reconstructed graph without creating a LoRA cycle', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{ file_name: 'detail.safetensors' }],
            gen_params: { prompt: 'portrait', clip_skip: 2 },
        });
        const clipLayer = Object.entries(result.prompt)
            .find(([, node]) => node.class_type === 'CLIPSetLastLayer');
        const lora = Object.entries(result.prompt)
            .find(([, node]) => node.class_type === 'LoraLoader');

        expect(clipLayer[1].inputs).toEqual({ clip: ['1', 1], stop_at_clip_layer: -2 });
        expect(lora[1].inputs.clip).toEqual([clipLayer[0], 0]);
        expect(result.prompt['2'].inputs.clip).toEqual([lora[0], 1]);
    });

    it('preserves separate AddNet model and clip strengths', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{
                file_name: 'detail.safetensors',
                weight: 0.7,
                strength_model: 0.65,
                strength_clip: 0.4,
            }],
            gen_params: { prompt: 'portrait' },
        });
        const loraNode = Object.values(result.prompt).find(
            node => node.class_type === 'LoraLoader'
        );

        expect(loraNode.inputs).toMatchObject({
            strength_model: 0.65,
            strength_clip: 0.4,
        });
    });

    it('matches a prompt alias to the structured Civitai LoRA instead of creating a missing duplicate', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{
                file_name: 'kaelakovalskia20IllustriousXL.safetensors',
                modelName: '[Dekinai] Kaela Kovalskia 2.0 Illustrious XL | Hololive',
                modelVersionId: 1591515,
                strength: 1,
            }],
            gen_params: {
                prompt: '<lora:Hololive - Kaela Kovalskia 2.0 (IL):0.5> portrait',
            },
        });
        const loraNodes = Object.values(result.prompt)
            .filter(node => node.class_type === 'LoraLoader');

        expect(loraNodes).toHaveLength(1);
        expect(loraNodes[0].inputs).toMatchObject({
            lora_name: 'kaelakovalskia20IllustriousXL.safetensors',
            strength_model: 0.5,
            strength_clip: 0.5,
        });
        expect(loraNodes[0]._meta.lora_aliases).toContain(
            'Hololive - Kaela Kovalskia 2.0 (IL)'
        );
    });

    it('does not fuzzy-match a short ambiguous prompt tag', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [
                { file_name: 'eyes_detail_a.safetensors' },
                { file_name: 'eyes_detail_b.safetensors' },
            ],
            gen_params: { prompt: '<lora:eyes:0.8> portrait' },
        });
        const loraNodes = Object.values(result.prompt)
            .filter(node => node.class_type === 'LoraLoader');

        expect(loraNodes).toHaveLength(3);
        expect(loraNodes.at(-1).inputs.lora_name).toBe('eyes.safetensors');
    });

    it('does not match unrelated LoRAs only because both names contain generic version tokens', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [{ file_name: 'MJ52.safetensors', modelVersionName: 'v1.0' }],
            gen_params: { prompt: '<lora:artfullyECHELIER_SDXL_V1:0.7> portrait' },
        });
        const loraNodes = Object.values(result.prompt)
            .filter(node => node.class_type === 'LoraLoader');

        expect(loraNodes).toHaveLength(2);
        expect(loraNodes.at(-1).inputs.lora_name).toBe('artfullyECHELIER_SDXL_V1.safetensors');
    });

    it('does not inject Civitai resource-list LoRAs absent from an authoritative A1111 prompt', () => {
        const result = buildRecipeWorkflow({
            a1111_parameters: '<lora:actual_style:0.4> portrait\nSteps: 20, Size: 768x1024',
            checkpoint: { file_name: 'model.safetensors' },
            loras: [
                { file_name: 'actual_style.safetensors' },
                { file_name: 'unreferenced_catalog_style.safetensors' },
            ],
            gen_params: { prompt: '<lora:actual_style:0.4> portrait' },
        });
        const names = Object.values(result.prompt)
            .filter(node => node.class_type === 'LoraLoader')
            .map(node => node.inputs.lora_name);

        expect(names).toEqual(['actual_style.safetensors']);
    });

    it('uses every evidence-backed manifest LoRA from A1111 resources', () => {
        const entries = [
            ['748cmSDXL', 0.45],
            ['NV_KawaiiTech_WM_IL_SH', 0.9],
            ['ILLMythP0rtr4itStyle', 0.7],
            ['tove-nikke-richy-v1_ixl', 1.0],
        ];
        const result = buildRecipeWorkflow({
            a1111_parameters: '<lora:tove-nikke-richy-v1_ixl:1> portrait\nSteps: 30, Size: 832x1216',
            checkpoint: { file_name: 'model.safetensors' },
            loras: entries.map(([name, strength]) => ({ file_name: name, strength })),
            gen_params: {
                prompt: '<lora:tove-nikke-richy-v1_ixl:1> portrait',
                size: '832x1216',
            },
            replay_manifest: {
                schema: 'lora-manager.replay-manifest',
                version: 1,
                manifest_hash: 'manifest-78353204',
                source_kind: 'a1111',
                errors: [],
                advisory_resources: [],
                required_resources: entries.map(([name, strength], index) => ({
                    requirement_id: `recipe:${index}`,
                    kind: 'lora',
                    required: true,
                    resource: { file_name: name },
                    resolution: { status: 'recipe_match', match: 'model_version_id' },
                    expected: { strength_model: strength, strength_clip: strength },
                    evidence: [{ source: 'a1111_civitai_resources' }],
                })),
            },
        });
        const loraNodes = Object.values(result.prompt)
            .filter(node => node.class_type === 'LoraLoader');

        expect(loraNodes.map(node => node.inputs.lora_name)).toEqual(
            entries.map(([name]) => `${name}.safetensors`)
        );
        expect(loraNodes.map(node => node.inputs.strength_model)).toEqual(
            entries.map(([, strength]) => strength)
        );
        expect(loraNodes.map(node => node._meta.replay_requirement.id)).toEqual([
            'recipe:0', 'recipe:1', 'recipe:2', 'recipe:3',
        ]);
        expect(result.replayManifest.manifest_hash).toBe('manifest-78353204');
    });

    it('keeps similarly named inline LoRAs distinct when no structured resources exist', () => {
        const result = buildRecipeWorkflow({
            checkpoint: { file_name: 'model.safetensors' },
            loras: [],
            gen_params: {
                prompt: [
                    '<lora:sampleStyleLora_v15:0.5>',
                    '<lora:sampleDetailLora_v10:0.2>',
                    'portrait',
                ].join(', '),
            },
        });
        const loraNodes = Object.values(result.prompt)
            .filter(node => node.class_type === 'LoraLoader');

        expect(loraNodes).toHaveLength(2);
        expect(loraNodes.map(node => node.inputs.lora_name)).toEqual([
            'sampleStyleLora_v15.safetensors',
            'sampleDetailLora_v10.safetensors',
        ]);
        expect(loraNodes.map(node => node.inputs.strength_model)).toEqual([0.5, 0.2]);
    });

    it('normalizes extensionless resource names', () => {
        expect(getResourceFilename({ file_name: 'folder\\model' })).toBe('model.safetensors');
    });

    it('uses the exact hash-matched local file for an installed recipe resource', () => {
        expect(getResourceFilename({
            inLibrary: true,
            file_name: 'model.safetensors',
            localPath: 'models/checkpoints/model-7eb867.safetensors',
        })).toBe('model-7eb867.safetensors');
    });

    it('preserves the ComfyUI-relative LoRA subfolder from an installed local path', () => {
        expect(getResourceFilename({
            inLibrary: true,
            file_name: 'GoodHands-beta2',
            localPath: 'D:/AI/forge/webui/models/Lora/SD 1.5/concept/GoodHands-beta2.safetensors',
        })).toBe('SD 1.5/concept/GoodHands-beta2.safetensors');
    });

    it('preserves the ComfyUI-relative checkpoint subfolder from an installed local path', () => {
        expect(getResourceFilename({
            inLibrary: true,
            file_name: 'anythingelseV4_v45',
            localPath: 'D:/AI/forge/webui/models/Stable-diffusion/SD 1.5/anime/anythingelseV4_v45.ckpt',
        }, 'Model')).toBe('SD 1.5/anime/anythingelseV4_v45.ckpt');
    });
});
