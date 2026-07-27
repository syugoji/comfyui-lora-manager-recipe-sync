import { describe, expect, it, vi } from 'vitest';

import {
    normalizeA1111Text,
    parseA1111Options,
    patchA1111Graph,
} from '../../../web/comfyui/a1111_generation_patch.js';

function node(comfyClass, values) {
    return {
        comfyClass,
        widgets: Object.entries(values).map(([name, value]) => ({
            name,
            value,
            callback: vi.fn(),
        })),
    };
}

function widgetValue(target, name) {
    return target.widgets.find(widget => widget.name === name)?.value;
}

describe('A1111 generation replay patch', () => {
    const parameters = [
        '<lora:USNR_STYLE_NB_Vpred_V3-000019:0.9>, (usnr), illustration',
        'Negative prompt: ',
        'Steps: 20, Sampler: Euler a, Schedule type: Exponential, CFG scale: 5, Seed: 2607780138, Size: 784x1040, Model: ntdmixvpredv1.5, Denoising strength: 0.52, Hires CFG Scale: 5, Hires upscale: 1.5, Hires upscaler: remacri_original',
    ].join('\n');

    it('repairs byte-swapped UTF-16 metadata', () => {
        const swapped = [...parameters].map(char => {
            const code = char.charCodeAt(0);
            return String.fromCharCode(((code & 0xff) << 8) | (code >>> 8));
        }).join('');

        expect(normalizeA1111Text(swapped)).toBe(parameters);
    });

    it('parses separate scheduler and hires values', () => {
        expect(parseA1111Options(parameters)).toMatchObject({
            sampler: 'Euler a',
            'schedule type': 'Exponential',
            size: '784x1040',
            'hires upscale': '1.5',
            'hires cfg scale': '5',
        });
    });

    it('patches native import nodes to exact source settings', () => {
        const baseSampler = node('KSampler', {
            seed: 0, steps: 1, cfg: 1, sampler_name: 'euler', scheduler: 'normal', denoise: 1,
        });
        const hiresSampler = node('KSampler', {
            seed: 0, steps: 1, cfg: 1, sampler_name: 'euler', scheduler: 'normal', denoise: 1,
        });
        const latent = node('EmptyLatentImage', { width: 832, height: 1088 });
        const scale = node('ImageScale', { width: 1216, height: 1600 });
        const checkpoint = node('CheckpointLoaderSimple', { ckpt_name: 'wrong.safetensors' });
        const graph = { _nodes: [baseSampler, hiresSampler, latent, scale, checkpoint] };

        const result = patchA1111Graph(graph, parameters, {
            checkpointName: 'models\\ntdmixvpredv1.5.safetensors',
        });

        expect(result.targetSize).toEqual({ width: 1176, height: 1560 });
        expect(widgetValue(baseSampler, 'sampler_name')).toBe('euler_ancestral');
        expect(widgetValue(baseSampler, 'scheduler')).toBe('exponential');
        expect(widgetValue(hiresSampler, 'cfg')).toBe(5);
        expect(widgetValue(hiresSampler, 'denoise')).toBe(0.52);
        expect(widgetValue(latent, 'width')).toBe(784);
        expect(widgetValue(latent, 'height')).toBe(1040);
        expect(widgetValue(scale, 'width')).toBe(1176);
        expect(widgetValue(scale, 'height')).toBe(1560);
        expect(widgetValue(checkpoint, 'ckpt_name')).toBe('models\\ntdmixvpredv1.5.safetensors');
    });
});
