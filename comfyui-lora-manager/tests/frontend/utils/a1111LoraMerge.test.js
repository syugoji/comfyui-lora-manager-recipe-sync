import { describe, expect, it } from 'vitest';

import {
    applyA1111LoraStrengths,
    injectA1111LoraTags,
} from '../../../web/comfyui/a1111_lora_merge.js';

describe('A1111 LoRA merge', () => {
    it('injects structured AddNet LoRAs into the positive prompt only once', () => {
        const parameters = 'portrait\nNegative prompt: bad\nSteps: 20, Sampler: Euler a';
        const loras = [
            { name: 'style\\Moxin_10.safetensors', strength_model: 0.3 },
            { name: 'firekeeper.safetensors', strength_model: 0.8 },
        ];

        const injected = injectA1111LoraTags(parameters, loras);
        const reinjected = injectA1111LoraTags(injected, loras);

        expect(injected).toContain(
            'portrait <lora:style\\Moxin_10.safetensors:0.3> <lora:firekeeper.safetensors:0.8>\nNegative prompt:'
        );
        expect(reinjected).toBe(injected);
    });

    it('restores separate model and clip strengths after native import', () => {
        const widgets = [
            { name: 'lora_name', value: 'style/Moxin_10.safetensors' },
            { name: 'strength_model', value: 1 },
            { name: 'strength_clip', value: 1 },
        ];
        const graph = {
            _nodes: [{ type: 'LoraLoader', widgets }],
        };

        applyA1111LoraStrengths(graph, [{
            name: 'style\\Moxin_10.safetensors',
            strength_model: 0.3,
            strength_clip: 0.2,
        }]);

        expect(widgets[1].value).toBe(0.3);
        expect(widgets[2].value).toBe(0.2);
        expect(widgets[0].value).toBe('style\\Moxin_10.safetensors');
    });

    it('rewrites existing extensionless tags to the resolved ComfyUI path before import', () => {
        const parameters = [
            'portrait, <lora:animeoutlineV4_16:1>',
            'Negative prompt: bad',
            'Steps: 20, Sampler: Euler a',
        ].join('\n');

        const normalized = injectA1111LoraTags(parameters, [{
            name: 'SD 1.5\\anime\\animeoutlineV4_16.safetensors',
            strength_model: 1,
            strength_clip: 1,
        }]);

        expect(normalized).toContain(
            '<lora:SD 1.5\\anime\\animeoutlineV4_16.safetensors:1>'
        );
        expect(normalized).not.toContain('<lora:animeoutlineV4_16:1>');
    });

    it('restores a raw t-prefixed prompt alias to the exact library path', () => {
        const widgets = [
            { name: 'lora_name', value: 'sampleDetailLora_v10' },
            { name: 'strength_model', value: 1 },
            { name: 'strength_clip', value: 1 },
        ];
        const graph = {
            _nodes: [{ type: 'LoraLoader', widgets }],
        };

        applyA1111LoraStrengths(graph, [{
            name: '_prompt_auto_resolved\\sampleDetailLora_v10.safetensors',
            strength_model: 0.2,
            strength_clip: 0.2,
        }]);

        expect(widgets[0].value).toBe(
            '_prompt_auto_resolved\\sampleDetailLora_v10.safetensors'
        );
        expect(widgets[1].value).toBe(0.2);
        expect(widgets[2].value).toBe(0.2);
    });

    it('rewrites a unique recipe alias to the canonical library path without duplication', () => {
        const parameters = [
            'portrait, <lora:Styles\\Twilight Style:0.8>',
            'Negative prompt: bad',
            'Steps: 20, Sampler: Euler a',
        ].join('\n');
        const lora = {
            name: 'Pony\\anime\\Concept Art Twilight Style.safetensors',
            aliases: ['Styles\\Twilight Style'],
            strength_model: 0.8,
            strength_clip: 0.8,
        };

        const normalized = injectA1111LoraTags(parameters, [lora]);
        const tags = normalized.match(/<lora:[^>]+>/g) || [];

        expect(tags).toEqual([
            '<lora:Pony\\anime\\Concept Art Twilight Style.safetensors:0.8>',
        ]);

        const widgets = [
            { name: 'lora_name', value: 'Styles\\Twilight Style' },
            { name: 'strength_model', value: 1 },
            { name: 'strength_clip', value: 1 },
        ];
        applyA1111LoraStrengths(
            { _nodes: [{ type: 'LoraLoader', widgets }] },
            [lora],
        );
        expect(widgets.map(widget => widget.value)).toEqual([
            'Pony\\anime\\Concept Art Twilight Style.safetensors',
            0.8,
            0.8,
        ]);
    });

    it('does not rewrite an alias claimed by more than one LoRA', () => {
        const parameters = 'portrait, <lora:shared alias:0.5>\nSteps: 20';
        const normalized = injectA1111LoraTags(parameters, [
            { name: 'first.safetensors', aliases: ['shared alias'] },
            { name: 'second.safetensors', aliases: ['shared alias'] },
        ]);

        expect(normalized).toContain('<lora:shared alias:0.5>');
    });
});
