import { describe, expect, it } from 'vitest';
import { buildRecipeReferenceText } from '../../../static/js/utils/recipeReferenceInfo.js';

describe('buildRecipeReferenceText', () => {
    it('assembles checkpoint, lora stack, prompts and params into plain text', () => {
        const recipe = {
            title: 'Civitai_Recipe_43591898',
            checkpoint: {
                name: 'WAI-NSFW-illustrious',
                version: 'v14.0',
                hash: 'abcdef0123456789',
                existsLocally: true,
            },
            loras: [
                {
                    name: 'USNR STYLE',
                    weight: 0.5,
                    hash: '1111222233334444',
                    existsLocally: true,
                },
                {
                    file_name: 'gone_lora',
                    strength: 0.8,
                    hash: '',
                    existsLocally: false,
                    isDeleted: true,
                },
            ],
            embeddings: [
                { name: 'EasyNegative', hash: '9999', existsLocally: false },
            ],
            gen_params: {
                prompt: '1girl, solo',
                negative_prompt: 'lowres',
                steps: 30,
                sampler: 'euler_ancestral',
                cfg_scale: 6,
                seed: 12345,
                size: '1024x1024',
            },
            source_path: 'https://civitai.com/images/101967666',
        };

        const text = buildRecipeReferenceText(recipe);

        expect(text).toContain('レシピ: Civitai_Recipe_43591898');
        expect(text).toContain(
            'Checkpoint: WAI-NSFW-illustrious (v14.0) [abcdef012345]'
        );
        expect(text).toContain('LoRA: USNR STYLE [111122223333] x0.5');
        expect(text).toContain('LoRA: gone_lora ※配布終了 x0.8');
        expect(text).toContain('Embedding: EasyNegative [9999] ※未所持');
        expect(text).toContain('Prompt:\n1girl, solo');
        expect(text).toContain('Negative prompt:\nlowres');
        expect(text).toContain(
            'Steps: 30 | Sampler: euler_ancestral | CFG: 6 | Seed: 12345 | Size: 1024x1024'
        );
        expect(text).toContain('Source: https://civitai.com/images/101967666');
    });

    it('marks locally missing resources without a deleted flag as 未所持', () => {
        const text = buildRecipeReferenceText({
            loras: [{ name: 'MissingLora', existsLocally: false }],
        });

        expect(text).toContain('LoRA: MissingLora ※未所持 x1');
    });

    it('marks 未所持 from the recipe detail payload, which uses inLibrary instead of existsLocally', () => {
        const text = buildRecipeReferenceText({
            checkpoint: { name: 'OwnedCheckpoint', inLibrary: true },
            loras: [
                { name: 'MissingLora', inLibrary: false, weight: 0.8 },
                { name: 'OwnedLora', inLibrary: true, weight: 0.5 },
            ],
            embeddings: [{ name: 'MissingEmbedding', inLibrary: false }],
        });

        expect(text).toContain('LoRA: MissingLora ※未所持 x0.8');
        expect(text).toContain('Embedding: MissingEmbedding ※未所持');
        expect(text).toContain('Checkpoint: OwnedCheckpoint');
        expect(text).not.toContain('OwnedCheckpoint ※');
        expect(text).toContain('LoRA: OwnedLora x0.5');
        expect(text).not.toContain('OwnedLora ※');
    });

    it('keeps 配布終了 taking precedence over 未所持', () => {
        const text = buildRecipeReferenceText({
            loras: [{ name: 'GoneLora', inLibrary: false, isDeleted: true }],
        });

        expect(text).toContain('LoRA: GoneLora ※配布終了 x1');
        expect(text).not.toContain('※未所持');
    });

    it('returns an empty string for empty or invalid input', () => {
        expect(buildRecipeReferenceText(null)).toBe('');
        expect(buildRecipeReferenceText({})).toBe('');
        expect(buildRecipeReferenceText('nope')).toBe('');
    });
});
