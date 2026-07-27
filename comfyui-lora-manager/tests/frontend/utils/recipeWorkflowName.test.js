import { describe, expect, it } from 'vitest';

import { createRecipeWorkflowName } from '../../../static/js/utils/recipeWorkflowName.js';

describe('recipe workflow name', () => {
    it('uses the Civitai recipe number instead of an internal UUID and timestamp', () => {
        expect(createRecipeWorkflowName({
            id: 'b1221d96-eb42-41d1-9107-337703c7f029',
            title: 'Civitai_Recipe_51644312',
        })).toBe('Civitai_Recipe_51644312');
    });

    it('falls back to a stable sanitized title', () => {
        expect(createRecipeWorkflowName({ title: 'My saved recipe' })).toBe('My_saved_recipe');
    });
});
