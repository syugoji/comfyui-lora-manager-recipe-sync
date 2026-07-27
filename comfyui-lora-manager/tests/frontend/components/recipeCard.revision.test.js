import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../static/js/utils/uiHelpers.js', () => ({
    showToast: vi.fn(),
    copyToClipboard: vi.fn(),
    sendLoraToWorkflow: vi.fn(),
}));

vi.mock('../../../static/js/api/recipeApi.js', () => ({
    fetchRecipeDetails: vi.fn().mockResolvedValue({}),
    updateRecipeMetadata: vi.fn(),
}));

vi.mock('../../../static/js/components/shared/ModelCard.js', () => ({
    configureModelCardVideo: vi.fn(),
}));

vi.mock('../../../static/js/managers/ModalManager.js', () => ({
    modalManager: { showModal: vi.fn() },
}));

vi.mock('../../../static/js/state/index.js', () => ({
    getCurrentPageState: vi.fn(() => ({ duplicatesMode: false })),
    state: {
        settings: { blur_mature_content: false },
        global: { settings: {} },
    },
}));

vi.mock('../../../static/js/managers/BulkManager.js', () => ({
    bulkManager: {},
}));

vi.mock('../../../static/js/utils/constants.js', () => ({
    NSFW_LEVELS: { PG: 1, PG13: 2, R: 4, X: 8, XXX: 16 },
    getBaseModelAbbreviation: vi.fn(value => value),
    getMatureBlurThreshold: vi.fn(() => 16),
}));

vi.mock('../../../static/js/utils/recipeReplayCapability.js', () => ({
    analyzeRecipeReplayCapability: vi.fn().mockResolvedValue({
        level: 'compatible', title: 'Compatible', iconClass: 'fas fa-check', label: 'Compatible',
    }),
    getRecipePromptStatus: vi.fn(() => 'generated'),
}));

describe('RecipeCard active revision badge', () => {
    beforeEach(() => {
        document.body.innerHTML = '';
    });

    it('marks an active LM Studio revision as modified without interpreting metadata as HTML', async () => {
        const { RecipeCard } = await import('../../../static/js/components/RecipeCard.js');
        const card = new RecipeCard({
            id: 'recipe-revision',
            title: 'Recipe revision',
            file_path: 'recipe.png',
            file_url: '/view?filename=recipe.png&type=output',
            base_model: 'Illustrious',
            loras: [],
            revision_summary: {
                active: true,
                stale: false,
                prompt_source: 'lm_studio',
                seed: 42,
                model: '<img src=x onerror=alert(1)>',
                created_at: '2026-07-17T00:00:00Z',
            },
        }, vi.fn());

        const badge = card.element.querySelector('.recipe-prompt-status.generated');
        expect(badge.textContent).toContain('AI補完・改変済み');
        expect(badge.title).toContain('元レシピ未変更');
        expect(badge.title).toContain('<img src=x');
        expect(badge.querySelector('img')).toBeNull();
    });
});
