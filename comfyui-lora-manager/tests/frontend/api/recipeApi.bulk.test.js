import { describe, it, beforeEach, afterEach, expect, vi } from 'vitest';

const showToastMock = vi.hoisted(() => vi.fn());
const loadingManagerMock = vi.hoisted(() => ({
  showSimpleLoading: vi.fn(),
  show: vi.fn(),
  hide: vi.fn(),
  restoreProgressBar: vi.fn(),
}));
const virtualScrollerMock = vi.hoisted(() => ({
  updateSingleItem: vi.fn(),
  refreshWithData: vi.fn(),
}));
const getCurrentPageStateMock = vi.hoisted(() => vi.fn());
const captureScrollPositionMock = vi.hoisted(() => vi.fn());
const restoreScrollPositionMock = vi.hoisted(() => vi.fn());

vi.mock('../../../static/js/utils/uiHelpers.js', () => {
  return {
    showToast: showToastMock,
  };
});

vi.mock('../../../static/js/components/RecipeCard.js', () => ({
  RecipeCard: vi.fn(() => ({ element: document.createElement('div') })),
}));

vi.mock('../../../static/js/state/index.js', () => {
  return {
    state: {
      loadingManager: loadingManagerMock,
      virtualScroller: virtualScrollerMock,
    },
    getCurrentPageState: getCurrentPageStateMock,
  };
});

vi.mock('../../../static/js/utils/infiniteScroll.js', () => ({
  captureScrollPosition: captureScrollPositionMock,
  restoreScrollPosition: restoreScrollPositionMock,
}));

import {
  RecipeSidebarApiClient,
  activateRecipeRevision,
  adoptRecipeRevision,
  createRecipePromptDraft,
  fetchRecipesPage,
  fetchRecipeDetails,
  listRecipeRevisions,
  releaseRecipePromptModel,
  resetAndReload,
  syncChanges,
  updateRecipeMetadata
} from '../../../static/js/api/recipeApi.js';

describe('RecipeSidebarApiClient bulk operations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn();
    getCurrentPageStateMock.mockReturnValue({
      pageSize: 50,
      currentPage: 1,
      hasMore: true,
      isLoading: false,
      sortBy: 'date:desc',
      showFavoritesOnly: false,
      activeFolder: null,
      searchOptions: { recursive: true },
      customFilter: { active: false },
      filters: {},
    });
  });

  afterEach(() => {
    delete global.fetch;
  });

  it('sends recipe IDs when moving in bulk', async () => {
    const api = new RecipeSidebarApiClient();
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        results: [
          {
            recipe_id: 'abc',
            original_file_path: '/recipes/abc.webp',
            new_file_path: '/recipes/target/abc.webp',
            success: true,
          },
        ],
        success_count: 1,
        failure_count: 0,
      }),
    });

    const results = await api.moveBulkModels(['/recipes/abc.webp'], '/target/folder');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/lm/recipes/move-bulk',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const { body } = global.fetch.mock.calls[0][1];
    expect(JSON.parse(body)).toEqual({
      recipe_ids: ['abc'],
      target_path: '/target/folder',
    });

    expect(showToastMock).toHaveBeenCalledWith(
      'toast.api.bulkMoveSuccess',
      { successCount: 1, type: 'Recipe' },
      'success'
    );
    expect(results[0].recipe_id).toBe('abc');
  });

  it('passes exclude-favorite and prompt provenance filters to the recipe list API', async () => {
    getCurrentPageStateMock.mockReturnValue({
      pageSize: 20,
      sortBy: 'date:desc',
      favoriteFilter: 'exclude',
      replayCapabilityFilter: 'all',
      promptStatusFilter: 'missing',
      activeFolder: null,
      searchOptions: { recursive: true },
      customFilter: { active: false },
      filters: {},
    });
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ items: [], total: 0, total_pages: 0 }),
    });

    await fetchRecipesPage(1, 20);

    const requestUrl = new URL(global.fetch.mock.calls[0][0], 'http://localhost');
    expect(requestUrl.searchParams.get('favorite')).toBe('false');
    expect(requestUrl.searchParams.get('prompt_status')).toBe('missing');
  });

  it('posts recipe IDs for bulk delete', async () => {
    const api = new RecipeSidebarApiClient();
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        total_deleted: 2,
        total_failed: 0,
        failed: [],
      }),
    });

    const result = await api.bulkDeleteModels(['/recipes/a.webp', '/recipes/b.webp']);

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/lm/recipes/bulk-delete',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const parsedBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(parsedBody.recipe_ids).toEqual(['a', 'b']);
    expect(result).toMatchObject({
      success: true,
      deleted_count: 2,
      failed_count: 0,
    });
    expect(loadingManagerMock.hide).toHaveBeenCalled();
  });

  it('encodes recipe IDs when fetching recipe details', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'abc' }),
    });

    await fetchRecipeDetails('recipe#1?name=foo%bar');

    expect(global.fetch).toHaveBeenCalledWith('/api/lm/recipe/recipe%231%3Fname%3Dfoo%25bar');
  });

  it('requests the active immutable variant only when explicitly selected', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'recipe-1', revision_summary: { active: true } }),
    });

    await fetchRecipeDetails('recipe-1', { variant: 'active' });

    expect(global.fetch).toHaveBeenCalledWith('/api/lm/recipe/recipe-1?variant=active');
    await expect(fetchRecipeDetails('recipe-1', { variant: 'unknown' }))
      .rejects.toThrow('Unsupported recipe variant');
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('lists, adopts, and activates immutable revisions with source ETag protection', async () => {
    global.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, source_etag: 'source-etag', revisions: [] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, revision_id: 'revision-1' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, revision_summary: { active: false } }),
      });
    const draft = { draft_hash: 'd'.repeat(64), manifest_hash: 'm'.repeat(64) };
    const candidate = { prompt_id: 'prompt-1', candidate_id: 'candidate-1', seed: 42 };

    const listed = await listRecipeRevisions('recipe#1');
    const adopted = await adoptRecipeRevision('recipe#1', {
      sourceEtag: 'source-etag', manifestHash: 'm'.repeat(64), draft, candidate,
    });
    const activated = await activateRecipeRevision('recipe#1', {
      sourceEtag: '"source-etag"', revisionId: null,
    });

    expect(listed.revisions).toEqual([]);
    expect(adopted.revision_id).toBe('revision-1');
    expect(activated.revision_summary.active).toBe(false);
    expect(global.fetch).toHaveBeenNthCalledWith(
      1, '/api/lm/recipe/recipe%231/revisions'
    );
    expect(global.fetch).toHaveBeenNthCalledWith(
      2,
      '/api/lm/recipe/recipe%231/revisions',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'If-Match': '"source-etag"',
        },
      })
    );
    expect(JSON.parse(global.fetch.mock.calls[1][1].body)).toEqual({
      action: 'adopt', manifest_hash: 'm'.repeat(64), draft, candidate,
    });
    expect(global.fetch).toHaveBeenNthCalledWith(
      3,
      '/api/lm/recipe/recipe%231/revisions/active',
      expect.objectContaining({
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'If-Match': '"source-etag"',
        },
      })
    );
    expect(JSON.parse(global.fetch.mock.calls[2][1].body)).toEqual({
      action: 'activate', revision_id: null,
    });
  });

  it('refuses revision writes without a source ETag before sending a request', async () => {
    await expect(adoptRecipeRevision('recipe-1', {
      manifestHash: 'm'.repeat(64),
      draft: { draft_hash: 'd'.repeat(64) },
      candidate: { prompt_id: 'prompt-1' },
    })).rejects.toThrow('source ETag');

    await expect(activateRecipeRevision('recipe-1', {
      sourceEtag: '', revisionId: 'revision-1',
    })).rejects.toThrow('source ETag');
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('surfaces a safe server revision error message', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 412,
      json: async () => ({
        success: false,
        error: 'RECIPE_ETAG_CHANGED',
        message: '元レシピの生成情報が更新されています。',
      }),
    });

    await expect(activateRecipeRevision('recipe-1', {
      sourceEtag: 'old-etag', revisionId: 'revision-1',
    })).rejects.toThrow('元レシピの生成情報が更新されています。');
  });

  it('requests a local AI prompt draft with the current manifest hash', async () => {
    const manifestHash = 'a'.repeat(64);
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, draft: { proposed_prompt: 'portrait' } }),
    });

    const draft = await createRecipePromptDraft('recipe#1', {
      manifestHash,
      model: 'qwythos-q5',
      forceRegenerate: true,
    });

    expect(draft.proposed_prompt).toBe('portrait');
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/lm/recipe/recipe%231/ai-prompt-draft',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );
    expect(JSON.parse(global.fetch.mock.calls[0][1].body)).toEqual({
      action: 'draft',
      manifest_hash: manifestHash,
      model: 'qwythos-q5',
      force_regenerate: true,
    });
  });

  it('releases the managed local AI model before image generation', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        released_models: ['lora-manager-qwen35-q6'],
      }),
    });

    const released = await releaseRecipePromptModel('recipe#1');

    expect(released).toEqual(['lora-manager-qwen35-q6']);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/lm/recipe/recipe%231/ai-prompt-draft',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );
    expect(JSON.parse(global.fetch.mock.calls[0][1].body)).toEqual({ action: 'release' });
  });

  it('updates the virtual scroller using the original list path when provided', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });

    await updateRecipeMetadata(
      '/recipes/new-folder/recipe#1.webp',
      { title: 'Updated Title' },
      { listFilePath: '/recipes/old-folder/recipe#1.webp' }
    );

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/lm/recipe/recipe%231/update',
      expect.objectContaining({ method: 'PUT' })
    );
    expect(virtualScrollerMock.updateSingleItem).toHaveBeenCalledWith(
      '/recipes/old-folder/recipe#1.webp',
      { title: 'Updated Title' }
    );
  });

  it('reloads recipes without preserving scroll', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [{ id: 'recipe-1' }],
        total: 1,
        total_pages: 1,
      }),
    });

    await resetAndReload(false);

    expect(captureScrollPositionMock).not.toHaveBeenCalled();
    expect(virtualScrollerMock.refreshWithData).toHaveBeenCalledWith(
      [{ id: 'recipe-1' }],
      1,
      false
    );
    expect(restoreScrollPositionMock).not.toHaveBeenCalled();
  });

  it('uses scroll-free reloads for syncChanges', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [],
        total: 0,
        total_pages: 0,
      }),
    });

    await syncChanges();

    expect(captureScrollPositionMock).not.toHaveBeenCalled();
    expect(restoreScrollPositionMock).not.toHaveBeenCalled();
    expect(loadingManagerMock.restoreProgressBar).toHaveBeenCalledTimes(1);
  });
});
