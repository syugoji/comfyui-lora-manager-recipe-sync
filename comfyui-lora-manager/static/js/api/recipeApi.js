import { RecipeCard } from '../components/RecipeCard.js';
import { state, getCurrentPageState } from '../state/index.js';
import { showToast } from '../utils/uiHelpers.js';
import { captureScrollPosition, restoreScrollPosition } from '../utils/infiniteScroll.js';
import { analyzeRecipeReplayCapability, getComfyObjectInfo } from '../utils/recipeReplayCapability.js';

const RECIPE_ENDPOINTS = {
    list: '/api/lm/recipes',
    detail: '/api/lm/recipe',
    scan: '/api/lm/recipes/scan',
    update: '/api/lm/recipe',
    roots: '/api/lm/recipes/roots',
    folders: '/api/lm/recipes/folders',
    folderTree: '/api/lm/recipes/folder-tree',
    unifiedFolderTree: '/api/lm/recipes/unified-folder-tree',
    move: '/api/lm/recipe/move',
    moveBulk: '/api/lm/recipes/move-bulk',
    bulkDelete: '/api/lm/recipes/bulk-delete',
    repairBulk: '/api/lm/recipes/repair-bulk',
};

const RECIPE_SIDEBAR_CONFIG = {
    config: {
        displayName: 'Recipe',
        supportsMove: true,
    },
    endpoints: RECIPE_ENDPOINTS,
};

const capabilityCache = new Map();
const CAPABILITY_CACHE_MS = 60_000;

async function mapWithConcurrency(items, concurrency, mapper) {
    const results = new Array(items.length);
    let nextIndex = 0;
    async function worker() {
        while (nextIndex < items.length) {
            const index = nextIndex++;
            results[index] = await mapper(items[index], index);
        }
    }
    await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
    return results;
}

async function filterByReplayCapability(items, requestedLevel) {
    if (!requestedLevel || requestedLevel === 'all' || items.length === 0) return items;
    const objectInfo = await getComfyObjectInfo();
    const classified = await mapWithConcurrency(items, 8, async item => {
        const recipeId = item.id || item.recipe_id || extractRecipeId(item.file_path);
        const cacheKey = `${recipeId}:${item.modified || ''}`;
        let cached = capabilityCache.get(cacheKey);
        if (!cached || Date.now() - cached.createdAt > CAPABILITY_CACHE_MS) {
            const promise = fetchRecipeDetails(recipeId)
                .then(recipe => analyzeRecipeReplayCapability(recipe, { objectInfo }));
            cached = { promise, createdAt: Date.now() };
            capabilityCache.set(cacheKey, cached);
        }
        try {
            const capability = await cached.promise;
            return capability.level === requestedLevel ? item : null;
        } catch (error) {
            capabilityCache.delete(cacheKey);
            return requestedLevel === 'unavailable' ? item : null;
        }
    });
    return classified.filter(Boolean);
}

export function extractRecipeId(filePath) {
    if (!filePath) return null;
    const basename = filePath.split('/').pop().split('\\').pop();
    const dotIndex = basename.lastIndexOf('.');
    return dotIndex > 0 ? basename.substring(0, dotIndex) : basename;
}

function requireRecipeId(recipeId) {
    if (!recipeId) {
        throw new Error('Unable to determine recipe ID');
    }
    return encodeURIComponent(recipeId);
}

async function readApiPayload(response, fallbackMessage) {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload?.success === false) {
        const fallback = response.statusText
            ? `${fallbackMessage}: ${response.statusText}`
            : response.status
                ? `${fallbackMessage} (${response.status})`
                : fallbackMessage;
        throw new Error(
            payload?.message || payload?.error || fallback
        );
    }
    return payload;
}

function formatIfMatch(sourceEtag) {
    const value = String(sourceEtag || '').trim();
    const opaque = value.startsWith('"') && value.endsWith('"')
        ? value.slice(1, -1)
        : value;
    if (!opaque || opaque.startsWith('W/') || /[\r\n"]/u.test(opaque)) {
        throw new Error('A valid source ETag is required');
    }
    return `"${opaque}"`;
}

export async function fetchRecipeDetails(recipeId, { variant = null } = {}) {
    const encodedRecipeId = requireRecipeId(recipeId);
    if (variant !== null && variant !== undefined && variant !== 'active') {
        throw new Error(`Unsupported recipe variant: ${variant}`);
    }

    const suffix = variant === 'active' ? '?variant=active' : '';
    const response = await fetch(`${RECIPE_ENDPOINTS.detail}/${encodedRecipeId}${suffix}`);
    return readApiPayload(response, 'Failed to load recipe');
}

export async function listRecipeRevisions(recipeId) {
    const encodedRecipeId = requireRecipeId(recipeId);
    const response = await fetch(`${RECIPE_ENDPOINTS.detail}/${encodedRecipeId}/revisions`);
    return readApiPayload(response, 'Failed to load recipe revisions');
}

export async function adoptRecipeRevision(
    recipeId,
    { sourceEtag, manifestHash, draft, candidate } = {}
) {
    const encodedRecipeId = requireRecipeId(recipeId);
    if (!manifestHash || !draft || typeof draft !== 'object'
        || !candidate || typeof candidate !== 'object') {
        throw new Error('Manifest, draft, and candidate are required to adopt a revision');
    }
    const response = await fetch(`${RECIPE_ENDPOINTS.detail}/${encodedRecipeId}/revisions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'If-Match': formatIfMatch(sourceEtag),
        },
        body: JSON.stringify({
            action: 'adopt',
            manifest_hash: manifestHash,
            draft,
            candidate,
        }),
    });
    return readApiPayload(response, 'Failed to adopt recipe revision');
}

export async function activateRecipeRevision(recipeId, { sourceEtag, revisionId = null } = {}) {
    const encodedRecipeId = requireRecipeId(recipeId);
    if (revisionId !== null && (typeof revisionId !== 'string' || !revisionId.trim())) {
        throw new Error('Revision ID must be a non-empty string or null');
    }
    const response = await fetch(
        `${RECIPE_ENDPOINTS.detail}/${encodedRecipeId}/revisions/active`,
        {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'If-Match': formatIfMatch(sourceEtag),
            },
            body: JSON.stringify({ action: 'activate', revision_id: revisionId }),
        }
    );
    return readApiPayload(response, 'Failed to activate recipe revision');
}

export async function createRecipePromptDraft(recipeId, {
    manifestHash = null,
    model = 'qwen35-q6',
    forceRegenerate = false,
} = {}) {
    if (!recipeId) throw new Error('Unable to determine recipe ID');
    const encodedRecipeId = encodeURIComponent(recipeId);
    const response = await fetch(
        `${RECIPE_ENDPOINTS.detail}/${encodedRecipeId}/ai-prompt-draft`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'draft',
                manifest_hash: manifestHash,
                model,
                force_regenerate: forceRegenerate,
            }),
        }
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.success) {
        throw new Error(payload.message || payload.error || `AI prompt draft failed (${response.status})`);
    }
    return payload.draft;
}

export async function releaseRecipePromptModel(recipeId) {
    if (!recipeId) throw new Error('Unable to determine recipe ID');
    const encodedRecipeId = encodeURIComponent(recipeId);
    const response = await fetch(
        `${RECIPE_ENDPOINTS.detail}/${encodedRecipeId}/ai-prompt-draft`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'release' }),
        }
    );
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.success) {
        throw new Error(payload.message || payload.error || `AI model release failed (${response.status})`);
    }
    return payload.released_models || [];
}

/**
 * Fetch recipes with pagination for virtual scrolling
 * @param {number} page - Page number to fetch
 * @param {number} pageSize - Number of items per page
 * @returns {Promise<Object>} Object containing items, total count, and pagination info
 */
export async function fetchRecipesPage(page = 1, pageSize = 100) {
    const pageState = getCurrentPageState();

    try {
        const capabilityFilter = pageState.replayCapabilityFilter || 'all';
        const needsCapabilityScan = capabilityFilter !== 'all';
        const params = new URLSearchParams({
            page: needsCapabilityScan ? 1 : page,
            page_size: needsCapabilityScan ? 10000 : (pageSize || pageState.pageSize || 20),
            sort_by: pageState.sortBy
        });

        const favoriteFilter = pageState.favoriteFilter
            || (pageState.showFavoritesOnly ? 'only' : 'all');
        if (favoriteFilter === 'only') {
            params.append('favorite', 'true');
        } else if (favoriteFilter === 'exclude') {
            params.append('favorite', 'false');
        }

        if (pageState.promptStatusFilter && pageState.promptStatusFilter !== 'all') {
            params.append('prompt_status', pageState.promptStatusFilter);
        }

        if (pageState.activeFolder !== null && pageState.activeFolder !== undefined) {
            params.append('folder', pageState.activeFolder);
            params.append('recursive', pageState.searchOptions?.recursive !== false);
        } else if (pageState.searchOptions?.recursive !== undefined) {
            params.append('recursive', pageState.searchOptions.recursive);
        }

        // If we have a specific recipe ID to load
        if (pageState.customFilter?.active && pageState.customFilter?.recipeId) {
            // Special case: load specific recipe
            const response = await fetch(
                `${RECIPE_ENDPOINTS.detail}/${encodeURIComponent(pageState.customFilter.recipeId)}`
            );

            if (!response.ok) {
                throw new Error(`Failed to load recipe: ${response.statusText}`);
            }

            const recipe = await response.json();

            // Return in expected format
            return {
                items: [recipe],
                totalItems: 1,
                totalPages: 1,
                currentPage: 1,
                hasMore: false
            };
        }

        // Add custom filter for Lora if present
        if (pageState.customFilter?.active && pageState.customFilter?.loraHash) {
            params.append('lora_hash', pageState.customFilter.loraHash);
            params.append('bypass_filters', 'true');
        } else if (pageState.customFilter?.active && pageState.customFilter?.checkpointHash) {
            params.append('checkpoint_hash', pageState.customFilter.checkpointHash);
            params.append('bypass_filters', 'true');
        } else {
            // Normal filtering logic

            // Add search filter if present
            if (pageState.filters?.search) {
                params.append('search', pageState.filters.search);

                // Add search option parameters
                if (pageState.searchOptions) {
                    params.append('search_title', pageState.searchOptions.title.toString());
                    params.append('search_tags', pageState.searchOptions.tags.toString());
                    params.append('search_lora_name', pageState.searchOptions.loraName.toString());
                    params.append('search_lora_model', pageState.searchOptions.loraModel.toString());
                    params.append('search_prompt', (pageState.searchOptions.prompt || false).toString());
                    params.append('fuzzy', 'true');
                }
            }

            // Add base model filters
            if (pageState.filters?.baseModel && pageState.filters.baseModel.length) {
                // Check for empty wildcard marker - if present, no models should match
                const EMPTY_WILDCARD_MARKER = '__EMPTY_WILDCARD_RESULT__';
                if (pageState.filters.baseModel.length === 1 && 
                    pageState.filters.baseModel[0] === EMPTY_WILDCARD_MARKER) {
                    // Wildcard resolved to no matches - return empty results
                    return {
                        items: [],
                        totalItems: 0,
                        totalPages: 0,
                        currentPage: page,
                        hasMore: false
                    };
                }
                params.append('base_models', pageState.filters.baseModel.join(','));
            }

            // Add tag filters
            if (pageState.filters?.tags && Object.keys(pageState.filters.tags).length) {
                Object.entries(pageState.filters.tags).forEach(([tag, state]) => {
                    if (state === 'include') {
                        params.append('tag_include', tag);
                    } else if (state === 'exclude') {
                        params.append('tag_exclude', tag);
                    }
                });
            }
        }

        // Fetch recipes
        const response = await fetch(`${RECIPE_ENDPOINTS.list}?${params.toString()}`);

        if (!response.ok) {
            throw new Error(`Failed to load recipes: ${response.statusText}`);
        }

        const data = await response.json();

        let items = data.items;
        let totalItems = data.total;
        let totalPages = data.total_pages;
        if (needsCapabilityScan) {
            const filtered = await filterByReplayCapability(items, capabilityFilter);
            totalItems = filtered.length;
            totalPages = Math.ceil(totalItems / pageSize);
            const start = (page - 1) * pageSize;
            items = filtered.slice(start, start + pageSize);
        }

        return {
            items,
            totalItems,
            totalPages,
            currentPage: page,
            hasMore: page < totalPages
        };
    } catch (error) {
        console.error('Error fetching recipes:', error);
        showToast('toast.recipes.fetchFailed', { message: error.message }, 'error');
        throw error;
    }
}

/**
 * Reset and reload models using virtual scrolling
 * @param {Object} options - Operation options
 * @returns {Promise<Object>} The fetch result
 */
export async function resetAndReloadWithVirtualScroll(options = {}) {
    const {
        modelType = 'lora',
        updateFolders = false,
        fetchPageFunction,
        preserveScroll = false
    } = options;

    const pageState = getCurrentPageState();
    const scrollSnapshot = preserveScroll ? captureScrollPosition() : null;

    try {
        pageState.isLoading = true;

        // Reset page counter
        pageState.currentPage = 1;

        const pageSize = state.virtualScroller?.pageSize || pageState.pageSize || 100;
        const result = await fetchPageFunction(1, pageSize);

        // Update the virtual scroller
        state.virtualScroller.refreshWithData(
            result.items,
            result.totalItems,
            result.hasMore
        );

        // Update state
        pageState.hasMore = result.hasMore;
        pageState.currentPage = 2; // Next page will be 2

        if (scrollSnapshot) {
            await restoreScrollPosition(scrollSnapshot);
        } else if (state.virtualScroller?.scrollContainer) {
            state.virtualScroller.scrollContainer.scrollTop = 0;
        }

        return result;
    } catch (error) {
        console.error(`Error reloading ${modelType}s:`, error);
        showToast('toast.recipes.reloadFailed', { modelType: modelType, message: error.message }, 'error');
        throw error;
    } finally {
        pageState.isLoading = false;
    }
}

/**
 * Load more models using virtual scrolling
 * @param {Object} options - Operation options
 * @returns {Promise<Object>} The fetch result
 */
export async function loadMoreWithVirtualScroll(options = {}) {
    const {
        modelType = 'lora',
        resetPage = false,
        updateFolders = false,
        fetchPageFunction,
        preserveScroll = false
    } = options;

    const pageState = getCurrentPageState();
    const scrollSnapshot = preserveScroll ? captureScrollPosition() : null;

    try {
        // Start loading state
        pageState.isLoading = true;

        // Reset to first page if requested
        if (resetPage) {
            pageState.currentPage = 1;
        }

        const pageSize = state.virtualScroller?.pageSize || pageState.pageSize || 100;
        const result = await fetchPageFunction(pageState.currentPage, pageSize);

        // Update virtual scroller with the new data
        state.virtualScroller.refreshWithData(
            result.items,
            result.totalItems,
            result.hasMore
        );

        // Update state
        pageState.hasMore = result.hasMore;
        pageState.currentPage = 2; // Next page to load would be 2

        if (scrollSnapshot) {
            await restoreScrollPosition(scrollSnapshot);
        }

        return result;
    } catch (error) {
        console.error(`Error loading ${modelType}s:`, error);
        showToast('toast.recipes.loadFailed', { modelType: modelType, message: error.message }, 'error');
        throw error;
    } finally {
        pageState.isLoading = false;
    }
}

/**
 * Reset and reload recipes using virtual scrolling
 * @param {boolean} updateFolders - Whether to update folder tags
 * @returns {Promise<Object>} The fetch result
 */
export async function resetAndReload(updateFolders = false, options = {}) {
    return resetAndReloadWithVirtualScroll({
        modelType: 'recipe',
        updateFolders,
        fetchPageFunction: fetchRecipesPage,
        preserveScroll: options.preserveScroll === true
    });
}

/**
 * Refreshes the recipe list by triggering a backend scan, then reloading.
 * @param {boolean} fullRebuild - If true, fully rebuild the cache; if false, incremental scan
 */
export async function syncChanges() {
    return refreshRecipes(false);
}

export async function refreshRecipes(fullRebuild = true) {
    const actionLabel = fullRebuild ? 'Rebuilding recipe cache' : 'Refreshing recipes';
    const actionToast = fullRebuild ? 'Full rebuild' : 'Refresh';

    try {
        state.loadingManager.show(`${actionLabel}...`, 0);

        const url = new URL(RECIPE_ENDPOINTS.scan, window.location.origin);
        url.searchParams.append('full_rebuild', fullRebuild);

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Failed to refresh recipe cache: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        if (data.status === 'cancelled') {
            showToast('toast.api.operationCancelled', {}, 'info');
            return;
        }

        await resetAndReload(false);

        showToast('toast.api.refreshComplete', { action: actionToast }, 'success');
    } catch (error) {
        console.error('Error refreshing recipes:', error);
        showToast('toast.api.refreshFailed', { action: fullRebuild ? 'rebuild' : 'refresh', type: 'recipe' }, 'error');
    } finally {
        state.loadingManager.hide();
        state.loadingManager.restoreProgressBar();
    }
}

/**
 * Load more recipes with pagination - updated to work with VirtualScroller
 * @param {boolean} resetPage - Whether to reset to the first page
 * @returns {Promise<void>}
 */
export async function loadMoreRecipes(resetPage = false) {
    const pageState = getCurrentPageState();

    // Use virtual scroller if available
    if (state.virtualScroller) {
        return loadMoreWithVirtualScroll({
            modelType: 'recipe',
            resetPage,
            updateFolders: false,
            fetchPageFunction: fetchRecipesPage
        });
    }
}

/**
 * Create a recipe card instance from recipe data
 * @param {Object} recipe - Recipe data
 * @returns {HTMLElement} Recipe card DOM element
 */
export function createRecipeCard(recipe) {
    const recipeCard = new RecipeCard(recipe, (recipe) => {
        if (window.recipeManager) {
            window.recipeManager.showRecipeDetails(recipe);
        }
    });
    return recipeCard.element;
}

/**
 * Update recipe metadata on the server
 * @param {string} filePath - The file path of the recipe (e.g. D:/Workspace/ComfyUI/models/loras/recipes/86b4c335-ecfc-4791-89d2-3746e55a7614.webp)
 * @param {Object} updates - The metadata updates to apply
 * @returns {Promise<Object>} The updated recipe data
 */
export async function updateRecipeMetadata(filePath, updates, options = {}) {
    try {
        state.loadingManager.showSimpleLoading('Saving metadata...');
        const listFilePath = options.listFilePath || filePath;

        // Extract recipeId from filePath (basename without extension)
        const recipeId = extractRecipeId(filePath);
        if (!recipeId) {
            throw new Error('Unable to determine recipe ID');
        }

        const response = await fetch(`${RECIPE_ENDPOINTS.update}/${encodeURIComponent(recipeId)}/update`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(updates)
        });

        const data = await response.json();

        if (!data.success) {
            showToast('toast.recipes.updateFailed', { error: data.error }, 'error');
            throw new Error(data.error || 'Failed to update recipe');
        }

        state.virtualScroller.updateSingleItem(listFilePath, updates);

        return data;
    } catch (error) {
        console.error('Error updating recipe:', error);
        showToast('toast.recipes.updateError', { message: error.message }, 'error');
        throw error;
    } finally {
        state.loadingManager.hide();
    }
}

export class RecipeSidebarApiClient {
    constructor() {
        this.apiConfig = RECIPE_SIDEBAR_CONFIG;
    }

    async fetchUnifiedFolderTree() {
        const response = await fetch(this.apiConfig.endpoints.unifiedFolderTree);
        if (!response.ok) {
            throw new Error('Failed to fetch recipe folder tree');
        }
        return response.json();
    }

    async fetchModelRoots() {
        const response = await fetch(this.apiConfig.endpoints.roots);
        if (!response.ok) {
            throw new Error('Failed to fetch recipe roots');
        }
        return response.json();
    }

    async fetchModelFolders() {
        const response = await fetch(this.apiConfig.endpoints.folders);
        if (!response.ok) {
            throw new Error('Failed to fetch recipe folders');
        }
        return response.json();
    }

    async moveBulkModels(filePaths, targetPath) {
        if (!this.apiConfig.config.supportsMove) {
            showToast('toast.api.bulkMoveNotSupported', { type: this.apiConfig.config.displayName }, 'warning');
            return [];
        }

        const recipeIds = filePaths
            .map((path) => extractRecipeId(path))
            .filter((id) => !!id);

        if (recipeIds.length === 0) {
            showToast('toast.models.noModelsSelected', {}, 'warning');
            return [];
        }

        const response = await fetch(this.apiConfig.endpoints.moveBulk, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recipe_ids: recipeIds,
                target_path: targetPath,
            }),
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || `Failed to move ${this.apiConfig.config.displayName}s`);
        }

        if (result.failure_count > 0) {
            showToast(
                'toast.api.bulkMovePartial',
                {
                    successCount: result.success_count,
                    type: this.apiConfig.config.displayName,
                    failureCount: result.failure_count,
                },
                'warning'
            );

            const failedFiles = (result.results || [])
                .filter((item) => !item.success)
                .map((item) => item.message || 'Unknown error');

            if (failedFiles.length > 0) {
                const failureMessage =
                    failedFiles.length <= 3
                        ? failedFiles.join('\n')
                        : `${failedFiles.slice(0, 3).join('\n')}\n(and ${failedFiles.length - 3} more)`;
                showToast('toast.api.bulkMoveFailures', { failures: failureMessage }, 'warning', 6000);
            }
        } else {
            showToast(
                'toast.api.bulkMoveSuccess',
                {
                    successCount: result.success_count,
                    type: this.apiConfig.config.displayName,
                },
                'success'
            );
        }

        return result.results || [];
    }

    async moveSingleModel(filePath, targetPath) {
        if (!this.apiConfig.config.supportsMove) {
            showToast('toast.api.moveNotSupported', { type: this.apiConfig.config.displayName }, 'warning');
            return null;
        }

        const recipeId = extractRecipeId(filePath);
        if (!recipeId) {
            showToast('toast.api.moveFailed', { message: 'Recipe ID missing' }, 'error');
            return null;
        }

        const response = await fetch(this.apiConfig.endpoints.move, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recipe_id: recipeId,
                target_path: targetPath,
            }),
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || `Failed to move ${this.apiConfig.config.displayName}`);
        }

        if (result.message) {
            showToast('toast.api.moveInfo', { message: result.message }, 'info');
        } else {
            showToast('toast.api.moveSuccess', { type: this.apiConfig.config.displayName }, 'success');
        }

        return {
            original_file_path: result.original_file_path || filePath,
            new_file_path: result.new_file_path || filePath,
            folder: result.folder || '',
            message: result.message,
        };
    }

    async repairBulkModels(filePaths) {
        if (!filePaths || filePaths.length === 0) {
            throw new Error('No file paths provided');
        }

        const recipeIds = filePaths
            .map((path) => extractRecipeId(path))
            .filter((id) => !!id);

        if (recipeIds.length === 0) {
            throw new Error('No recipe IDs could be derived from file paths');
        }

        const response = await fetch(this.apiConfig.endpoints.repairBulk, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recipe_ids: recipeIds,
            }),
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Failed to repair recipes');
        }

        return result;
    }

    async bulkDeleteModels(filePaths) {
        if (!filePaths || filePaths.length === 0) {
            throw new Error('No file paths provided');
        }

        const recipeIds = filePaths
            .map((path) => extractRecipeId(path))
            .filter((id) => !!id);

        if (recipeIds.length === 0) {
            throw new Error('No recipe IDs could be derived from file paths');
        }

        try {
            state.loadingManager?.showSimpleLoading('Deleting recipes...');

            const response = await fetch(this.apiConfig.endpoints.bulkDelete, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    recipe_ids: recipeIds,
                }),
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to delete recipes');
            }

            return {
                success: true,
                deleted_count: result.total_deleted,
                failed_count: result.total_failed || 0,
                errors: result.failed || [],
            };
        } finally {
            state.loadingManager?.hide();
        }
    }
}
