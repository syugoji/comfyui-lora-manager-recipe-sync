// Recipe Modal Component
import { showToast, copyToClipboard, sendLoraToWorkflow, sendModelPathToWorkflow, openCivitaiByMetadata, openCivitaiUrl, stripLoraTags, sendPromptToWorkflow, sendGenParamsToWorkflow } from '../utils/uiHelpers.js';
import { buildCivArchiveModelUrl } from '../utils/civitaiUtils.js';
import { translate } from '../utils/i18nHelpers.js';
import { state } from '../state/index.js';
import { setSessionItem, removeSessionItem, getStorageItem, setStorageItem } from '../utils/storageHelpers.js';
import {
    adoptRecipeRevision,
    createRecipePromptDraft,
    fetchRecipeDetails,
    releaseRecipePromptModel,
    updateRecipeMetadata,
} from '../api/recipeApi.js';
import { downloadManager } from '../managers/DownloadManager.js';
import { MODEL_TYPES } from '../api/apiConfig.js';
import { openMediaViewer } from './shared/MediaViewer.js';
import { renderCompactTags, setupTagTooltip } from './shared/utils.js';
import { setupTagEditMode } from './shared/ModelTags.js';
import { recipeWorkflowReplayManager } from '../managers/RecipeWorkflowReplayManager.js';
import { recipeTrialManager } from '../managers/RecipeTrialManager.js';
import { getRecipeFileParams } from '../managers/BulkMissingLoraDownloadManager.js';
import { buildRecipeReferenceText, PARAM_DISPLAY_NAMES } from '../utils/recipeReferenceInfo.js';

const ALLOWED_GEN_PARAM_KEYS = new Set([
    'prompt',
    'negative_prompt',
    'steps',
    'sampler',
    'scheduler',
    'cfg_scale',
    'seed',
    'size',
    'clip_skip',
    'denoising_strength',
    'model',
    'vae',
    'hires_upscale',
    'hires_resize',
    'hires_steps',
    'hires_upscaler',
    'hires_cfg_scale',
]);

const AI_PROMPT_MODELS = {
    'qwen35-q6': 'Qwen3.5 9B Q6_K',
    'qwythos-q5': 'Qwythos 9B Q5_K_M',
};
const DEFAULT_AI_PROMPT_MODEL = 'qwen35-q6';
const AI_PROMPT_MODEL_STORAGE_KEY = 'recipe_ai_prompt_model';

const GEN_PARAM_NORMALIZATION = {
    cfg: 'cfg_scale',
    cfgScale: 'cfg_scale',
    clipSkip: 'clip_skip',
    negativePrompt: 'negative_prompt',
    Sampler: 'sampler',
    sampler_name: 'sampler',
    scheduler: 'scheduler',
    'Schedule type': 'scheduler',
    Steps: 'steps',
    Seed: 'seed',
    Size: 'size',
    Prompt: 'prompt',
    'Negative prompt': 'negative_prompt',
    'Cfg scale': 'cfg_scale',
    'Clip skip': 'clip_skip',
    'Denoising strength': 'denoising_strength',
    Model: 'model',
    VAE: 'vae',
    'Hires upscale': 'hires_upscale',
    'Hires resize': 'hires_resize',
    'Hires steps': 'hires_steps',
    'Hires upscaler': 'hires_upscaler',
    'Hires CFG Scale': 'hires_cfg_scale',
};

class RecipeModal {
    constructor() {
        this.promptEditorState = {};
        this.recipeHydrationRequestId = 0;
        this.promptDraftRequestId = 0;
        this.promptTrialRequestId = 0;
        this.promptRevisionRequestId = 0;
        this.currentPromptDraft = null;
        this.currentPromptTrialJob = null;
        this.adoptingCandidateId = null;
        this.adoptedCandidateId = null;
        this.resetLocalEditState();
        this.init();
    }

    createLocalEditState() {
        return {
            title: { commitVersion: 0, isDirty: false },
            tags: { commitVersion: 0, isDirty: false },
            notes: { commitVersion: 0, isDirty: false },
            prompt: { commitVersion: 0, isDirty: false },
            negative_prompt: { commitVersion: 0, isDirty: false },
            source_path: { commitVersion: 0, isDirty: false },
        };
    }

    resetLocalEditState() {
        this.localEditState = this.createLocalEditState();
        this.sourceUrlEditState = this.localEditState.source_path;
    }

    getLocalEditState(field) {
        if (!this.localEditState[field]) {
            this.localEditState[field] = { commitVersion: 0, isDirty: false };
        }
        return this.localEditState[field];
    }

    markFieldDirty(field) {
        this.getLocalEditState(field).isDirty = true;
    }

    clearFieldDirty(field) {
        this.getLocalEditState(field).isDirty = false;
    }

    commitField(field) {
        const fieldState = this.getLocalEditState(field);
        fieldState.isDirty = false;
        fieldState.commitVersion += 1;
    }

    captureLocalEditVersions() {
        return Object.fromEntries(
            Object.entries(this.localEditState).map(([field, state]) => [
                field,
                state.commitVersion,
            ])
        );
    }

    shouldPreserveField(field, requestVersions) {
        const fieldState = this.getLocalEditState(field);
        const requestVersion = requestVersions?.[field] ?? fieldState.commitVersion;
        return fieldState.isDirty || fieldState.commitVersion !== requestVersion;
    }

    hasFieldCommittedSinceRequest(field, requestVersions) {
        const fieldState = this.getLocalEditState(field);
        const requestVersion = requestVersions?.[field] ?? fieldState.commitVersion;
        return fieldState.commitVersion !== requestVersion;
    }

    init() {
        this.setupCopyButtons();
        this.setupStripLoraToggle();
        this.setupPromptEditors();
        this.setupNotesEditor();
        // Set up tooltip positioning handlers after DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            this.setupTooltipPositioning();
        });

        // Set up document click handler to close edit fields
        document.addEventListener('click', (event) => {
            const recipeModal = document.getElementById('recipeModal');
            if (recipeModal && recipeModal.style.display !== 'none') {
                const mediaEl = event.target.closest('.recipe-preview-media');
                if (mediaEl && mediaEl.tagName) {
                    event.stopPropagation();
                    const isVideo = mediaEl.tagName === 'VIDEO';
                    const url = mediaEl.src || mediaEl.currentSrc;
                    if (url) {
                        openMediaViewer(url, {
                            type: isVideo ? 'video' : 'image',
                            title: document.getElementById('recipeModalTitle')?.textContent || ''
                        });
                    }
                    return;
                }
            }

            // Handle title edit
            const titleEditor = document.getElementById('recipeTitleEditor');
            if (titleEditor && titleEditor.classList.contains('active') &&
                !titleEditor.contains(event.target) &&
                !event.target.closest('.edit-icon')) {
                this.saveTitleEdit();
            }

            // Handle reconnect input
            const reconnectContainers = document.querySelectorAll('.lora-reconnect-container');
            reconnectContainers.forEach(container => {
                if (container.classList.contains('active') &&
                    !container.contains(event.target) &&
                    !event.target.closest('.deleted-badge.reconnectable')) {
                    this.hideReconnectInput(container);
                }
            });
        });
    }

    // Add tooltip positioning handler to ensure correct positioning of fixed tooltips
    setupTooltipPositioning() {
        document.addEventListener('mouseover', (event) => {
            // Check if we're hovering over a local-badge
            if (event.target.closest('.local-badge')) {
                const badge = event.target.closest('.local-badge');
                const tooltip = badge.querySelector('.local-path');

                if (tooltip) {
                    // Get badge position
                    const badgeRect = badge.getBoundingClientRect();

                    // Position the tooltip
                    tooltip.style.top = (badgeRect.bottom + 4) + 'px';
                    tooltip.style.left = (badgeRect.right - tooltip.offsetWidth) + 'px';
                }
            }

            // Add tooltip positioning for missing badge
            if (event.target.closest('.recipe-status.missing')) {
                const badge = event.target.closest('.recipe-status.missing');
                const tooltip = badge.querySelector('.missing-tooltip');

                if (tooltip) {
                    // Get badge position
                    const badgeRect = badge.getBoundingClientRect();

                    // Position the tooltip
                    tooltip.style.top = (badgeRect.bottom + 4) + 'px';
                    tooltip.style.left = (badgeRect.left) + 'px';
                }
            }
        }, true);
    }

    showRecipeDetails(recipe) {
        const hydratedRecipe = recipe || {};
        this.promptDraftRequestId += 1;
        this.promptTrialRequestId += 1;
        this.promptRevisionRequestId += 1;
        this.adoptingCandidateId = null;
        this.adoptedCandidateId = null;
        this.resetPromptDraftUI();
        this.resetPromptTrialUI();
        this.resetLocalEditState();
        // Store the full recipe for editing
        this.currentRecipe = hydratedRecipe;
        this.resetPromptEditors();

        // Set modal title with edit icon
        const modalTitle = document.getElementById('recipeModalTitle');
        if (modalTitle) {
            modalTitle.innerHTML = `
                <div class="editable-content">
                    <span class="content-text">${hydratedRecipe.title || 'Recipe Details'}</span>
                    <button class="edit-icon" title="Edit recipe name"><i class="fas fa-pencil-alt"></i></button>
                </div>
                <div id="recipeTitleEditor" class="content-editor">
                    <input type="text" class="title-input" value="${hydratedRecipe.title || ''}">
                </div>
            `;

            // Add event listener for title editing
            const editIcon = modalTitle.querySelector('.edit-icon');
            editIcon.addEventListener('click', () => this.showTitleEditor());

            // Add key event listener for Enter key
            const titleInput = modalTitle.querySelector('.title-input');
            titleInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.saveTitleEdit();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    this.cancelTitleEdit();
                }
            });
        }

        // Store the recipe ID for copy syntax API call
        this.recipeId = hydratedRecipe.id;
        this.filePath = hydratedRecipe.file_path;
        this.listFilePath = hydratedRecipe.file_path;

        // Render tags using shared utility
        const tagsContainer = document.getElementById('recipeTagsContainer');
        if (tagsContainer) {
            this.updateTagsDisplay(tagsContainer, hydratedRecipe.tags || []);
        }

        // Set recipe image
        const mediaContainer = document.getElementById('recipePreviewContainer');
        if (mediaContainer) {
            this.syncPreviewMedia(hydratedRecipe);
            mediaContainer.querySelector('.source-url-container')?.remove();
            mediaContainer.querySelector('.source-url-editor')?.remove();

            // Add source URL container if the recipe has a source_path
            const sourceUrlContainer = document.createElement('div');
            sourceUrlContainer.className = 'source-url-container';
            const hasSourceUrl = hydratedRecipe.source_path && hydratedRecipe.source_path.trim().length > 0;
            const sourceUrl = hasSourceUrl ? hydratedRecipe.source_path : '';
            const isValidUrl = hasSourceUrl && (sourceUrl.startsWith('http://') || sourceUrl.startsWith('https://'));

            sourceUrlContainer.innerHTML = `
                <div class="source-url-content">
                    <span class="source-url-icon"><i class="fas fa-link"></i></span>
                    <span class="source-url-text" title="${isValidUrl ? 'Click to open source URL' : 'No valid URL'}">${hasSourceUrl ? sourceUrl : 'No source URL'
                }</span>
                </div>
                <button class="source-url-edit-btn" title="Edit source URL">
                    <i class="fas fa-pencil-alt"></i>
                </button>
            `;

            // Add source URL editor
            const sourceUrlEditor = document.createElement('div');
            sourceUrlEditor.className = 'source-url-editor';
            sourceUrlEditor.innerHTML = `
                <input type="text" class="source-url-input" placeholder="Enter source URL (e.g., https://civitai.com/...)" value="${sourceUrl}">
                <div class="source-url-actions">
                    <button class="source-url-cancel-btn">Cancel</button>
                    <button class="source-url-save-btn">Save</button>
                </div>
            `;

            // Append both containers to the media container
            mediaContainer.appendChild(sourceUrlContainer);
            mediaContainer.appendChild(sourceUrlEditor);

            // Delay binding slightly so modal layout is stable, but skip if this render was torn down.
            const sourceUrlContainerRef = sourceUrlContainer;
            const sourceUrlEditorRef = sourceUrlEditor;
            setTimeout(() => {
                if (!document.body.contains(sourceUrlContainerRef) || !document.body.contains(sourceUrlEditorRef)) {
                    return;
                }
                this.setupSourceUrlHandlers();
            }, 50);
        }

        this.syncGenerationParams(hydratedRecipe.gen_params);
        this.syncNotesField(hydratedRecipe.notes || '');
        this.syncResourcesSection(hydratedRecipe);
        this.syncSourceUrlAction();
        this.renderRecipeRevisionBanner(hydratedRecipe);

        // Show the modal
        modalManager.showModal('recipeModal');

        if (this.recipeId) {
            const hydrationRequestId = ++this.recipeHydrationRequestId;
            const requestEditVersions = this.captureLocalEditVersions();
            this.hydrateRecipeDetails(
                this.recipeId,
                hydrationRequestId,
                requestEditVersions
            );
        }
    }

    async hydrateRecipeDetails(recipeId, requestId, requestEditVersions = {}) {
        try {
            const fullRecipe = await fetchRecipeDetails(recipeId);
            if (requestId !== this.recipeHydrationRequestId || !fullRecipe) {
                return;
            }

            let activeRecipe = null;
            if (fullRecipe.revision_summary?.active === true
                && fullRecipe.revision_summary?.stale !== true) {
                try {
                    activeRecipe = await fetchRecipeDetails(recipeId, { variant: 'active' });
                } catch (error) {
                    console.warn('Failed to hydrate active recipe revision:', error);
                }
                if (requestId !== this.recipeHydrationRequestId) return;
            }

            const nextRecipe = { ...this.currentRecipe };

            if (!this.hasFieldCommittedSinceRequest('title', requestEditVersions) && fullRecipe.title !== undefined) {
                nextRecipe.title = fullRecipe.title;
            }

            if (!this.hasFieldCommittedSinceRequest('tags', requestEditVersions) && fullRecipe.tags !== undefined) {
                nextRecipe.tags = Array.isArray(fullRecipe.tags) ? [...fullRecipe.tags] : fullRecipe.tags;
            }

            if (!this.hasFieldCommittedSinceRequest('notes', requestEditVersions) && fullRecipe.notes !== undefined) {
                nextRecipe.notes = fullRecipe.notes || '';
            }

            if (!this.hasFieldCommittedSinceRequest('source_path', requestEditVersions)) {
                nextRecipe.source_path = fullRecipe.source_path || '';
            }

            const previousFilePath = nextRecipe.file_path;
            if (fullRecipe.file_path !== undefined) {
                nextRecipe.file_path = fullRecipe.file_path;
            }
            if (fullRecipe.file_url !== undefined) {
                nextRecipe.file_url = fullRecipe.file_url;
            }
            if (fullRecipe.preview_url !== undefined) {
                nextRecipe.preview_url = fullRecipe.preview_url;
            }
            if (
                fullRecipe.file_path !== undefined &&
                fullRecipe.file_path !== previousFilePath &&
                fullRecipe.file_url === undefined &&
                fullRecipe.preview_url === undefined
            ) {
                delete nextRecipe.file_url;
                delete nextRecipe.preview_url;
            }

            if (fullRecipe.gen_params !== undefined) {
                const previousGenParams = nextRecipe.gen_params || {};
                const incomingGenParams = { ...(fullRecipe.gen_params || {}) };
                for (const [key, value] of Object.entries(previousGenParams)) {
                    if (this.hasFieldCommittedSinceRequest(key, requestEditVersions)) {
                        incomingGenParams[key] = value;
                    }
                }
                nextRecipe.gen_params = incomingGenParams;
            } else {
                const previousGenParams = nextRecipe.gen_params || {};
                const preservedGenParams = {};
                for (const [key, value] of Object.entries(previousGenParams)) {
                    if (this.hasFieldCommittedSinceRequest(key, requestEditVersions)) {
                        preservedGenParams[key] = value;
                    }
                }
                nextRecipe.gen_params = preservedGenParams;
            }

            if (fullRecipe.checkpoint !== undefined) {
                nextRecipe.checkpoint = fullRecipe.checkpoint;
            } else {
                delete nextRecipe.checkpoint;
            }
            if (fullRecipe.loras !== undefined) {
                nextRecipe.loras = Array.isArray(fullRecipe.loras) ? [...fullRecipe.loras] : fullRecipe.loras;
            } else {
                delete nextRecipe.loras;
            }
            if (fullRecipe.replay_manifest !== undefined) {
                nextRecipe.replay_manifest = fullRecipe.replay_manifest;
            } else {
                delete nextRecipe.replay_manifest;
            }
            if (fullRecipe.source_etag !== undefined) {
                nextRecipe.source_etag = fullRecipe.source_etag;
            }
            if (fullRecipe.revision_summary !== undefined) {
                nextRecipe.revision_summary = fullRecipe.revision_summary;
            } else {
                delete nextRecipe.revision_summary;
            }
            if (activeRecipe) {
                nextRecipe.active_revision = {
                    gen_params: { ...(activeRecipe.gen_params || {}) },
                    revision_summary: activeRecipe.revision_summary,
                };
            } else {
                delete nextRecipe.active_revision;
            }

            this.currentRecipe = nextRecipe;
            this.filePath = this.currentRecipe.file_path || this.filePath;

            this.syncHydratedRecipeFields(requestEditVersions);
            this.recoverPromptTrials();
        } catch (error) {
            // Keep the cached recipe visible if hydration fails.
            console.warn('Failed to hydrate recipe details:', error);
        }
    }

    syncHydratedRecipeFields(requestEditVersions = {}) {
        this.syncPreviewMedia(this.currentRecipe);

        if (!this.shouldPreserveField('title', requestEditVersions)) {
            this.syncTitleDisplay(this.currentRecipe?.title || '');
        }

        if (!this.shouldPreserveField('tags', requestEditVersions)) {
            this.syncTagsDisplay(this.currentRecipe?.tags || []);
        }

        if (!this.shouldPreserveField('notes', requestEditVersions)) {
            this.syncNotesField(this.currentRecipe?.notes || '');
        }

        if (!this.shouldPreserveField('prompt', requestEditVersions)) {
            this.syncPromptField(
                'prompt',
                this.currentRecipe?.gen_params?.prompt || '',
                'No prompt information available'
            );
        }

        if (!this.shouldPreserveField('negative_prompt', requestEditVersions)) {
            this.syncPromptField(
                'negative_prompt',
                this.currentRecipe?.gen_params?.negative_prompt || '',
                'No negative prompt information available'
            );
        }

        this.syncGenerationParams(this.currentRecipe?.gen_params, { promptFieldsOnly: true });
        this.syncResourcesSection(this.currentRecipe);
        this.renderRecipeRevisionBanner(this.currentRecipe);

        if (!this.shouldPreserveField('source_path', requestEditVersions)) {
            this.updateSourceUrlDisplay(this.currentRecipe.source_path || '', { forceInputSync: true });
        } else {
            this.updateSourceUrlDisplay(this.currentRecipe.source_path || '');
        }
        this.syncSourceUrlAction();
    }

    renderRecipeRevisionBanner(recipe = this.currentRecipe) {
        const banner = document.getElementById('recipeRevisionBanner');
        const meta = document.getElementById('recipeRevisionMeta');
        const prompt = document.getElementById('recipeRevisionPrompt');
        if (!banner) return;
        const summary = recipe?.revision_summary || {};
        if (summary.active !== true) {
            banner.hidden = true;
            banner.classList.remove('stale');
            if (meta) meta.textContent = '';
            if (prompt) prompt.textContent = '';
            return;
        }

        banner.hidden = false;
        banner.classList.toggle('stale', summary.stale === true);
        const details = [];
        if (summary.seed !== undefined && summary.seed !== null) {
            details.push(`seed ${summary.seed}`);
        }
        if (summary.model) details.push(String(summary.model));
        if (summary.created_at) details.push(String(summary.created_at));
        if (meta) meta.textContent = details.join(' / ');

        if (!prompt) return;
        if (summary.stale === true) {
            prompt.textContent = '元レシピの生成情報が変わったため、この改変版は再生成が必要です。';
            return;
        }
        const activePrompt = recipe?.active_revision?.gen_params?.prompt;
        prompt.textContent = activePrompt
            ? `採用中のprompt: ${activePrompt}`
            : '採用中のpromptとseedは「再現」で自動適用されます。';
    }

    getPreviewMediaUrl(recipe = {}) {
        return recipe.file_url ||
            recipe.preview_url ||
            (recipe.file_path ? `/loras_static/root1/preview/${recipe.file_path.split('/').pop()}` :
                '/loras_static/images/no-preview.png');
    }

    syncPreviewMedia(recipe = {}) {
        const mediaContainer = document.getElementById('recipePreviewContainer');
        if (!mediaContainer) {
            return;
        }

        const previewUrl = this.getPreviewMediaUrl(recipe);
        const isVideo = previewUrl.toLowerCase().endsWith('.mp4');
        const expectedElementId = isVideo ? 'recipeModalVideo' : 'recipeModalImage';
        let previewElement = mediaContainer.querySelector(`#${expectedElementId}`);
        const existingPreviewElement = mediaContainer.querySelector('.recipe-preview-media');

        if (!previewElement || (existingPreviewElement && existingPreviewElement !== previewElement)) {
            if (existingPreviewElement?.tagName === 'VIDEO') {
                const existingVideo = existingPreviewElement;
                existingVideo.pause();
                existingVideo.currentTime = 0;
            }

            existingPreviewElement?.remove();
            previewElement = document.createElement(isVideo ? 'video' : 'img');
            previewElement.id = expectedElementId;
            previewElement.className = 'recipe-preview-media';
            mediaContainer.prepend(previewElement);
        }

        previewElement.src = previewUrl;
        previewElement.alt = recipe.title || 'Recipe Preview';

        if (isVideo) {
            previewElement.controls = true;
            previewElement.autoplay = false;
            previewElement.loop = true;
            previewElement.muted = true;
        }
    }

    getMetadataUpdateOptions() {
        return this.listFilePath ? { listFilePath: this.listFilePath } : {};
    }

    syncTitleDisplay(title) {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (!titleContainer) {
            return;
        }

        const contentText = titleContainer.querySelector('.content-text');
        if (contentText) {
            contentText.textContent = title || 'Recipe Details';
        }

        const titleInput = titleContainer.querySelector('.title-input');
        if (titleInput) {
            titleInput.value = title || '';
        }
    }

    syncSourceUrlAction() {
        const actionsContainer = document.getElementById('recipeHeaderActions');
        if (!actionsContainer) {
            return;
        }

        actionsContainer.innerHTML = '';

        const sourcePath = this.currentRecipe?.source_path || '';
        const isValidUrl = sourcePath.startsWith('http://') || sourcePath.startsWith('https://');
        if (!isValidUrl) {
            return;
        }

        const btn = document.createElement('button');
        btn.className = 'recipe-source-url-btn';
        btn.title = sourcePath;
        btn.innerHTML = '<i class="fas fa-globe"></i> Open Source URL';
        btn.addEventListener('click', () => {
            window.open(sourcePath, '_blank');
        });
        actionsContainer.appendChild(btn);
    }

    syncTagsDisplay(tags) {
        const container = document.getElementById('recipeTagsContainer');
        if (!container) return;
        this.updateTagsDisplay(container, tags || []);
    }

    // Re-render tags display using shared utility, wire edit mode with ModelTags
    updateTagsDisplay(container, tags) {
        const filePath = this.filePath || '';

        container.innerHTML = renderCompactTags(tags, filePath);

        // Setup tooltip for all tags
        setupTagTooltip(container);

        // Wire edit button using shared tag editing (no suggestions for recipes)
        setupTagEditMode(null, {
            container: container,
            showSuggestions: false,
            normalizeTag: false,
            saveHandler: async (filePath, tags) => {
                await updateRecipeMetadata(filePath, { tags }, this.getMetadataUpdateOptions());
            },
            onSaved: (tags) => {
                this.currentRecipe.tags = tags;
                this.commitField('tags');
                const c = document.getElementById('recipeTagsContainer');
                if (c) this.updateTagsDisplay(c, tags);
            },
        });
    }

    syncPromptField(field, value, placeholder) {
        const contentId = field === 'prompt' ? 'recipePrompt' : 'recipeNegativePrompt';
        const editorId = field === 'prompt' ? 'recipePromptEditor' : 'recipeNegativePromptEditor';
        const inputId = field === 'prompt' ? 'recipePromptInput' : 'recipeNegativePromptInput';

        this.renderPromptContent(document.getElementById(contentId), value, placeholder);

        const input = document.getElementById(inputId);
        if (input) {
            input.value = value || '';
        }
    }

    syncGenerationParams(genParams, options = {}) {
        const promptElement = document.getElementById('recipePrompt');
        const negativePromptElement = document.getElementById('recipeNegativePrompt');
        const otherParamsElement = document.getElementById('recipeOtherParams');
        const promptInput = document.getElementById('recipePromptInput');
        const negativePromptInput = document.getElementById('recipeNegativePromptInput');
        const promptFieldsOnly = options.promptFieldsOnly === true;
        const sanitizedGenParams = this.sanitizeGenParams(genParams);

        if (sanitizedGenParams) {
            if (!promptFieldsOnly) {
                this.renderPromptContent(promptElement, sanitizedGenParams.prompt, 'No prompt information available');
                this.renderPromptContent(negativePromptElement, sanitizedGenParams.negative_prompt, 'No negative prompt information available');

                if (promptInput) {
                    promptInput.value = sanitizedGenParams.prompt || '';
                }

                if (negativePromptInput) {
                    negativePromptInput.value = sanitizedGenParams.negative_prompt || '';
                }
            }

            if (otherParamsElement) {
                otherParamsElement.innerHTML = '';
                const excludedParams = ['prompt', 'negative_prompt'];

                for (const [key, value] of Object.entries(sanitizedGenParams)) {
                    if (!excludedParams.includes(key) && value !== undefined && value !== null) {
                        const displayName = PARAM_DISPLAY_NAMES[key] || key;
                        const paramTag = document.createElement('div');
                        paramTag.className = 'param-tag';
                        paramTag.innerHTML = `
                            <span class="param-name">${displayName}:</span>
                            <span class="param-value">${value}</span>
                        `;
                        otherParamsElement.appendChild(paramTag);
                    }
                }

                if (otherParamsElement.children.length === 0) {
                    otherParamsElement.innerHTML = '<div class="no-params">No additional parameters available</div>';
                }
            }
            return;
        }

        if (!promptFieldsOnly) {
            this.renderPromptContent(promptElement, '', 'No prompt information available');
            this.renderPromptContent(negativePromptElement, '', 'No negative prompt information available');
            if (promptInput) promptInput.value = '';
            if (negativePromptInput) negativePromptInput.value = '';
        }

        if (otherParamsElement) {
            otherParamsElement.innerHTML = '<div class="no-params">No parameters available</div>';
        }
    }

    sanitizeGenParams(genParams) {
        if (!genParams || typeof genParams !== 'object') {
            return null;
        }

        const sanitized = {};

        for (const [key, value] of Object.entries(genParams)) {
            if (value === undefined || value === null || value === '') {
                continue;
            }

            if (!ALLOWED_GEN_PARAM_KEYS.has(key)) {
                continue;
            }

            sanitized[key] = value;
        }

        for (const [key, value] of Object.entries(genParams)) {
            if (value === undefined || value === null || value === '') {
                continue;
            }

            const normalizedKey = GEN_PARAM_NORMALIZATION[key] || key;
            if (!ALLOWED_GEN_PARAM_KEYS.has(normalizedKey)) {
                continue;
            }

            if (sanitized[normalizedKey] === undefined || sanitized[normalizedKey] === null || sanitized[normalizedKey] === '') {
                sanitized[normalizedKey] = value;
            }
        }

        return sanitized;
    }

    syncResourcesSection(recipe = {}) {
        const checkpointContainer = document.getElementById('recipeCheckpoint');
        const resourceDivider = document.getElementById('recipeResourceDivider');
        const lorasListElement = document.getElementById('recipeLorasList');
        const lorasCountElement = document.getElementById('recipeLorasCount');
        const embeddingDivider = document.getElementById('recipeEmbeddingDivider');
        const embeddingsHeader = document.getElementById('recipeEmbeddingsHeader');
        const embeddingsList = document.getElementById('recipeEmbeddingsList');
        const loras = Array.isArray(recipe.loras) ? recipe.loras : [];
        const embeddings = Array.isArray(recipe.embeddings) ? recipe.embeddings : [];

        if (checkpointContainer) {
            checkpointContainer.innerHTML = '';
            if (recipe.checkpoint && typeof recipe.checkpoint === 'object') {
                checkpointContainer.innerHTML = this.renderCheckpoint(recipe.checkpoint);
                this.setupCheckpointActions(checkpointContainer, recipe.checkpoint);
                this.setupCheckpointNavigation(checkpointContainer, recipe.checkpoint);
            }
        }

        let allLorasAvailable = true;
        let missingLorasCount = 0;
        let deletedLorasCount = 0;

        loras.forEach(lora => {
            if (lora.isDeleted) {
                deletedLorasCount++;
            } else if (!lora.inLibrary) {
                allLorasAvailable = false;
                missingLorasCount++;
            }
        });

        if (lorasCountElement) {
            const totalCount = loras.length;
            let statusHTML = '';
            if (totalCount > 0) {
                if (allLorasAvailable && deletedLorasCount === 0) {
                    statusHTML = `<div class="recipe-status ready"><i class="fas fa-check-circle"></i> Ready to use</div>`;
                } else if (missingLorasCount > 0) {
                    statusHTML = `<div class="recipe-status missing">
                        <i class="fas fa-exclamation-triangle"></i> ${missingLorasCount} missing
                        <div class="missing-tooltip">Click to download missing LoRAs</div>
                    </div>`;
                } else if (deletedLorasCount > 0 && missingLorasCount === 0) {
                    statusHTML = `<div class="recipe-status partial"><i class="fas fa-info-circle"></i> ${deletedLorasCount} deleted</div>`;
                }
            }

            lorasCountElement.innerHTML = `<i class="fas fa-layer-group"></i> ${totalCount} LoRAs ${statusHTML}`;

            setTimeout(() => {
                const viewRecipeLorasBtn = document.getElementById('viewRecipeLorasBtn');
                if (viewRecipeLorasBtn) {
                    viewRecipeLorasBtn.addEventListener('click', () => this.navigateToLorasPage());
                }

                const missingStatus = document.querySelector('.recipe-status.missing');
                if (missingStatus && missingLorasCount > 0) {
                    missingStatus.classList.add('clickable');
                    missingStatus.addEventListener('click', () => this.showDownloadMissingLorasModal());
                }
            }, 100);
        }

        if (lorasListElement && loras.length > 0) {
            lorasListElement.innerHTML = loras.map((lora, loraIndex) => {
                const existsLocally = lora.inLibrary;
                const isDeleted = lora.isDeleted;
                const localPath = lora.localPath || '';
                const replayRequirement = recipe.replay_manifest?.required_resources?.find(
                    item => item?.required === true && item?.requirement_id === `recipe:${loraIndex}`
                );
                const replayStrength = Number(replayRequirement?.expected?.strength_model);
                const savedStrength = Number(lora.strength);
                const displayedStrength = Number.isFinite(replayStrength)
                    ? replayStrength
                    : (Number.isFinite(savedStrength) ? savedStrength : 1.0);
                const strengthLabel = Number.isFinite(replayStrength) ? 'Replay weight' : 'Weight';

                let localStatus;
                if (existsLocally) {
                    localStatus = `
                        <div class="local-badge">
                            <i class="fas fa-check"></i> In Library
                            <div class="local-path">${localPath}</div>
                        </div>`;
                } else if (isDeleted) {
                    localStatus = `
                        <div class="deleted-badge reconnectable" data-lora-index="${loraIndex}">
                            <span class="badge-text"><i class="fas fa-trash-alt"></i> Deleted</span>
                            <div class="reconnect-tooltip">Click to reconnect with a local LoRA</div>
                        </div>`;
                } else if (this.canDownloadLora(lora)) {
                    localStatus = `
                        <button type="button" class="missing-badge lora-download" data-lora-index="${loraIndex}" title="Download with LoRA Manager">
                            <i class="fas fa-download"></i> Download
                        </button>`;
                } else if (lora.unresolved) {
                    localStatus = `
                        <div class="missing-badge unresolved" title="No exact Civitai version is currently resolved">
                            <i class="fas fa-search"></i> Civitai ID unresolved
                        </div>`;
                } else {
                    localStatus = `
                        <div class="missing-badge">
                            <i class="fas fa-exclamation-triangle"></i> Not in Library
                        </div>`;
                }

                const isPreviewVideo = lora.preview_url && lora.preview_url.toLowerCase().endsWith('.mp4');
                const previewMedia = isPreviewVideo ?
                    `<video class="thumbnail-video" autoplay loop muted playsinline>
                        <source src="${lora.preview_url}" type="video/mp4">
                     </video>` :
                    `<img src="${lora.preview_url || '/loras_static/images/no-preview.png'}" alt="LoRA preview">`;

                let loraItemClass = 'recipe-lora-item';
                if (existsLocally) {
                    loraItemClass += ' exists-locally';
                } else if (isDeleted) {
                    loraItemClass += ' is-deleted';
                } else {
                    loraItemClass += ' missing-locally';
                }

                return `
                    <div class="${loraItemClass}" data-lora-index="${loraIndex}">
                        <div class="recipe-lora-thumbnail">
                            ${previewMedia}
                        </div>
                        <div class="recipe-lora-content">
                            <div class="recipe-lora-header">
                                <h4>${lora.modelName}</h4>
                                <div class="badge-container">${localStatus}</div>
                            </div>
                            <div class="recipe-lora-info">
                                ${lora.modelVersionName ? `<div class="recipe-lora-version">${lora.modelVersionName}</div>` : ''}
                                <div class="recipe-lora-weight">${strengthLabel}: ${displayedStrength}</div>
                                ${lora.baseModel ? `<div class="base-model">${lora.baseModel}</div>` : ''}
                            </div>
                            <div class="lora-reconnect-container" data-lora-index="${loraIndex}">
                                <div class="reconnect-instructions">
                                    <p>Enter LoRA Syntax or Name to Reconnect:</p>
                                    <small>Example: <code>&lt;lora:Boris_Vallejo_BV_flux_D:1&gt;</code> or just <code>Boris_Vallejo_BV_flux_D</code></small>
                                </div>
                                <div class="reconnect-form">
                                    <input type="text" class="reconnect-input" placeholder="Enter LoRA name or syntax">
                                    <div class="reconnect-actions">
                                        <button class="reconnect-cancel-btn">Cancel</button>
                                        <button class="reconnect-confirm-btn">Reconnect</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

            setTimeout(() => {
                this.setupReconnectButtons();
                this.setupLoraDownloadActions();
                this.setupLoraItemsClickable();
            }, 100);

            this.recipeLorasSyntax = '';
        } else if (lorasListElement) {
            lorasListElement.innerHTML = '<div class="no-loras">No LoRAs associated with this recipe</div>';
            this.recipeLorasSyntax = '';
        }

        if (embeddingsHeader) {
            embeddingsHeader.style.display = embeddings.length ? 'block' : 'none';
            const missingEmbeddings = embeddings.filter(
                embedding => !embedding.inLibrary && !embedding.isDeleted
            ).length;
            embeddingsHeader.innerHTML = embeddings.length
                ? `<h4><i class="fas fa-font"></i> ${embeddings.length} Embedding(s)${missingEmbeddings ? ` · ${missingEmbeddings} missing` : ''}</h4>`
                : '';
        }
        if (embeddingDivider) {
            embeddingDivider.style.display = embeddings.length ? 'block' : 'none';
        }
        if (embeddingsList) {
            embeddingsList.innerHTML = embeddings.map((embedding, embeddingIndex) => {
                const existsLocally = !!embedding.inLibrary;
                const isDeleted = !!embedding.isDeleted;
                const embeddingName = embedding.modelName || embedding.name || embedding.file_name || 'Embedding';
                const versionName = embedding.modelVersionName || embedding.version || '';
                const previewUrl = embedding.preview_url || embedding.thumbnailUrl || '/loras_static/images/no-preview.png';
                const status = existsLocally
                    ? `<div class="local-badge"><i class="fas fa-check"></i> In Library</div>`
                    : isDeleted
                        ? `<div class="deleted-badge"><i class="fas fa-trash-alt"></i> Deleted</div>`
                        : this.canDownloadEmbedding(embedding)
                            ? `<button type="button" class="missing-badge embedding-download" data-embedding-index="${embeddingIndex}"><i class="fas fa-download"></i> Download</button>`
                            : `<div class="missing-badge"><i class="fas fa-exclamation-triangle"></i> Not in Library</div>`;
                return `
                    <div class="recipe-lora-item embedding-item ${existsLocally ? 'exists-locally' : 'missing-locally'}">
                        <div class="recipe-lora-thumbnail"><img src="${previewUrl}" alt="Embedding preview"></div>
                        <div class="recipe-lora-content">
                            <div class="recipe-lora-header"><h4>${embeddingName}</h4><div class="badge-container">${status}</div></div>
                            <div class="recipe-lora-info">
                                ${versionName ? `<div class="recipe-lora-version">${versionName}</div>` : ''}
                                ${embedding.baseModel ? `<div class="base-model">${embedding.baseModel}</div>` : ''}
                            </div>
                        </div>
                    </div>`;
            }).join('');
            this.setupEmbeddingDownloadActions();
        }

        if (resourceDivider) {
            const hasCheckpoint = checkpointContainer && checkpointContainer.querySelector('.recipe-lora-item');
            const hasLoraItems = lorasListElement && lorasListElement.querySelector('.recipe-lora-item');
            resourceDivider.style.display = hasCheckpoint && hasLoraItems ? 'block' : 'none';
        }
    }

    updateSourceUrlDisplay(sourcePath, options = {}) {
        const sourceUrlContainer = document.querySelector('.source-url-container');
        const sourceUrlEditor = document.querySelector('.source-url-editor');
        if (!sourceUrlContainer || !sourceUrlEditor) {
            return;
        }

        const sourceUrlText = sourceUrlContainer.querySelector('.source-url-text');
        const sourceUrlInput = sourceUrlEditor.querySelector('.source-url-input');
        if (!sourceUrlText || !sourceUrlInput) {
            return;
        }

        const normalizedSourcePath = typeof sourcePath === 'string' ? sourcePath.trim() : '';
        const isValidUrl = normalizedSourcePath.startsWith('http://') || normalizedSourcePath.startsWith('https://');

        sourceUrlText.textContent = normalizedSourcePath || 'No source URL';
        sourceUrlText.title = normalizedSourcePath
            ? (isValidUrl ? 'Click to open source URL' : 'No valid URL')
            : 'No valid URL';
        if (options.forceInputSync || !sourceUrlEditor.classList.contains('active') || !this.sourceUrlEditState.isDirty) {
            sourceUrlInput.value = normalizedSourcePath;
        }
    }

    // Title editing methods
    showTitleEditor() {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (titleContainer) {
            titleContainer.querySelector('.editable-content').classList.add('hide');
            const editor = titleContainer.querySelector('#recipeTitleEditor');
            editor.classList.add('active');
            const input = editor.querySelector('input');
            input.oninput = () => this.markFieldDirty('title');
            input.focus();
            input.select();
        }
    }

    saveTitleEdit() {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (titleContainer) {
            const editor = titleContainer.querySelector('#recipeTitleEditor');
            const input = editor.querySelector('input');
            const newTitle = input.value.trim();

            // Check if title changed
            if (newTitle && newTitle !== this.currentRecipe.title) {
                // Update title in the UI
                titleContainer.querySelector('.content-text').textContent = newTitle;

                // Update the recipe on the server
                updateRecipeMetadata(this.filePath, { title: newTitle }, this.getMetadataUpdateOptions())
                    .then(data => {
                        // Show success toast
                        showToast('toast.recipes.nameUpdated', {}, 'success');

                        // Update the current recipe object
                        this.currentRecipe.title = newTitle;
                        this.commitField('title');
                    })
                    .catch(error => {
                        // Error is handled in the API function
                        // Reset the UI if needed
                        titleContainer.querySelector('.content-text').textContent = this.currentRecipe.title || '';
                        this.clearFieldDirty('title');
                    });
            } else {
                this.clearFieldDirty('title');
            }

            // Hide editor
            editor.classList.remove('active');
            titleContainer.querySelector('.editable-content').classList.remove('hide');
        }
    }

    cancelTitleEdit() {
        const titleContainer = document.getElementById('recipeModalTitle');
        if (titleContainer) {
            // Reset input value
            const editor = titleContainer.querySelector('#recipeTitleEditor');
            const input = editor.querySelector('input');
            input.value = this.currentRecipe.title || '';
            this.clearFieldDirty('title');

            // Hide editor
            editor.classList.remove('active');
            titleContainer.querySelector('.editable-content').classList.remove('hide');
        }
    }

    setupNotesEditor() {
        const input = document.getElementById('recipeNotesInput');
        const saveButton = document.getElementById('saveRecipeNotesBtn');
        if (!input || !saveButton) return;

        input.addEventListener('input', () => this.markFieldDirty('notes'));
        input.addEventListener('keydown', event => {
            if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                this.saveNotes();
            }
        });
        saveButton.addEventListener('click', () => this.saveNotes());
    }

    syncNotesField(notes) {
        const input = document.getElementById('recipeNotesInput');
        if (input) input.value = String(notes || '');
    }

    async saveNotes() {
        const input = document.getElementById('recipeNotesInput');
        if (!input || !this.currentRecipe || !this.filePath) return;
        const notes = input.value.trim();
        const previousNotes = String(this.currentRecipe.notes || '');
        if (notes === previousNotes) {
            this.clearFieldDirty('notes');
            return;
        }

        try {
            await updateRecipeMetadata(
                this.filePath,
                { notes },
                this.getMetadataUpdateOptions()
            );
            this.currentRecipe.notes = notes;
            this.commitField('notes');
            showToast('toast.recipes.notesUpdated', {}, 'success', 'メモを保存しました');
        } catch (error) {
            input.value = previousNotes;
            this.clearFieldDirty('notes');
        }
    }

    setupPromptEditors() {
        const promptConfigs = [
            {
                editButtonId: 'editPromptBtn',
                contentId: 'recipePrompt',
                editorId: 'recipePromptEditor',
                inputId: 'recipePromptInput',
                field: 'prompt',
                placeholder: 'No prompt information available',
                successKey: 'toast.recipes.promptUpdated',
                successFallback: 'Prompt updated successfully',
            },
            {
                editButtonId: 'editNegativePromptBtn',
                contentId: 'recipeNegativePrompt',
                editorId: 'recipeNegativePromptEditor',
                inputId: 'recipeNegativePromptInput',
                field: 'negative_prompt',
                placeholder: 'No negative prompt information available',
                successKey: 'toast.recipes.negativePromptUpdated',
                successFallback: 'Negative prompt updated successfully',
            }
        ];

        promptConfigs.forEach((config) => {
            const editButton = document.getElementById(config.editButtonId);
            const input = document.getElementById(config.inputId);

            if (editButton) {
                editButton.addEventListener('click', () => this.showPromptEditor(config));
            }

            if (input) {
                input.addEventListener('input', () => this.markFieldDirty(config.field));
                input.addEventListener('keydown', (event) => {
                    if (event.key === 'Escape') {
                        event.preventDefault();
                        event.stopPropagation();
                        this.cancelPromptEdit(config);
                        return;
                    }

                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        event.stopPropagation();
                        this.promptEditorState[config.field] = {
                            ...(this.promptEditorState[config.field] || {}),
                            skipBlurSave: true,
                        };
                        this.savePromptEdit(config);
                    }
                });
                input.addEventListener('blur', () => {
                    const promptState = this.promptEditorState[config.field] || {};
                    if (promptState.skipBlurSave) {
                        this.promptEditorState[config.field] = {
                            ...promptState,
                            skipBlurSave: false,
                        };
                        return;
                    }

                    this.savePromptEdit(config);
                });
            }
        });
    }

    renderPromptContent(element, value, placeholder) {
        if (!element) {
            return;
        }

        const text = value || '';
        if (text) {
            element.textContent = text;
            element.classList.remove('is-placeholder');
        } else {
            element.textContent = placeholder;
            element.classList.add('is-placeholder');
        }
    }

    resetPromptEditors() {
        this.hidePromptEditor({ contentId: 'recipePrompt', editorId: 'recipePromptEditor' });
        this.hidePromptEditor({ contentId: 'recipeNegativePrompt', editorId: 'recipeNegativePromptEditor' });
    }

    showPromptEditor(config) {
        const content = document.getElementById(config.contentId);
        const editor = document.getElementById(config.editorId);
        const input = document.getElementById(config.inputId);

        if (!content || !editor || !input) {
            return;
        }

        const currentValue = this.currentRecipe?.gen_params?.[config.field] || '';
        input.value = currentValue;
        this.promptEditorState[config.field] = {
            initialValue: currentValue,
            skipBlurSave: false,
            isSaving: false,
        };
        content.classList.add('hide');
        editor.classList.add('active');
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
    }

    async savePromptEdit(config) {
        const content = document.getElementById(config.contentId);
        const editor = document.getElementById(config.editorId);
        const input = document.getElementById(config.inputId);

        if (!content || !editor || !input || !this.currentRecipe) {
            return;
        }

        const promptState = this.promptEditorState[config.field] || {};
        if (promptState.isSaving) {
            return;
        }

        const currentGenParams = this.currentRecipe.gen_params || {};
        const nextValue = input.value.trim() === '' ? '' : input.value;
        const currentValue = this.sanitizeGenParams(currentGenParams)?.[config.field] || '';

        if (nextValue === currentValue) {
            this.clearFieldDirty(config.field);
            this.hidePromptEditor(config);
            return;
        }

        const nextGenParams = {
            ...currentGenParams,
            [config.field]: nextValue,
        };

        try {
            this.promptEditorState[config.field] = {
                ...promptState,
                isSaving: true,
            };
            await updateRecipeMetadata(this.filePath, { gen_params: nextGenParams }, this.getMetadataUpdateOptions());
            this.currentRecipe.gen_params = nextGenParams;
            this.renderPromptContent(content, nextValue, config.placeholder);
            showToast(config.successKey, {}, 'success', config.successFallback);
            this.commitField(config.field);
        } catch (error) {
            this.renderPromptContent(content, currentValue, config.placeholder);
            input.value = currentValue;
            this.clearFieldDirty(config.field);
        } finally {
            this.clearFieldDirty(config.field);
            this.hidePromptEditor(config);
        }
    }

    cancelPromptEdit(config) {
        const input = document.getElementById(config.inputId);
        if (input) {
            input.value = this.currentRecipe?.gen_params?.[config.field] || '';
        }

        this.clearFieldDirty(config.field);
        this.hidePromptEditor(config);
    }

    hidePromptEditor(config) {
        const content = document.getElementById(config.contentId);
        const editor = document.getElementById(config.editorId);

        if (content) {
            content.classList.remove('hide');
        }

        if (editor) {
            editor.classList.remove('active');
        }

        delete this.promptEditorState[config.field];
    }

    // Setup source URL handlers
    setupSourceUrlHandlers() {
        const sourceUrlContainer = document.querySelector('.source-url-container');
        const sourceUrlEditor = document.querySelector('.source-url-editor');
        if (!sourceUrlContainer || !sourceUrlEditor) {
            return;
        }
        const sourceUrlText = sourceUrlContainer.querySelector('.source-url-text');
        const sourceUrlEditBtn = sourceUrlContainer.querySelector('.source-url-edit-btn');
        const sourceUrlCancelBtn = sourceUrlEditor.querySelector('.source-url-cancel-btn');
        const sourceUrlSaveBtn = sourceUrlEditor.querySelector('.source-url-save-btn');
        const sourceUrlInput = sourceUrlEditor.querySelector('.source-url-input');

        if (!sourceUrlText || !sourceUrlEditBtn || !sourceUrlCancelBtn || !sourceUrlSaveBtn || !sourceUrlInput) {
            return;
        }

        // Show editor on edit button click
        sourceUrlEditBtn.addEventListener('click', () => {
            sourceUrlContainer.classList.add('hide');
            sourceUrlEditor.classList.add('active');
            sourceUrlInput.focus();
        });

        sourceUrlInput.addEventListener('input', () => {
            this.sourceUrlEditState.isDirty = true;
        });

        // Cancel editing
        sourceUrlCancelBtn.addEventListener('click', () => {
            sourceUrlEditor.classList.remove('active');
            sourceUrlContainer.classList.remove('hide');
            this.updateSourceUrlDisplay(this.currentRecipe.source_path || '', { forceInputSync: true });
            this.clearFieldDirty('source_path');
        });

        // Save new source URL
        sourceUrlSaveBtn.addEventListener('click', () => {
            const newSourceUrl = sourceUrlInput.value.trim();
            if (newSourceUrl !== this.currentRecipe.source_path) {
                // Update the recipe on the server
                updateRecipeMetadata(this.filePath, { source_path: newSourceUrl }, this.getMetadataUpdateOptions())
                    .then(data => {
                        // Show success toast
                        showToast('toast.recipes.sourceUrlUpdated', {}, 'success');

                        // Update source URL in the UI
                        this.commitField('source_path');
                        this.updateSourceUrlDisplay(newSourceUrl, { forceInputSync: true });
                        this.syncSourceUrlAction();

                        // Update the current recipe object
                        this.currentRecipe.source_path = newSourceUrl;
                    })
                    .catch(error => {
                        // Error is handled in the API function
                        this.clearFieldDirty('source_path');
                    });
            } else {
                this.clearFieldDirty('source_path');
            }

            // Hide editor
            sourceUrlEditor.classList.remove('active');
            sourceUrlContainer.classList.remove('hide');
        });

        // Open source URL in a new tab if it's valid
        sourceUrlText.addEventListener('click', () => {
            const url = sourceUrlText.textContent.trim();
            if (url.startsWith('http://') || url.startsWith('https://')) {
                window.open(url, '_blank');
            }
        });
    }

    resetPromptDraftUI() {
        this.currentPromptDraft = null;
        const panel = document.getElementById('aiPromptDraftPanel');
        const status = document.getElementById('aiPromptDraftStatus');
        const source = document.getElementById('aiPromptImageSource');
        const warning = document.getElementById('aiPromptDraftWarning');
        const fragments = document.getElementById('aiPromptProtectedFragments');
        const description = document.getElementById('aiPromptDescription');
        const text = document.getElementById('aiPromptDraftText');
        if (panel) panel.hidden = true;
        if (status) status.textContent = '';
        if (source) source.textContent = '';
        if (warning) {
            warning.textContent = '';
            warning.hidden = true;
        }
        if (fragments) fragments.replaceChildren();
        if (description) description.textContent = '';
        if (text) text.value = '';
        const startButton = document.getElementById('startAiPromptTrialsBtn');
        if (startButton) startButton.disabled = true;
    }

    resetPromptTrialUI() {
        this.currentPromptTrialJob = null;
        const status = document.getElementById('aiPromptTrialStatus');
        const grid = document.getElementById('aiPromptTrialGrid');
        const cancelButton = document.getElementById('cancelAiPromptTrialsBtn');
        if (status) status.textContent = '';
        if (grid) {
            grid.replaceChildren();
            grid.hidden = true;
        }
        if (cancelButton) cancelButton.hidden = true;
    }

    getSelectedPromptModel() {
        const select = document.getElementById('aiPromptModelSelect');
        const selected = select?.value || getStorageItem(
            AI_PROMPT_MODEL_STORAGE_KEY,
            DEFAULT_AI_PROMPT_MODEL
        );
        return Object.hasOwn(AI_PROMPT_MODELS, selected)
            ? selected
            : DEFAULT_AI_PROMPT_MODEL;
    }

    async requestPromptDraft({ forceRegenerate = false } = {}) {
        const recipeId = this.currentRecipe?.id || this.recipeId;
        if (!recipeId) return;
        const requestId = ++this.promptDraftRequestId;
        const createButton = document.getElementById('createAiPromptDraftBtn');
        const regenerateButton = document.getElementById('regenerateAiPromptDraftBtn');
        const modelSelect = document.getElementById('aiPromptModelSelect');
        const panel = document.getElementById('aiPromptDraftPanel');
        const status = document.getElementById('aiPromptDraftStatus');
        const model = this.getSelectedPromptModel();
        const modelLabel = AI_PROMPT_MODELS[model];
        this.currentPromptDraft = null;
        if (panel) panel.hidden = false;
        if (status) {
            status.textContent = forceRegenerate
                ? `${modelLabel}でキャッシュを使わず再解析しています…`
                : `${modelLabel}を準備しています（保存済み結果を確認中）…`;
        }
        if (createButton) createButton.disabled = true;
        if (regenerateButton) regenerateButton.disabled = true;
        if (modelSelect) modelSelect.disabled = true;

        try {
            let manifestHash = this.currentRecipe?.replay_manifest?.manifest_hash;
            if (!manifestHash) {
                const details = await fetchRecipeDetails(recipeId);
                if (requestId !== this.promptDraftRequestId
                    || String(this.currentRecipe?.id || this.recipeId) !== String(recipeId)) return;
                manifestHash = details?.replay_manifest?.manifest_hash;
                if (manifestHash) {
                    this.currentRecipe = {
                        ...(this.currentRecipe || {}),
                        replay_manifest: details.replay_manifest,
                    };
                }
            }
            if (!manifestHash) {
                throw new Error('再現manifestを取得できません。レシピ詳細を再読み込みしてください。');
            }
            const draft = await createRecipePromptDraft(recipeId, {
                manifestHash,
                model,
                forceRegenerate,
            });
            if (requestId !== this.promptDraftRequestId
                || String(this.currentRecipe?.id || this.recipeId) !== String(recipeId)) return;
            this.renderPromptDraft(draft);
        } catch (error) {
            if (requestId !== this.promptDraftRequestId) return;
            this.renderPromptDraftError(error);
        } finally {
            if (requestId === this.promptDraftRequestId && createButton) {
                createButton.disabled = false;
            }
            if (requestId === this.promptDraftRequestId && regenerateButton) {
                regenerateButton.disabled = false;
            }
            if (requestId === this.promptDraftRequestId && modelSelect) {
                modelSelect.disabled = false;
            }
        }
    }

    renderPromptDraft(draft) {
        this.currentPromptDraft = draft;
        const panel = document.getElementById('aiPromptDraftPanel');
        const status = document.getElementById('aiPromptDraftStatus');
        const source = document.getElementById('aiPromptImageSource');
        const warning = document.getElementById('aiPromptDraftWarning');
        const fragments = document.getElementById('aiPromptProtectedFragments');
        const description = document.getElementById('aiPromptDescription');
        const text = document.getElementById('aiPromptDraftText');
        if (panel) panel.hidden = false;
        if (status) {
            const lmStudio = draft?.lm_studio || {};
            const modelLabel = lmStudio.model_label || lmStudio.model || '画像認識モデル';
            const cacheLabel = lmStudio.cache_hit ? '保存済み結果を再利用' : '新規解析';
            status.textContent = `${modelLabel} / CPU / ${cacheLabel}`;
        }
        if (source) {
            const image = draft?.image || {};
            const kind = image.preview_used ? 'プレビュー画像' : '元画像';
            source.textContent = `使用画像: ${kind}（${image.input_width || '?'} × ${image.input_height || '?'}）`;
        }
        if (warning) {
            warning.textContent = draft?.image?.warning || '';
            warning.hidden = !warning.textContent;
        }
        if (fragments) {
            fragments.replaceChildren();
            const protectedParts = draft?.protected || {};
            const values = [
                ...(protectedParts.lora_tags || []),
                ...(protectedParts.trigger_tokens || []),
                ...(protectedParts.embeddings || []),
            ];
            for (const value of values) {
                const chip = document.createElement('span');
                chip.className = 'ai-prompt-protected-fragment';
                chip.textContent = value;
                chip.title = value;
                fragments.appendChild(chip);
            }
            const negative = document.createElement('span');
            negative.className = 'ai-prompt-protected-fragment';
            negative.textContent = 'Negative prompt: 変更なし';
            fragments.appendChild(negative);
        }
        if (description) description.textContent = draft?.description || '';
        if (text) text.value = draft?.proposed_prompt || '';
        const startButton = document.getElementById('startAiPromptTrialsBtn');
        if (startButton) startButton.disabled = !draft?.proposed_prompt;
    }

    renderPromptDraftError(error) {
        const panel = document.getElementById('aiPromptDraftPanel');
        const status = document.getElementById('aiPromptDraftStatus');
        const warning = document.getElementById('aiPromptDraftWarning');
        if (panel) panel.hidden = false;
        if (status) status.textContent = 'AI補完を停止しました';
        if (warning) {
            warning.textContent = `${error?.message || error} 元レシピは変更していません。`;
            warning.hidden = false;
        }
        const startButton = document.getElementById('startAiPromptTrialsBtn');
        if (startButton) startButton.disabled = true;
    }

    async startPromptTrials() {
        const recipeId = this.currentRecipe?.id || this.recipeId;
        const draft = this.currentPromptDraft;
        if (!recipeId || !draft) return;
        const requestId = ++this.promptTrialRequestId;
        const startButton = document.getElementById('startAiPromptTrialsBtn');
        const cancelButton = document.getElementById('cancelAiPromptTrialsBtn');
        const status = document.getElementById('aiPromptTrialStatus');
        if (startButton) startButton.disabled = true;
        if (cancelButton) cancelButton.hidden = false;
        if (status) status.textContent = 'ComfyUIの空きキューを確認しています…';

        const renderIfCurrent = job => {
            const currentId = this.currentRecipe?.id || this.recipeId;
            if (requestId === this.promptTrialRequestId && String(currentId) === String(recipeId)) {
                this.renderPromptTrialJob(job);
            }
        };
        try {
            if (status) status.textContent = '画像生成用にAI画像認識モデルを解放しています…';
            await releaseRecipePromptModel(recipeId);
            if (requestId !== this.promptTrialRequestId) return;
            if (status) status.textContent = 'ComfyUIの空きキューを確認しています…';
            const recipe = await fetchRecipeDetails(recipeId);
            if (requestId !== this.promptTrialRequestId) return;
            await recipeTrialManager.start({ recipe, draft, onUpdate: renderIfCurrent });
        } catch (error) {
            if (requestId !== this.promptTrialRequestId) return;
            if (status) {
                status.textContent = `${error?.message || error} 同じ候補は自動再送していません。`;
            }
        } finally {
            if (requestId === this.promptTrialRequestId) {
                if (cancelButton) cancelButton.hidden = true;
                if (startButton) startButton.disabled = !this.currentPromptDraft;
            }
        }
    }

    async recoverPromptTrials() {
        const recipeId = this.currentRecipe?.id || this.recipeId;
        if (!recipeId) return;
        const requestId = ++this.promptTrialRequestId;
        const renderIfCurrent = job => {
            const currentId = this.currentRecipe?.id || this.recipeId;
            if (requestId === this.promptTrialRequestId && String(currentId) === String(recipeId)) {
                this.renderPromptTrialJob(job);
            }
        };
        try {
            const job = await recipeTrialManager.recover(recipeId, { onUpdate: renderIfCurrent });
            if (job) renderIfCurrent(job);
        } catch (error) {
            if (requestId !== this.promptTrialRequestId) return;
            const status = document.getElementById('aiPromptTrialStatus');
            if (status) status.textContent = `候補履歴を確認できません: ${error?.message || error}`;
        }
    }

    async cancelPromptTrials() {
        const recipeId = this.currentRecipe?.id || this.recipeId;
        const requestId = this.promptTrialRequestId;
        try {
            const job = await recipeTrialManager.cancel();
            if (job && requestId === this.promptTrialRequestId
                && String(job.recipe_id) === String(recipeId)) {
                this.renderPromptTrialJob(job);
            }
        } catch (error) {
            const status = document.getElementById('aiPromptTrialStatus');
            if (status) status.textContent = `停止要求に失敗しました: ${error?.message || error}`;
        }
    }

    getPromptTrialDraft(job) {
        const stored = job?.draft_snapshot;
        if (stored?.draft_hash && stored.draft_hash === job?.draft_hash) return stored;
        const current = this.currentPromptDraft;
        if (current?.draft_hash && current.draft_hash === job?.draft_hash) return current;
        return null;
    }

    async adoptPromptTrialCandidate(job, candidate, imageInfo) {
        const recipeId = this.currentRecipe?.id || this.recipeId;
        const draft = this.getPromptTrialDraft(job);
        if (!recipeId || !draft || !candidate?.candidate_id || !candidate?.prompt_id
            || !imageInfo?.output_node_id
            || !Number.isInteger(imageInfo?.image_index)) {
            const status = document.getElementById('aiPromptTrialStatus');
            if (status) {
                status.textContent = '採用情報が不足しています。AI下書きと候補履歴を作り直してください。';
            }
            return;
        }

        const requestId = ++this.promptRevisionRequestId;
        let completionMessage = '';
        this.adoptingCandidateId = candidate.candidate_id;
        this.renderPromptTrialJob(job);
        try {
            const result = await adoptRecipeRevision(recipeId, {
                sourceEtag: job.source_etag || this.currentRecipe?.source_etag,
                manifestHash: job.manifest_hash,
                draft,
                candidate: {
                    candidate_id: candidate.candidate_id,
                    prompt_id: candidate.prompt_id,
                    output_node_id: imageInfo.output_node_id,
                    image_index: imageInfo.image_index,
                    seed: candidate.seed,
                },
            });
            const currentId = this.currentRecipe?.id || this.recipeId;
            if (requestId !== this.promptRevisionRequestId
                || String(currentId) !== String(recipeId)) return;

            const revisionSummary = result.revision_summary || {
                active: true,
                prompt_source: 'lm_studio',
                seed: candidate.seed,
            };
            this.currentRecipe = {
                ...(this.currentRecipe || {}),
                revision_summary: revisionSummary,
                active_revision: {
                    gen_params: {
                        ...((this.currentRecipe || {}).gen_params || {}),
                        prompt: draft.proposed_prompt,
                        negative_prompt: draft.negative_prompt,
                        seed: candidate.seed,
                        prompt_source: 'lm_studio',
                    },
                    revision_summary: revisionSummary,
                },
            };
            this.adoptedCandidateId = candidate.candidate_id;
            this.renderRecipeRevisionBanner(this.currentRecipe);
            state.virtualScroller?.updateSingleItem(
                this.listFilePath || this.currentRecipe.file_path,
                { revision_summary: revisionSummary }
            );
            completionMessage = 'この候補を採用しました。元レシピは変更していません。';
            showToast(
                'toast.recipes.revisionAdopted',
                {},
                'success',
                'AI補完・改変版を採用しました。元レシピは変更していません。'
            );
        } catch (error) {
            if (requestId !== this.promptRevisionRequestId) return;
            completionMessage = `${error?.message || error} 元レシピは変更していません。`;
        } finally {
            if (requestId === this.promptRevisionRequestId) {
                this.adoptingCandidateId = null;
                this.renderPromptTrialJob(job);
                const status = document.getElementById('aiPromptTrialStatus');
                if (status && completionMessage) status.textContent = completionMessage;
            }
        }
    }

    renderPromptTrialJob(job) {
        if (!job) return;
        this.currentPromptTrialJob = job;
        const panel = document.getElementById('aiPromptDraftPanel');
        const status = document.getElementById('aiPromptTrialStatus');
        const grid = document.getElementById('aiPromptTrialGrid');
        const startButton = document.getElementById('startAiPromptTrialsBtn');
        const cancelButton = document.getElementById('cancelAiPromptTrialsBtn');
        if (panel) panel.hidden = false;

        const activeNumber = Number.isInteger(job.active_index) ? job.active_index + 1 : null;
        const succeeded = (job.candidates || []).filter(item => item.status === 'succeeded').length;
        const jobLabels = {
            running: activeNumber ? `候補 ${activeNumber}/4 を逐次生成中（完了 ${succeeded}/4）` : '4候補の生成を準備中',
            recovering: `既存promptを監視中（完了 ${succeeded}/4、自動再送なし）`,
            completed: '4候補すべて完了しました。',
            partial: `一部完了しました（${succeeded}/4）。完了済み候補は採用できます。`,
            failed: `候補生成を停止しました（完了 ${succeeded}/4）。`,
            canceled: `候補生成を停止しました（完了 ${succeeded}/4）。`,
        };
        if (status) status.textContent = jobLabels[job.status] || job.error || '';
        const isActive = ['running', 'recovering'].includes(job.status);
        if (startButton) startButton.disabled = isActive || !this.currentPromptDraft;
        if (cancelButton) cancelButton.hidden = !isActive;

        if (!grid) return;
        grid.replaceChildren();
        const candidateLabels = {
            pending: '未開始',
            submitting: '送信確認中',
            queued: 'キュー待ち',
            running: '生成中',
            succeeded: '生成完了',
            failed: '失敗',
            canceled: '停止済み',
            not_submitted: '未送信',
            submission_unknown: '送信状態不明（再送なし）',
        };
        for (const candidate of job.candidates || []) {
            const card = document.createElement('article');
            card.className = 'ai-prompt-trial-card';
            card.dataset.candidateIndex = String(candidate.index);
            const header = document.createElement('div');
            header.className = 'ai-prompt-trial-card-header';
            const title = document.createElement('strong');
            title.textContent = `候補 ${candidate.index + 1}${candidate.seed_origin === 'original' ? '（元seed）' : ''}`;
            const seed = document.createElement('span');
            seed.className = 'ai-prompt-trial-seed';
            seed.textContent = `seed ${candidate.seed}`;
            seed.title = seed.textContent;
            header.append(title, seed);
            card.appendChild(header);

            const imageInfo = candidate.images?.[0];
            if (imageInfo?.url) {
                const image = document.createElement('img');
                image.className = 'ai-prompt-trial-image';
                image.src = imageInfo.url;
                image.alt = `候補 ${candidate.index + 1}`;
                image.loading = 'lazy';
                card.appendChild(image);
            }
            const candidateStatus = document.createElement('div');
            candidateStatus.className = 'ai-prompt-trial-card-status';
            candidateStatus.textContent = candidate.error
                ? `${candidateLabels[candidate.status] || candidate.status}: ${candidate.error}`
                : (candidateLabels[candidate.status] || candidate.status || '');
            card.appendChild(candidateStatus);

            if (candidate.status === 'succeeded' && imageInfo) {
                const actions = document.createElement('div');
                actions.className = 'ai-prompt-trial-card-actions';
                const adoptButton = document.createElement('button');
                adoptButton.className = 'ai-prompt-trial-adopt';
                adoptButton.type = 'button';
                const hasDraft = Boolean(this.getPromptTrialDraft(job));
                const isAdopting = this.adoptingCandidateId === candidate.candidate_id;
                const isAdopted = this.adoptedCandidateId === candidate.candidate_id;
                adoptButton.disabled = !hasDraft || isAdopting || isAdopted;
                adoptButton.textContent = isAdopting
                    ? '採用中…'
                    : (isAdopted ? '採用済み' : 'この結果を採用');
                if (!hasDraft) {
                    adoptButton.title = '保存済みAI下書きがないため、この候補は採用できません。';
                } else {
                    adoptButton.title = '元レシピを変更せず、AI補完・改変版として保存します。';
                }
                adoptButton.addEventListener('click', () => {
                    this.adoptPromptTrialCandidate(job, candidate, imageInfo);
                });
                actions.appendChild(adoptButton);
                card.appendChild(actions);
            }
            grid.appendChild(card);
        }
        grid.hidden = (job.candidates || []).length === 0;
    }

    // Setup copy buttons for prompts and recipe syntax
    setupCopyButtons() {
        const copyPromptBtn = document.getElementById('copyPromptBtn');
        const copyNegativePromptBtn = document.getElementById('copyNegativePromptBtn');
        const copyRecipeSyntaxBtn = document.getElementById('copyRecipeSyntaxBtn');
        const copyRecipeReferenceBtn = document.getElementById('copyRecipeReferenceBtn');
        const sendRecipeBtn = document.getElementById('sendRecipeBtn');
        const replayRecipeWorkflowBtn = document.getElementById('replayRecipeWorkflowBtn');
        const createAiPromptDraftBtn = document.getElementById('createAiPromptDraftBtn');
        const regenerateAiPromptDraftBtn = document.getElementById('regenerateAiPromptDraftBtn');
        const aiPromptModelSelect = document.getElementById('aiPromptModelSelect');
        const copyAiPromptDraftBtn = document.getElementById('copyAiPromptDraftBtn');
        const startAiPromptTrialsBtn = document.getElementById('startAiPromptTrialsBtn');
        const refreshAiPromptTrialsBtn = document.getElementById('refreshAiPromptTrialsBtn');
        const cancelAiPromptTrialsBtn = document.getElementById('cancelAiPromptTrialsBtn');

        if (createAiPromptDraftBtn) {
            createAiPromptDraftBtn.addEventListener('click', () => this.requestPromptDraft());
        }
        if (regenerateAiPromptDraftBtn) {
            regenerateAiPromptDraftBtn.addEventListener(
                'click',
                () => this.requestPromptDraft({ forceRegenerate: true })
            );
        }
        if (aiPromptModelSelect) {
            const storedModel = getStorageItem(
                AI_PROMPT_MODEL_STORAGE_KEY,
                DEFAULT_AI_PROMPT_MODEL
            );
            aiPromptModelSelect.value = Object.hasOwn(AI_PROMPT_MODELS, storedModel)
                ? storedModel
                : DEFAULT_AI_PROMPT_MODEL;
            aiPromptModelSelect.addEventListener('change', () => {
                const model = this.getSelectedPromptModel();
                aiPromptModelSelect.value = model;
                setStorageItem(AI_PROMPT_MODEL_STORAGE_KEY, model);
            });
        }
        if (copyAiPromptDraftBtn) {
            copyAiPromptDraftBtn.addEventListener('click', () => {
                const prompt = this.currentPromptDraft?.proposed_prompt || '';
                if (prompt) this.copyToClipboard(prompt, 'AI prompt draft copied');
            });
        }
        if (startAiPromptTrialsBtn) {
            startAiPromptTrialsBtn.addEventListener('click', () => this.startPromptTrials());
        }
        if (refreshAiPromptTrialsBtn) {
            refreshAiPromptTrialsBtn.addEventListener('click', () => this.recoverPromptTrials());
        }
        if (cancelAiPromptTrialsBtn) {
            cancelAiPromptTrialsBtn.addEventListener('click', () => this.cancelPromptTrials());
        }

        if (copyPromptBtn) {
            copyPromptBtn.addEventListener('click', () => {
                let promptText = this.currentRecipe?.gen_params?.prompt || '';
                if (this.shouldStripLoraOnCopy()) {
                    promptText = RecipeModal.stripLoraTags(promptText);
                }
                this.copyToClipboard(promptText, 'Prompt copied to clipboard');
            });
        }

        if (copyNegativePromptBtn) {
            copyNegativePromptBtn.addEventListener('click', () => {
                let negativePromptText = this.currentRecipe?.gen_params?.negative_prompt || '';
                if (this.shouldStripLoraOnCopy()) {
                    negativePromptText = RecipeModal.stripLoraTags(negativePromptText);
                }
                this.copyToClipboard(negativePromptText, 'Negative prompt copied to clipboard');
            });
        }

        if (copyRecipeSyntaxBtn) {
            copyRecipeSyntaxBtn.addEventListener('click', () => {
                // Use backend API to get recipe syntax
                this.fetchAndCopyRecipeSyntax();
            });
        }

        if (copyRecipeReferenceBtn) {
            copyRecipeReferenceBtn.addEventListener('click', () => {
                const text = buildRecipeReferenceText(this.currentRecipe);
                if (!text) {
                    showToast(
                        'toast.recipes.referenceUnavailable',
                        {},
                        'warning',
                        '参考情報を取得できませんでした'
                    );
                    return;
                }
                this.copyToClipboard(text, '参考情報をコピーしました');
            });
        }

        if (sendRecipeBtn) {
            sendRecipeBtn.addEventListener('click', () => {
                // Send recipe to ComfyUI workflow
                this.sendRecipeToWorkflow();
            });
        }

        if (replayRecipeWorkflowBtn) {
            replayRecipeWorkflowBtn.addEventListener('click', async () => {
                if (!this.currentRecipe && !this.recipeId) return;
                replayRecipeWorkflowBtn.disabled = true;
                try {
                    await releaseRecipePromptModel(this.currentRecipe?.id || this.recipeId);
                    await recipeWorkflowReplayManager.replay(this.currentRecipe || this.recipeId);
                } catch (error) {
                    console.error('Error replaying recipe workflow:', error);
                    showToast(
                        'toast.recipes.workflowReplayFailed',
                        { message: error.message },
                        'error',
                        `ComfyUIワークフローを再現できませんでした: ${error.message}`
                    );
                } finally {
                    replayRecipeWorkflowBtn.disabled = false;
                }
            });
        }

        // Send prompt to workflow buttons
        const sendPromptBtn = document.getElementById('sendPromptBtn');
        const sendNegativePromptBtn = document.getElementById('sendNegativePromptBtn');

        if (sendPromptBtn) {
            sendPromptBtn.addEventListener('click', () => {
                let promptText = this.currentRecipe?.gen_params?.prompt || '';
                if (this.shouldStripLoraOnCopy()) {
                    promptText = RecipeModal.stripLoraTags(promptText);
                }
                if (!promptText.trim()) {
                    showToast('toast.recipes.noPromptToSend', {}, 'warning');
                    return;
                }
                sendPromptToWorkflow(promptText);
            });
        }

        if (sendNegativePromptBtn) {
            sendNegativePromptBtn.addEventListener('click', () => {
                let negativePromptText = this.currentRecipe?.gen_params?.negative_prompt || '';
                if (this.shouldStripLoraOnCopy()) {
                    negativePromptText = RecipeModal.stripLoraTags(negativePromptText);
                }
                if (!negativePromptText.trim()) {
                    showToast('toast.recipes.noPromptToSend', {}, 'warning');
                    return;
                }
                sendPromptToWorkflow(negativePromptText, {
                    actionTypeText: 'Negative Prompt',
                });
            });
        }

        // Send params to workflow button
        const sendParamsBtn = document.getElementById('sendParamsBtn');
        if (sendParamsBtn) {
            sendParamsBtn.addEventListener('click', () => {
                const genParams = this.currentRecipe?.gen_params || {};
                if (!genParams || Object.keys(genParams).length === 0) {
                    showToast('No generation parameters available', {}, 'warning');
                    return;
                }
                sendGenParamsToWorkflow(genParams);
            });
        }
    }

    /**
     * Strip <lora:...> tags from prompt text and clean up residual punctuation/whitespace.
     * Handles both unescaped (<lora:...>) and HTML-escaped (&lt;lora:...&gt;) variants.
     * Cleans up artifacts like leading ", ", double commas, and extra whitespace.
     */
    static stripLoraTags(text) {
        return stripLoraTags(text);
    }

    shouldStripLoraOnCopy() {
        const toggle = document.getElementById('stripLoraOnCopyToggle');
        return toggle ? toggle.checked : false;
    }

    setupStripLoraToggle() {
        const toggle = document.getElementById('stripLoraOnCopyToggle');
        if (!toggle) return;

        const stored = getStorageItem('strip_lora_on_copy');
        if (stored !== null) {
            toggle.checked = stored === true;
        }

        toggle.addEventListener('change', () => {
            const checked = toggle.checked;
            setStorageItem('strip_lora_on_copy', checked);
            state.global.settings.strip_lora_on_copy = checked;
        });
    }

    // Fetch recipe syntax from backend and copy to clipboard
    async fetchAndCopyRecipeSyntax() {
        if (!this.recipeId) {
            showToast('toast.recipes.noRecipeId', {}, 'error');
            return;
        }

        try {
            // Fetch recipe syntax from backend
            const response = await fetch(`/api/lm/recipe/${this.recipeId}/syntax`);

            if (!response.ok) {
                throw new Error(`Failed to get recipe syntax: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success && data.syntax) {
                // Use the centralized copyToClipboard utility function
                await copyToClipboard(data.syntax, 'Recipe syntax copied to clipboard');
            } else {
                throw new Error(data.error || 'No syntax returned from server');
            }
        } catch (error) {
            console.error('Error fetching recipe syntax:', error);
            showToast('toast.recipes.copyFailed', { message: error.message }, 'error');
        }
    }

    // Helper method to copy text to clipboard
    copyToClipboard(text, successMessage) {
        copyToClipboard(text, successMessage);
    }

    // Send recipe to ComfyUI workflow
    async sendRecipeToWorkflow() {
        if (!this.recipeId) {
            showToast('toast.recipes.noRecipeId', {}, 'error');
            return;
        }

        try {
            // Fetch recipe syntax from backend
            const response = await fetch(`/api/lm/recipe/${this.recipeId}/syntax`);

            if (!response.ok) {
                throw new Error(`Failed to get recipe syntax: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success && data.syntax) {
                // Send the recipe syntax to ComfyUI workflow
                await sendLoraToWorkflow(data.syntax, false, 'recipe');
            } else {
                throw new Error(data.error || 'No syntax returned from server');
            }
        } catch (error) {
            console.error('Error sending recipe to workflow:', error);
            showToast('toast.recipes.sendToWorkflowFailed', { message: error.message }, 'error');
        }
    }

    // Add new method to handle downloading missing LoRAs
    async showDownloadMissingLorasModal() {
        console.log("currentRecipe", this.currentRecipe);
        // Get missing LoRAs from the current recipe
        const missingLoras = this.currentRecipe.loras.filter(lora => !lora.inLibrary);
        console.log("missingLoras", missingLoras);

        if (missingLoras.length === 0) {
            showToast('toast.recipes.noMissingLoras', {}, 'info');
            return;
        }

        try {
            state.loadingManager.showSimpleLoading('Getting version info for missing LoRAs...');

            // Get version info for each missing LoRA by calling the appropriate API endpoint
            const missingLorasWithVersionInfoPromises = missingLoras.map(async lora => {
                let endpoint;

                // Determine which endpoint to use based on available data
                if (lora.modelVersionId) {
                    endpoint = `/api/lm/loras/civitai/model/version/${lora.modelVersionId}`;
                } else if (lora.hash) {
                    endpoint = `/api/lm/loras/civitai/model/hash/${lora.hash}`;
                } else {
                    console.error("Missing both hash and modelVersionId for lora:", lora);
                    return null;
                }

                const response = await fetch(endpoint);
                const versionInfo = await response.json();

                // Return original lora data combined with version info
                return {
                    ...lora,
                    civitaiInfo: versionInfo
                };
            });

            // Wait for all API calls to complete
            const lorasWithVersionInfo = await Promise.all(missingLorasWithVersionInfoPromises);
            console.log("Loras with version info:", lorasWithVersionInfo);

            // Filter out null values (failed requests)
            const validLoras = lorasWithVersionInfo.filter(lora => lora !== null);

            if (validLoras.length === 0) {
                showToast('toast.recipes.missingLorasInfoFailed', {}, 'error');
                return;
            }

            // Close the recipe modal first
            modalManager.closeModal('recipeModal');

            // Prepare data for import manager using the retrieved information
            const recipeData = {
                loras: validLoras.map(lora => {
                    const civitaiInfo = lora.civitaiInfo;
                    const modelFile = civitaiInfo.files ?
                        civitaiInfo.files.find(file => file.type === 'Model') : null;

                    return {
                        // Basic lora info
                        name: civitaiInfo.model?.name || lora.name,
                        version: civitaiInfo.name || '',
                        strength: lora.strength || 1.0,

                        // Model identifiers
                        modelId: civitaiInfo.modelId || lora.modelId,
                        hash: modelFile?.hashes?.SHA256?.toLowerCase() || lora.hash,
                        id: civitaiInfo.id || lora.modelVersionId,

                        // Metadata
                        thumbnailUrl: civitaiInfo.images?.[0]?.url || '',
                        baseModel: civitaiInfo.baseModel || '',
                        downloadUrl: civitaiInfo.downloadUrl || '',
                        size: modelFile ? (modelFile.sizeKB * 1024) : 0,
                        file_name: modelFile ? modelFile.name.split('.')[0] : '',

                        // Status flags
                        existsLocally: false,
                        isDeleted: civitaiInfo.error === "Model not found",
                        isEarlyAccess: !!civitaiInfo.earlyAccessEndsAt,
                        earlyAccessEndsAt: civitaiInfo.earlyAccessEndsAt || ''
                    };
                })
            };

            console.log("recipeData for import:", recipeData);

            // Call ImportManager's download missing LoRAs method
            window.importManager.downloadMissingLoras(recipeData, this.currentRecipe.id);
        } catch (error) {
            console.error("Error downloading missing LoRAs:", error);
            showToast('toast.recipes.preparingForDownloadFailed', {}, 'error');
        } finally {
            state.loadingManager.hide();
        }
    }

    // New methods for reconnecting LoRAs
    setupReconnectButtons() {
        // Add event listeners to all deleted badges
        const deletedBadges = document.querySelectorAll('.deleted-badge.reconnectable');
        deletedBadges.forEach(badge => {
            badge.addEventListener('mouseenter', () => {
                badge.querySelector('.badge-text').innerHTML = 'Reconnect';
            });

            badge.addEventListener('mouseleave', () => {
                badge.querySelector('.badge-text').innerHTML = '<i class="fas fa-trash-alt"></i> Deleted';
            });

            badge.addEventListener('click', (e) => {
                const loraIndex = badge.getAttribute('data-lora-index');
                this.showReconnectInput(loraIndex);
            });
        });

        // Add event listeners to reconnect cancel buttons
        const cancelButtons = document.querySelectorAll('.reconnect-cancel-btn');
        cancelButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const container = button.closest('.lora-reconnect-container');
                this.hideReconnectInput(container);
            });
        });

        // Add event listeners to reconnect confirm buttons
        const confirmButtons = document.querySelectorAll('.reconnect-confirm-btn');
        confirmButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const container = button.closest('.lora-reconnect-container');
                const input = container.querySelector('.reconnect-input');
                const loraIndex = container.getAttribute('data-lora-index');
                this.reconnectLora(loraIndex, input.value);
            });
        });

        // Add keydown handlers to reconnect inputs
        const reconnectInputs = document.querySelectorAll('.reconnect-input');
        reconnectInputs.forEach(input => {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const container = input.closest('.lora-reconnect-container');
                    const loraIndex = container.getAttribute('data-lora-index');
                    this.reconnectLora(loraIndex, input.value);
                } else if (e.key === 'Escape') {
                    const container = input.closest('.lora-reconnect-container');
                    this.hideReconnectInput(container);
                }
            });
        });
    }

    showReconnectInput(loraIndex) {
        // Hide any currently active reconnect containers
        document.querySelectorAll('.lora-reconnect-container.active').forEach(active => {
            active.classList.remove('active');
        });

        // Show the reconnect container for this lora
        const container = document.querySelector(`.lora-reconnect-container[data-lora-index="${loraIndex}"]`);
        if (container) {
            container.classList.add('active');
            const input = container.querySelector('.reconnect-input');
            input.focus();
        }
    }

    hideReconnectInput(container) {
        if (container && container.classList.contains('active')) {
            container.classList.remove('active');
            const input = container.querySelector('.reconnect-input');
            if (input) input.value = '';
        }
    }

    async reconnectLora(loraIndex, inputValue) {
        if (!inputValue || !inputValue.trim()) {
            showToast('toast.recipes.enterLoraName', {}, 'error');
            return;
        }

        try {
            // Parse input value to extract file_name
            let loraSyntaxMatch = inputValue.match(/<lora:([^:>]+)(?::[^>]+)?>/);
            let fileName = loraSyntaxMatch ? loraSyntaxMatch[1] : inputValue.trim();

            // Remove .safetensors extension if present
            fileName = fileName.replace(/\.safetensors$/, '');

            state.loadingManager.showSimpleLoading('Reconnecting LoRA...');

            // Call API to reconnect the LoRA
            const response = await fetch('/api/lm/recipe/lora/reconnect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    recipe_id: this.recipeId,
                    lora_index: loraIndex,
                    target_name: fileName
                })
            });

            const result = await response.json();

            if (result.success) {
                // Hide the reconnect input
                const container = document.querySelector(`.lora-reconnect-container[data-lora-index="${loraIndex}"]`);
                this.hideReconnectInput(container);

                // Update the current recipe with the updated lora data
                this.currentRecipe.loras[loraIndex] = result.updated_lora;

                // Show success message
                showToast('toast.recipes.reconnectedSuccessfully', {}, 'success');

                // Refresh modal to show updated content
                setTimeout(() => {
                    this.showRecipeDetails(this.currentRecipe);
                }, 500);

                state.virtualScroller.updateSingleItem(this.listFilePath || this.currentRecipe.file_path, {
                    loras: this.currentRecipe.loras
                });
            } else {
                showToast('toast.recipes.reconnectFailed', { message: result.error }, 'error');
            }
        } catch (error) {
            console.error('Error reconnecting LoRA:', error);
            showToast('toast.recipes.reconnectFailed', { message: error.message }, 'error');
        } finally {
            state.loadingManager.hide();
        }
    }

    renderCheckpoint(checkpoint) {
        const existsLocally = !!checkpoint.inLibrary;
        const localPath = checkpoint.localPath || '';
        const previewUrl = checkpoint.preview_url || checkpoint.thumbnailUrl || '/loras_static/images/no-preview.png';
        const isPreviewVideo = typeof previewUrl === 'string' && previewUrl.toLowerCase().endsWith('.mp4');
        const checkpointName = checkpoint.name || checkpoint.modelName || checkpoint.file_name || 'Checkpoint';
        const versionLabel = checkpoint.version || checkpoint.modelVersionName || '';
        const baseModel = checkpoint.baseModel || checkpoint.base_model || '';
        const modelTypeRaw = (checkpoint.sub_type || checkpoint.type || 'checkpoint').toLowerCase();
        const modelTypeLabel = modelTypeRaw === 'diffusion_model' ? 'Diffusion Model' : 'Checkpoint';

        const previewMedia = isPreviewVideo ? `
            <video class="thumbnail-video" autoplay loop muted playsinline>
                <source src="${previewUrl}" type="video/mp4">
            </video>
        ` : `<img src="${previewUrl}" alt="Checkpoint preview">`;

        const badge = existsLocally ? `
            <div class="local-badge">
                <i class="fas fa-check"></i> In Library
                <div class="local-path">${localPath}</div>
            </div>
        ` : `
            <div class="missing-badge">
                <i class="fas fa-exclamation-triangle"></i> Not in Library
            </div>
        `;

        let headerAction = '';
        if (existsLocally && localPath) {
            headerAction = `
                <button class="resource-action primary compact checkpoint-send">
                    <i class="fas fa-paper-plane"></i>
                    <span>${translate('recipes.actions.sendCheckpoint', {}, 'Send to ComfyUI')}</span>
                </button>
            `;
        } else if (this.canDownloadCheckpoint(checkpoint)) {
            headerAction = `
                <button class="resource-action primary compact checkpoint-download">
                    <i class="fas fa-download"></i>
                    <span>${translate('modals.model.versions.actions.download', {}, 'Download')}</span>
                </button>
            `;
        }

        return `
            <div class="recipe-lora-item checkpoint-item ${existsLocally ? 'exists-locally' : 'missing-locally'}">
                <div class="recipe-lora-thumbnail">
                    ${previewMedia}
                </div>
                <div class="recipe-lora-content">
                    <div class="recipe-lora-header">
                        <h4>${checkpointName}</h4>
                        <div class="badge-container">${headerAction}</div>
                    </div>
                    <div class="recipe-lora-info recipe-checkpoint-meta">
                        ${versionLabel ? `<div class="recipe-lora-version">${versionLabel}</div>` : ''}
                        ${baseModel ? `<div class="base-model">${baseModel}</div>` : ''}
                        ${modelTypeLabel ? `<div class="checkpoint-type">${modelTypeLabel}</div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    setupCheckpointActions(container, checkpoint) {
        const sendBtn = container.querySelector('.checkpoint-send');
        if (sendBtn) {
            sendBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.sendCheckpointToWorkflow(checkpoint);
            });
        }

        const downloadBtn = container.querySelector('.checkpoint-download');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await this.downloadCheckpoint(checkpoint, downloadBtn);
            });
        }
    }

    setupCheckpointNavigation(container, checkpoint) {
        const checkpointItem = container.querySelector('.checkpoint-item');
        if (!checkpointItem) return;

        checkpointItem.addEventListener('click', () => {
            this.navigateToCheckpointPage(checkpoint);
        });
    }

    canDownloadLora(lora) {
        if (!lora || lora.inLibrary || lora.isDeleted) return false;
        const modelId = lora.modelId || lora.modelID || lora.model_id;
        const versionId = lora.id || lora.modelVersionId;
        return !!(modelId && versionId);
    }

    setupLoraDownloadActions() {
        document.querySelectorAll('.lora-download').forEach(button => {
            button.addEventListener('click', async event => {
                event.preventDefault();
                event.stopPropagation();
                const loraIndex = Number.parseInt(button.dataset.loraIndex, 10);
                const lora = this.currentRecipe?.loras?.[loraIndex];
                await this.downloadLora(lora, button);
            });
        });
    }

    async downloadLora(lora, button) {
        if (!this.canDownloadLora(lora)) {
            showToast('toast.recipes.missingLorasInfoFailed', {}, 'error');
            return;
        }

        const modelId = lora.modelId || lora.modelID || lora.model_id;
        const versionId = lora.id || lora.modelVersionId;
        const versionName = lora.version || lora.modelVersionName || lora.modelName || lora.name || 'LoRA';

        button.disabled = true;
        try {
            const success = await downloadManager.downloadVersionWithDefaults(
                MODEL_TYPES.LORA,
                modelId,
                versionId,
                {
                    versionName,
                    source: lora.metadataSource === 'civarchive' ? 'civarchive' : null,
                    fileParams: getRecipeFileParams(lora),
                }
            );

            if (!success) return;

            await this.refreshRecipeResourceState([MODEL_TYPES.LORA]);
        } catch (error) {
            console.error('Error downloading LoRA from recipe:', error);
            showToast('toast.downloads.downloadError', { message: error.message }, 'error');
        } finally {
            button.disabled = false;
        }
    }

    canDownloadEmbedding(embedding) {
        if (!embedding || embedding.inLibrary || embedding.isDeleted) return false;
        const modelId = embedding.modelId || embedding.modelID || embedding.model_id;
        const versionId = embedding.id || embedding.modelVersionId;
        return !!(modelId && versionId);
    }

    setupEmbeddingDownloadActions() {
        document.querySelectorAll('.embedding-download').forEach(button => {
            button.addEventListener('click', async event => {
                event.preventDefault();
                event.stopPropagation();
                const embeddingIndex = Number.parseInt(button.dataset.embeddingIndex, 10);
                const embedding = this.currentRecipe?.embeddings?.[embeddingIndex];
                if (!this.canDownloadEmbedding(embedding)) return;
                const modelId = embedding.modelId || embedding.modelID || embedding.model_id;
                const versionId = embedding.id || embedding.modelVersionId;
                const versionName = embedding.version || embedding.modelVersionName || embedding.modelName || embedding.name || 'Embedding';
                button.disabled = true;
                try {
                    const success = await downloadManager.downloadVersionWithDefaults(
                        MODEL_TYPES.EMBEDDING,
                        modelId,
                        versionId,
                        {
                            versionName,
                            source: 'recipe-modal',
                            fileParams: getRecipeFileParams(embedding),
                        }
                    );
                    if (!success) return;
                    await this.refreshRecipeResourceState([MODEL_TYPES.EMBEDDING]);
                } catch (error) {
                    console.error('Error downloading embedding from recipe:', error);
                    showToast('toast.downloads.downloadError', { message: error.message }, 'error');
                } finally {
                    button.disabled = false;
                }
            });
        });
    }

    canDownloadCheckpoint(checkpoint) {
        if (!checkpoint) return false;
        const modelId = checkpoint.modelId || checkpoint.modelID || checkpoint.model_id;
        const versionId = checkpoint.id || checkpoint.modelVersionId;
        return !!(modelId && versionId);
    }

    async sendCheckpointToWorkflow(checkpoint) {
        if (!checkpoint || !checkpoint.localPath) {
            showToast('toast.recipes.missingCheckpointPath', {}, 'error');
            return;
        }

        const modelType = (checkpoint.sub_type || checkpoint.type || 'checkpoint').toLowerCase();
        const isDiffusionModel = modelType === 'diffusion_model' || modelType === 'unet';
        const widgetName = isDiffusionModel ? 'unet_name' : 'ckpt_name';

        const actionTypeText = translate(
            isDiffusionModel ? 'uiHelpers.nodeSelector.diffusionModel' : 'uiHelpers.nodeSelector.checkpoint',
            {},
            isDiffusionModel ? 'Diffusion Model' : 'Checkpoint'
        );
        const successMessage = translate(
            'uiHelpers.workflow.modelUpdated',
            {},
            'Model updated in workflow'
        );
        const failureMessage = translate(
            'uiHelpers.workflow.modelFailed',
            {},
            'Failed to update model node'
        );
        const missingNodesMessage = translate(
            'uiHelpers.workflow.noMatchingNodes',
            {},
            'No compatible nodes available in the current workflow'
        );
        const missingTargetMessage = translate(
            'uiHelpers.workflow.noTargetNodeSelected',
            {},
            'No target node selected'
        );

        await sendModelPathToWorkflow(checkpoint.localPath, {
            widgetName,
            collectionType: MODEL_TYPES.CHECKPOINT,
            actionTypeText,
            successMessage,
            failureMessage,
            missingNodesMessage,
            missingTargetMessage,
        });
    }

    async downloadCheckpoint(checkpoint, button) {
        if (!this.canDownloadCheckpoint(checkpoint)) {
            showToast('toast.recipes.missingCheckpointInfo', {}, 'error');
            return;
        }

        const modelId = checkpoint.modelId || checkpoint.modelID || checkpoint.model_id;
        const versionId = checkpoint.id || checkpoint.modelVersionId;
        const versionName = checkpoint.version || checkpoint.modelVersionName || checkpoint.name || 'Checkpoint';

        if (button) {
            button.disabled = true;
        }

        try {
            const success = await downloadManager.downloadVersionWithDefaults(
                MODEL_TYPES.CHECKPOINT,
                modelId,
                versionId,
                {
                    versionName,
                    source: 'recipe-modal',
                    fileParams: getRecipeFileParams(checkpoint),
                }
            );
            if (!success) return;
            await this.refreshRecipeResourceState([MODEL_TYPES.CHECKPOINT]);
        } catch (error) {
            console.error('Error downloading checkpoint:', error);
            showToast('toast.recipes.downloadCheckpointFailed', { message: error.message }, 'error');
        } finally {
            if (button) {
                button.disabled = false;
            }
        }
    }

    async refreshRecipeResourceState(resourceTypes = []) {
        const endpointByType = {
            [MODEL_TYPES.LORA]: '/api/lm/loras/scan',
            [MODEL_TYPES.CHECKPOINT]: '/api/lm/checkpoints/scan',
            [MODEL_TYPES.EMBEDDING]: '/api/lm/embeddings/scan',
        };
        const endpoints = [...new Set(resourceTypes.map(type => endpointByType[type]).filter(Boolean))];

        for (const endpoint of endpoints) {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`Failed to refresh downloaded model library (${response.status})`);
            }
        }

        const recipeScan = await fetch('/api/lm/recipes/scan');
        if (!recipeScan.ok) {
            throw new Error(`Failed to refresh recipes (${recipeScan.status})`);
        }
        if (!this.recipeId) return;

        const refreshedRecipe = await fetchRecipeDetails(this.recipeId);
        this.currentRecipe = refreshedRecipe;
        this.syncResourcesSection(refreshedRecipe);
    }

    navigateToCheckpointPage(checkpoint) {
        if (!checkpoint.inLibrary) {
            const modelId = checkpoint.modelId || checkpoint.modelID || checkpoint.model_id;
            const versionId = checkpoint.id || checkpoint.modelVersionId;
            const modelName = checkpoint.name || checkpoint.modelName || checkpoint.file_name;

            if (modelId || versionId || modelName) {
                openCivitaiByMetadata(modelId, versionId, modelName);
                return;
            }
        }

        const checkpointHash = this._getCheckpointHash(checkpoint);

        if (!checkpointHash) {
            showToast('toast.recipes.missingCheckpointInfo', {}, 'error');
            return;
        }

        modalManager.closeModal('recipeModal');

        removeSessionItem('recipe_to_checkpoint_filterHash');
        removeSessionItem('recipe_to_checkpoint_filterHashes');
        removeSessionItem('filterCheckpointRecipeName');

        setSessionItem('recipe_to_checkpoint_filterHash', checkpointHash.toLowerCase());
        if (this.currentRecipe?.title) {
            setSessionItem('filterCheckpointRecipeName', this.currentRecipe.title);
        }

        window.location.href = '/checkpoints';
    }

    _getCheckpointHash(checkpoint) {
        if (!checkpoint) return '';
        const hash =
            checkpoint.hash ||
            checkpoint.sha256 ||
            checkpoint.sha256_hash ||
            checkpoint.sha256Hash ||
            checkpoint.SHA256;
        return hash ? hash.toString() : '';
    }

    // New method to navigate to the LoRAs page
    navigateToLorasPage(specificLoraIndex = null) {
        // Close the current modal
        modalManager.closeModal('recipeModal');

        // Clear any previous filters first
        removeSessionItem('recipe_to_lora_filterLoraHash');
        removeSessionItem('recipe_to_lora_filterLoraHashes');
        removeSessionItem('filterRecipeName');
        removeSessionItem('viewLoraDetail');

        if (specificLoraIndex !== null) {
            // If a specific LoRA index is provided, navigate to view just that one LoRA
            const lora = this.currentRecipe.loras[specificLoraIndex];

            if (lora && !lora.inLibrary) {
                const modelId = lora.modelId || lora.modelID || lora.model_id;
                const versionId = lora.id || lora.modelVersionId;
                const modelName = lora.modelName || lora.name || lora.file_name;

                if (lora.metadataSource === 'civarchive') {
                    const archiveUrl = buildCivArchiveModelUrl(modelId, versionId);
                    if (archiveUrl) {
                        openCivitaiUrl(archiveUrl);
                        return;
                    }
                }

                if (modelId || versionId || modelName) {
                    openCivitaiByMetadata(modelId, versionId, modelName);
                    return;
                }
            }

            if (lora && lora.hash) {
                // Set session storage to open the LoRA modal directly
                setSessionItem('recipe_to_lora_filterLoraHash', lora.hash.toLowerCase());
                setSessionItem('viewLoraDetail', 'true');
                setSessionItem('filterRecipeName', this.currentRecipe.title);
            }
        } else {
            // If no specific LoRA index is provided, show all LoRAs from this recipe
            // Collect all hashes from the recipe's LoRAs
            const loraHashes = this.currentRecipe.loras
                .filter(lora => lora.hash)
                .map(lora => lora.hash.toLowerCase());

            if (loraHashes.length > 0) {
                // Store the LoRA hashes and recipe name in sessionStorage
                setSessionItem('recipe_to_lora_filterLoraHashes', JSON.stringify(loraHashes));
                setSessionItem('filterRecipeName', this.currentRecipe.title);
            }
        }

        // Navigate to the LoRAs page
        window.location.href = '/loras';
    }

    // New method to make LoRA items clickable
    setupLoraItemsClickable() {
        const loraItems = document.querySelectorAll('.recipe-lora-item:not(.checkpoint-item)');
        loraItems.forEach(item => {
            // Get the lora index from the data attribute
            const loraIndex = parseInt(item.dataset.loraIndex);

            item.addEventListener('click', (e) => {
                // If the click is on the reconnect container or badge, don't navigate
                if (e.target.closest('.lora-reconnect-container') ||
                    e.target.closest('.lora-download') ||
                    e.target.closest('.deleted-badge') ||
                    e.target.closest('.reconnect-tooltip')) {
                    return;
                }

                // Navigate to the LoRAs page with the specific LoRA index
                this.navigateToLorasPage(loraIndex);
            });
        });
    }
}

export { RecipeModal };
