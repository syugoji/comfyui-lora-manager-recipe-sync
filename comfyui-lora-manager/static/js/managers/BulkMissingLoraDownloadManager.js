import { showToast } from '../utils/uiHelpers.js';
import { translate } from '../utils/i18nHelpers.js';
import { getModelApiClient } from '../api/modelApiFactory.js';
import { MODEL_TYPES } from '../api/apiConfig.js';
import { state } from '../state/index.js';
import { modalManager } from './ModalManager.js';

const ALREADY_IN_LIBRARY_ERROR = /^Model version already exists in (?:lora|checkpoint|embedding) library$/i;

function normalizeHash(value) {
    return String(value || '').trim().toLowerCase();
}

export function getRecipeFileParams(resource) {
    const files = resource?.civitai?.files;
    const targetFileId = resource.fileId ?? resource.file_id;
    const targetHash = normalizeHash(resource.hash || resource.sha256);
    if (!Array.isArray(files) || files.length === 0) {
        return targetHash.length >= 8 ? { sha256: targetHash } : null;
    }
    let selectedFile = null;

    if (targetFileId !== undefined && targetFileId !== null) {
        selectedFile = files.find(file => String(file?.id) === String(targetFileId)) || null;
    }

    if (!selectedFile && targetHash.length >= 8) {
        selectedFile = files.find(file => Object.values(file?.hashes || {}).some(value => {
            const candidate = normalizeHash(value);
            return candidate === targetHash
                || candidate.startsWith(targetHash)
                || targetHash.startsWith(candidate);
        })) || null;
    }

    if (!selectedFile) {
        return targetHash.length >= 8 ? { sha256: targetHash } : null;
    }

    const metadata = selectedFile.metadata || {};
    return Object.fromEntries(Object.entries({
        fileId: selectedFile.id,
        sha256: selectedFile.hashes?.SHA256,
        type: selectedFile.type,
        format: metadata.format,
        size: metadata.size,
        fp: metadata.fp,
        isPrimary: selectedFile.primary === true,
    }).filter(([, value]) => value !== undefined && value !== null && value !== ''));
}

/**
 * Manager for downloading missing LoRAs for selected recipes in bulk
 */
export class BulkMissingLoraDownloadManager {
    constructor() {
        this.loraApiClient = getModelApiClient(MODEL_TYPES.LORA);
        this.checkpointApiClient = getModelApiClient(MODEL_TYPES.CHECKPOINT);
        this.embeddingApiClient = getModelApiClient(MODEL_TYPES.EMBEDDING);
        this.pendingLoras = [];
        this.pendingResources = [];
        this.pendingRecipes = [];
        this.downloadCompletion = null;
        this.downloadCompletionResolve = null;
    }

    /**
     * Collect missing LoRAs from selected recipes with deduplication
     * @param {Array} selectedRecipes - Array of selected recipe objects
     * @returns {Object} - Object containing unique missing LoRAs and statistics
     */
    collectMissingLoras(selectedRecipes) {
        const uniqueLoras = new Map(); // key: hash or modelVersionId, value: lora object
        const missingLorasByRecipe = new Map();
        let totalMissingCount = 0;

        selectedRecipes.forEach(recipe => {
            const missingLoras = [];
            
            if (recipe.loras && Array.isArray(recipe.loras)) {
                recipe.loras.forEach(lora => {
                    // Only include LoRAs not in library and not deleted
                    if (!lora.inLibrary && !lora.isDeleted) {
                        const uniqueKey = lora.hash || lora.id || lora.modelVersionId;
                        
                        if (uniqueKey && !uniqueLoras.has(uniqueKey)) {
                            // Store the LoRA info
                            uniqueLoras.set(uniqueKey, {
                                ...lora,
                                type: MODEL_TYPES.LORA,
                                modelId: lora.modelId || lora.model_id,
                                id: lora.id || lora.modelVersionId,
                            });
                        }
                        
                        missingLoras.push(lora);
                        totalMissingCount++;
                    }
                });
            }
            
            if (missingLoras.length > 0) {
                missingLorasByRecipe.set(recipe.id || recipe.file_path, {
                    recipe,
                    missingLoras
                });
            }
        });

        return {
            uniqueLoras: Array.from(uniqueLoras.values()),
            uniqueCount: uniqueLoras.size,
            totalMissingCount,
            missingLorasByRecipe
        };
    }

    /**
     * Collect missing checkpoints, LoRAs, and embeddings from selected recipes. Resources
     * are deduplicated by type and version so the same file is never queued twice.
     */
    collectMissingResources(selectedRecipes) {
        const resources = new Map();
        let totalMissingCount = 0;

        const addResource = (resource, type) => {
            if (!resource || resource.inLibrary) return;

            const modelId = resource.modelId || resource.model_id || resource.civitai?.modelId;
            const versionId = resource.id || resource.modelVersionId || resource.civitai?.id;
            const fileParams = getRecipeFileParams(resource);
            const fileIdentity = fileParams?.sha256 || fileParams?.fileId;
            const filename = String(resource.file_name || resource.filename || '')
                .replaceAll('\\', '/').split('/').at(-1) || '';
            const genericPlaceholder = /^(?:flux(?:1)?|model|checkpoint|unet|unknown)(?:\.safetensors)?$/i
                .test(filename.trim());
            const hasStrongFileIdentity = Boolean(
                fileIdentity
                || resource.hash
                || resource.fileId
                || resource.file_id
                || (Array.isArray(resource.civitai?.files) && resource.civitai.files.length > 0)
            );
            if (type === MODEL_TYPES.CHECKPOINT && genericPlaceholder && !hasStrongFileIdentity) {
                // Civitai's API-only generators can expose a model/version ID but no
                // downloadable file. Re-queueing their placeholder name (for example
                // "FLUX") can never satisfy ComfyUI and causes an endless prompt.
                return;
            }
            const hasRecipeFileIdentity = Boolean(
                fileIdentity
                || resource.hash
                || resource.file_name
                || resource.filename
            );
            if (
                resource.isDeleted
                && type !== MODEL_TYPES.CHECKPOINT
                && !hasRecipeFileIdentity
            ) {
                // A deleted optional resource with only a remote version ID
                // cannot be verified against the recipe. Archive mirrors may
                // serve a different file under that ID, so do not ask for the
                // same unverifiable download on every replay.
                return;
            }
            const uniqueKey = `${type}:${fileIdentity || resource.hash || versionId || modelId}`;
            if (!versionId && !modelId) return;

            totalMissingCount++;
            if (!resources.has(uniqueKey)) {
                resources.set(uniqueKey, {
                    ...resource,
                    type,
                    modelId,
                    id: versionId,
                    name: resource.name || resource.modelName || resource.civitai?.name,
                    version: resource.version || resource.modelVersionName || resource.civitai?.name,
                    fileParams,
                    // Deleted Civitai resources may still be available through
                    // CivArchive mirrors. Let the downloader verify availability
                    // instead of replaying an invalid placeholder model name.
                    metadataSource: resource.metadataSource
                        || (resource.isDeleted ? 'civarchive' : undefined),
                });
            }
        };

        selectedRecipes.forEach(recipe => {
            addResource(recipe.checkpoint, MODEL_TYPES.CHECKPOINT);
            (recipe.loras || []).forEach(lora => addResource(lora, MODEL_TYPES.LORA));
            (recipe.embeddings || []).forEach(embedding => addResource(embedding, MODEL_TYPES.EMBEDDING));
        });

        return {
            uniqueResources: Array.from(resources.values()),
            uniqueCount: resources.size,
            totalMissingCount,
        };
    }

    /**
     * Show confirmation modal for downloading missing LoRAs
     * @param {Object} stats - Statistics about missing LoRAs
     * @returns {Promise<boolean>} - Whether user confirmed
     */
    async showConfirmationModal(stats) {
        const { uniqueCount, totalMissingCount } = stats;
        const resources = stats.uniqueResources || stats.uniqueLoras || [];

        if (uniqueCount === 0) {
            showToast('toast.recipes.noMissingLoras', {}, 'info');
            return false;
        }

        // Store pending data for confirmation
        this.pendingResources = resources;
        this.pendingLoras = resources;

        // Update modal content
        const messageEl = document.getElementById('bulkDownloadMissingLorasMessage');
        const listEl = document.getElementById('bulkDownloadMissingLorasList');
        const confirmBtn = document.getElementById('bulkDownloadMissingLorasConfirmBtn');
        const titleEl = document.getElementById('bulkDownloadMissingResourcesTitle');
        const previewTitleEl = document.getElementById('bulkDownloadMissingResourcesPreviewTitle');

        if (titleEl) {
            titleEl.textContent = translate('modals.bulkDownloadMissingResources.title', {}, 'Download missing recipe resources');
        }
        if (previewTitleEl) {
            previewTitleEl.textContent = translate('modals.bulkDownloadMissingResources.previewTitle', {}, 'Resources to download:');
        }

        if (messageEl) {
            messageEl.textContent = translate('modals.bulkDownloadMissingResources.message', {
                uniqueCount, 
                totalCount: totalMissingCount 
            }, `Found ${uniqueCount} unique missing resources (from ${totalMissingCount} total across selected recipes).`);
        }

        if (listEl) {
            listEl.innerHTML = resources.slice(0, 10).map(resource => `
                <li>
                    <span class="lora-name">${this.getResourceTypeLabel(resource.type)}: ${resource.name || resource.modelName || resource.file_name || 'Unknown'}</span>
                    ${resource.version ? `<span class="lora-version">${resource.version}</span>` : ''}
                </li>
            `).join('') + 
            (resources.length > 10 ? `
                <li class="more-items">${translate('modals.bulkDownloadMissingLoras.moreItems', { count: resources.length - 10 }, `...and ${resources.length - 10} more`)}</li>
            ` : '');
        }

        if (confirmBtn) {
            confirmBtn.innerHTML = `
                <i class="fas fa-download"></i>
                ${translate('modals.bulkDownloadMissingResources.downloadButton', { count: uniqueCount }, `Download ${uniqueCount} resource(s)`)}
            `;
        }

        // Show modal
        modalManager.showModal('bulkDownloadMissingLorasModal');

        this.downloadCompletion = new Promise((resolve) => {
            this.downloadCompletionResolve = resolve;
        });
        
        // Return a promise that will be resolved when user confirms or cancels
        return new Promise((resolve) => {
            this.confirmResolve = resolve;
        });
    }

    /**
     * Called when user confirms download in modal
     */
    async confirmDownload() {
        modalManager.closeModal('bulkDownloadMissingLorasModal');
        const resources = [...this.pendingResources];

        if (this.confirmResolve) {
            this.confirmResolve(true);
            this.confirmResolve = null;
        }

        let result = false;
        try {
            result = await this.executeDownload(resources);
        } catch (error) {
            console.error('Recipe resource download failed:', error);
            state.loadingManager?.hide();
            showToast(
                'toast.recipes.resourcesDownloadFailed',
                { message: error.message },
                'error',
                `Recipe resource download failed: ${error.message}`
            );
        } finally {
            if (this.downloadCompletionResolve) {
                this.downloadCompletionResolve(result);
                this.downloadCompletionResolve = null;
            }
            this.pendingLoras = [];
            this.pendingResources = [];
        }
    }

    /**
     * Close the confirmation modal without starting a download.
     */
    cancelDownload() {
        modalManager.closeModal('bulkDownloadMissingLorasModal');
        if (this.confirmResolve) {
            this.confirmResolve(false);
            this.confirmResolve = null;
        }
        if (this.downloadCompletionResolve) {
            this.downloadCompletionResolve(false);
            this.downloadCompletionResolve = null;
        }
        this.pendingLoras = [];
        this.pendingResources = [];
    }

    /**
     * Download missing LoRAs for selected recipes
     * @param {Array} selectedRecipes - Array of selected recipe objects
     */
    async downloadMissingLoras(selectedRecipes) {
        if (!selectedRecipes || selectedRecipes.length === 0) {
            showToast('toast.recipes.noRecipesSelected', {}, 'warning');
            return;
        }

        // Store selected recipes
        this.pendingRecipes = selectedRecipes;

        // Collect missing LoRAs with deduplication
        const stats = this.collectMissingLoras(selectedRecipes);
        
        if (stats.uniqueCount === 0) {
            showToast('toast.recipes.noMissingLorasInSelection', {}, 'info');
            return;
        }

        // Show confirmation modal
        const confirmed = await this.showConfirmationModal(stats);
        if (!confirmed) {
            return false;
        }
        return await this.downloadCompletion;
    }

    /**
     * Download every missing checkpoint, LoRA, and embedding referenced by the selected recipes.
     * @param {Array} selectedRecipes - Array of selected recipe objects
     */
    async downloadMissingResources(selectedRecipes) {
        if (!selectedRecipes || selectedRecipes.length === 0) {
            showToast('toast.recipes.noRecipesSelected', {}, 'warning');
            return;
        }

        this.pendingRecipes = selectedRecipes;
        const stats = this.collectMissingResources(selectedRecipes);
        if (stats.uniqueCount === 0) {
            showToast('toast.recipes.noMissingResourcesInSelection', {}, 'info', 'No missing checkpoints, LoRAs, or embeddings found in the selected recipes.');
            return;
        }

        const confirmed = await this.showConfirmationModal(stats);
        if (!confirmed) return false;
        return await this.downloadCompletion;
    }

    /**
     * Execute the download process
     * @param {Array} resourcesToDownload - Array of unique recipe resources to download
     */
    async executeDownload(resourcesToDownload) {
        const totalResources = resourcesToDownload.length;
        const roots = {};

        for (const type of new Set(resourcesToDownload.map(resource => resource.type))) {
            roots[type] = await this.getModelRoot(type);
            if (!roots[type]) {
                showToast('toast.recipes.noModelRootConfigured', { type }, 'error', `No ${type} root directory is configured.`);
                return false;
            }
        }

        // Generate batch download ID
        const batchDownloadId = Date.now().toString();
        
        // Use default paths
        const useDefaultPaths = true;

        // Set up WebSocket for progress updates
        const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        const ws = new WebSocket(`${wsProtocol}${window.location.host}/ws/download-progress?id=${batchDownloadId}`);

        // Show download progress UI
        const loadingManager = state.loadingManager;
        const updateProgress = loadingManager.showDownloadProgress(totalResources);

        let completedDownloads = 0;
        let failedDownloads = 0;
        let currentResourceProgress = 0;

        // Set up WebSocket message handler
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            // Handle download ID confirmation
            if (data.type === 'download_id') {
                console.log(`Connected to batch download progress with ID: ${data.download_id}`);
                return;
            }

            // Process progress updates
            if (data.status === 'progress' && data.download_id && data.download_id.startsWith(batchDownloadId)) {
                currentResourceProgress = data.progress;
                
                const currentResource = resourcesToDownload[completedDownloads + failedDownloads];
                const resourceName = currentResource ? (currentResource.name || currentResource.file_name || 'Unknown') : '';

                const metrics = {
                    bytesDownloaded: data.bytes_downloaded,
                    totalBytes: data.total_bytes,
                    bytesPerSecond: data.bytes_per_second
                };

                updateProgress(currentResourceProgress, completedDownloads, resourceName, metrics);

                // Update status message
                if (currentResourceProgress < 3) {
                    loadingManager.setStatus(
                        translate('recipes.controls.import.startingDownload', 
                            { current: completedDownloads + failedDownloads + 1, total: totalResources },
                            `Starting download ${completedDownloads + failedDownloads + 1}/${totalResources}`
                        )
                    );
                } else if (currentResourceProgress > 3 && currentResourceProgress < 100) {
                    loadingManager.setStatus(
                        translate('recipes.controls.import.downloadingResources', {}, `Downloading recipe resources...`)
                    );
                }
            }
        };

        // Wait for WebSocket to connect
        await new Promise((resolve, reject) => {
            ws.onopen = resolve;
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
        });

        // Download each resource sequentially so the shared progress UI stays accurate.
        for (let i = 0; i < resourcesToDownload.length; i++) {
            const resource = resourcesToDownload[i];
            
            currentResourceProgress = 0;
            
            loadingManager.setStatus(
                translate('recipes.controls.import.startingDownload', 
                    { current: i + 1, total: totalResources },
                    `Starting download ${i + 1}/${totalResources}`
                )
            );
            updateProgress(0, completedDownloads, resource.name || resource.file_name || 'Unknown');

            try {
                const modelId = resource.modelId || resource.model_id;
                const versionId = resource.id || resource.modelVersionId;

                if (!modelId && !versionId) {
                    console.warn('Skipping recipe resource without model/version ID:', resource);
                    failedDownloads++;
                    continue;
                }

                const apiClient = resource.type === MODEL_TYPES.CHECKPOINT
                    ? this.checkpointApiClient
                    : resource.type === MODEL_TYPES.EMBEDDING
                        ? this.embeddingApiClient
                        : this.loraApiClient;
                const response = await apiClient.downloadModel(
                    modelId,
                    versionId,
                    roots[resource.type],
                    '', // Empty relative path, use default paths
                    useDefaultPaths,
                    batchDownloadId,
                    resource.metadataSource === 'civarchive' ? 'civarchive' : null,
                    resource.fileParams || null
                );

                const alreadyInLibrary =
                    !response.success && ALREADY_IN_LIBRARY_ERROR.test(response.error || '');

                if (!response.success && !alreadyInLibrary) {
                    console.error(`Failed to download recipe resource ${resource.name || resource.file_name}: ${response.error}`);
                    failedDownloads++;
                } else {
                    if (alreadyInLibrary) {
                        console.info(`Recipe resource already exists locally: ${resource.name || resource.file_name}`);
                    }
                    completedDownloads++;
                    updateProgress(100, completedDownloads, '');
                }
            } catch (error) {
                console.error(`Error downloading recipe resource ${resource.name || resource.file_name}:`, error);
                failedDownloads++;
            }
        }

        // Close WebSocket
        ws.close();

        // Hide loading UI
        loadingManager.hide();

        // Show completion message
        if (failedDownloads === 0) {
            showToast('toast.recipes.allResourcesDownloadSuccessful', { count: completedDownloads }, 'success', `Downloaded ${completedDownloads} recipe resource(s).`);
        } else {
            showToast('toast.recipes.resourcesDownloadPartialSuccess', {
                completed: completedDownloads,
                total: totalResources
            }, 'warning', `Downloaded ${completedDownloads} of ${totalResources} recipe resource(s).`);
        }

        try {
            const endpointByType = {
                [MODEL_TYPES.LORA]: '/api/lm/loras/scan',
                [MODEL_TYPES.CHECKPOINT]: '/api/lm/checkpoints/scan',
                [MODEL_TYPES.EMBEDDING]: '/api/lm/embeddings/scan',
            };
            const endpoints = [...new Set(resourcesToDownload
                .map(resource => endpointByType[resource.type])
                .filter(Boolean))];
            for (const endpoint of endpoints) {
                const response = await fetch(endpoint);
                if (!response.ok) {
                    throw new Error(`Library scan failed (${response.status})`);
                }
            }

            const recipeScan = await fetch('/api/lm/recipes/scan');
            if (!recipeScan.ok) {
                throw new Error(`Recipe scan failed (${recipeScan.status})`);
            }
        } catch (error) {
            console.warn('Failed to rescan recipes after resource download:', error);
        }

        // Update each affected recipe card with fresh data (LoRA inLibrary flags changed)
        if (state.virtualScroller) {
            const { extractRecipeId } = await import('../api/recipeApi.js');
            for (const recipe of this.pendingRecipes) {
                const recipeId = extractRecipeId(recipe.file_path);
                if (!recipeId) continue;
                try {
                    const detailRes = await fetch(`/api/lm/recipe/${encodeURIComponent(recipeId)}`);
                    if (detailRes.ok) {
                        const updated = await detailRes.json();
                        state.virtualScroller.updateSingleItem(recipe.file_path, updated);
                    }
                } catch (e) {
                        console.warn('Failed to update recipe card after recipe resource download:', e);
                }
            }
        }

        return failedDownloads === 0;
    }

    /**
     * Get the configured root directory for a recipe resource type.
     * @param {string} modelType - LoRA or checkpoint model type
     * @returns {Promise<string|null>} - Root directory or null
     */
    async getModelRoot(modelType) {
        try {
            const apiClient = modelType === MODEL_TYPES.CHECKPOINT
                ? this.checkpointApiClient
                : modelType === MODEL_TYPES.EMBEDDING
                    ? this.embeddingApiClient
                    : this.loraApiClient;
            const rootsData = await apiClient.fetchModelRoots();
            
            if (!rootsData || !rootsData.roots || rootsData.roots.length === 0) {
                console.error(`No ${modelType} roots available`);
                return null;
            }

            const defaultRootKey = modelType === MODEL_TYPES.CHECKPOINT
                ? 'default_checkpoint_root'
                : modelType === MODEL_TYPES.EMBEDDING
                    ? 'default_embedding_root'
                    : 'default_lora_root';
            const defaultRoot = state.global?.settings?.[defaultRootKey];
            
            if (defaultRoot && rootsData.roots.includes(defaultRoot)) {
                return defaultRoot;
            }
            
            return rootsData.roots[0];
            
        } catch (error) {
            console.error(`Error getting ${modelType} root:`, error);
            return null;
        }
    }

    async getLoraRoot() {
        return this.getModelRoot(MODEL_TYPES.LORA);
    }

    getResourceTypeLabel(modelType) {
        if (modelType === MODEL_TYPES.CHECKPOINT) return 'Checkpoint';
        if (modelType === MODEL_TYPES.EMBEDDING) return 'Embedding';
        return 'LoRA';
    }
}

// Export singleton instance
export const bulkMissingLoraDownloadManager = new BulkMissingLoraDownloadManager();

// Make available globally for HTML onclick handlers
if (typeof window !== 'undefined') {
    window.bulkMissingLoraDownloadManager = bulkMissingLoraDownloadManager;
}
