import { beforeEach, describe, expect, it, vi } from 'vitest';

const clients = vi.hoisted(() => ({
    loras: { fetchModelRoots: vi.fn() },
    checkpoints: { fetchModelRoots: vi.fn() },
    embeddings: { fetchModelRoots: vi.fn() },
}));

vi.mock('../../../static/js/api/modelApiFactory.js', () => ({
    getModelApiClient: type => clients[type],
}));
vi.mock('../../../static/js/utils/uiHelpers.js', () => ({ showToast: vi.fn() }));
vi.mock('../../../static/js/utils/i18nHelpers.js', () => ({
    translate: (_key, _params, fallback) => fallback,
}));
vi.mock('../../../static/js/state/index.js', () => ({
    state: {
        loadingManager: {
            showDownloadProgress: vi.fn(() => vi.fn()),
            setStatus: vi.fn(),
            hide: vi.fn(),
        },
        virtualScroller: null,
        global: {
            settings: {
                default_lora_root: 'L:/loras',
                default_checkpoint_root: 'C:/checkpoints',
                default_embedding_root: 'E:/embeddings',
            },
        },
    },
}));
vi.mock('../../../static/js/managers/ModalManager.js', () => ({
    modalManager: { showModal: vi.fn(), closeModal: vi.fn() },
}));

import { MODEL_TYPES } from '../../../static/js/api/apiConfig.js';
import { BulkMissingLoraDownloadManager } from '../../../static/js/managers/BulkMissingLoraDownloadManager.js';

describe('BulkMissingLoraDownloadManager recipe resources', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        clients.loras.fetchModelRoots.mockResolvedValue({ roots: ['L:/loras'] });
        clients.checkpoints.fetchModelRoots.mockResolvedValue({ roots: ['C:/checkpoints'] });
        clients.embeddings.fetchModelRoots.mockResolvedValue({ roots: ['E:/embeddings'] });
        clients.loras.downloadModel = vi.fn().mockResolvedValue({ success: true });
        global.fetch = vi.fn().mockResolvedValue({ ok: true });
        global.WebSocket = class {
            static OPEN = 1;
            constructor() {
                this.readyState = 1;
                queueMicrotask(() => this.onopen?.());
            }
            close() {}
        };
    });

    it('collects and deduplicates missing LoRAs, checkpoints, and embeddings', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([
            {
                id: 'recipe-1',
                checkpoint: { id: 10, modelId: 1, name: 'Base', inLibrary: false },
                loras: [{ modelVersionId: 20, modelId: 2, modelName: 'Style', inLibrary: false }],
                embeddings: [{ modelVersionId: 30, modelId: 3, modelName: 'Easy Negative', inLibrary: false }],
            },
            {
                id: 'recipe-2',
                embeddings: [{ modelVersionId: 30, modelId: 3, modelName: 'Easy Negative', inLibrary: false }],
            },
        ]);

        expect(stats.uniqueCount).toBe(3);
        expect(stats.totalMissingCount).toBe(4);
        expect(stats.uniqueResources.map(resource => resource.type)).toEqual([
            MODEL_TYPES.CHECKPOINT,
            MODEL_TYPES.LORA,
            MODEL_TYPES.EMBEDDING,
        ]);
    });

    it('selects the exact non-primary recipe file from its short hash', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([{
            checkpoint: {
                id: 235605,
                modelId: 97744,
                hash: '7eb8672531',
                inLibrary: false,
                civitai: {
                    files: [
                        {
                            id: 181972,
                            primary: true,
                            type: 'Model',
                            metadata: { format: 'SafeTensor', size: 'pruned', fp: 'fp16' },
                            hashes: { AutoV2: 'C060E0173D', SHA256: 'C060E0173D2E' },
                        },
                        {
                            id: 181973,
                            primary: false,
                            type: 'Model',
                            metadata: { format: 'SafeTensor', size: 'full', fp: 'fp32' },
                            hashes: { AutoV2: '7EB8672531', SHA256: '7EB8672531ABCDEF' },
                        },
                    ],
                },
            },
        }]);

        expect(stats.uniqueResources[0].fileParams).toEqual({
            fileId: 181973,
            sha256: '7EB8672531ABCDEF',
            type: 'Model',
            format: 'SafeTensor',
            size: 'full',
            fp: 'fp32',
            isPrimary: false,
        });
    });

    it('forwards a recipe hash even when expanded Civitai file metadata is absent', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([{
            checkpoint: {
                id: 2967640,
                modelId: 2642932,
                hash: '09c48368ae410664d2cc1d6d163edf79a75ca4101529739afe786dd5456f884d',
                inLibrary: false,
            },
        }]);

        expect(stats.uniqueResources[0].fileParams).toEqual({
            sha256: '09c48368ae410664d2cc1d6d163edf79a75ca4101529739afe786dd5456f884d',
        });
    });

    it('collects deleted resources with IDs for CivArchive recovery', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([{
            checkpoint: {
                id: 665047,
                modelId: 153568,
                name: 'Real Dream SDXL Pony 9',
                isDeleted: true,
                inLibrary: false,
            },
        }]);

        expect(stats.uniqueCount).toBe(1);
        expect(stats.uniqueResources[0]).toMatchObject({
            type: MODEL_TYPES.CHECKPOINT,
            id: 665047,
            modelId: 153568,
            metadataSource: 'civarchive',
        });
    });

    it('skips deleted optional resources that have no recipe file identity', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([{
            loras: [{
                modelVersionId: 250592,
                modelId: 222153,
                isDeleted: true,
                inLibrary: false,
            }],
            embeddings: [{
                modelVersionId: 999,
                modelId: 888,
                isDeleted: true,
                inLibrary: false,
            }],
        }]);

        expect(stats.uniqueCount).toBe(0);
        expect(stats.totalMissingCount).toBe(0);
    });

    it('keeps a deleted LoRA recoverable when the recipe has a hash', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([{
            loras: [{
                modelVersionId: 13304,
                modelId: 8214,
                hash: 'acd58eb244848f0ffddc647435ddd983eec5ceeb10c70baa6ca333a089715b69',
                isDeleted: true,
                inLibrary: false,
            }],
        }]);

        expect(stats.uniqueCount).toBe(1);
        expect(stats.uniqueResources[0]).toMatchObject({
            type: MODEL_TYPES.LORA,
            id: 13304,
            modelId: 8214,
            metadataSource: 'civarchive',
        });
    });

    it('does not repeatedly queue an API-only checkpoint placeholder with no file identity', () => {
        const manager = new BulkMissingLoraDownloadManager();
        const stats = manager.collectMissingResources([{
            checkpoint: {
                id: 1088507,
                modelId: 618692,
                file_name: 'FLUX',
                inLibrary: false,
            },
        }]);

        expect(stats.uniqueCount).toBe(0);
        expect(stats.totalMissingCount).toBe(0);
    });

    it('uses the configured embedding root and embedding API client', async () => {
        const manager = new BulkMissingLoraDownloadManager();

        await expect(manager.getModelRoot(MODEL_TYPES.EMBEDDING)).resolves.toBe('E:/embeddings');
        expect(clients.embeddings.fetchModelRoots).toHaveBeenCalledOnce();
        expect(clients.loras.fetchModelRoots).not.toHaveBeenCalled();
    });

    it('forwards the CivArchive source for archived recipe resources', async () => {
        const manager = new BulkMissingLoraDownloadManager();
        manager.pendingRecipes = [];

        await expect(manager.executeDownload([{
            type: MODEL_TYPES.LORA,
            id: 13304,
            modelId: 8214,
            modelName: 'Firekeeper',
            metadataSource: 'civarchive',
        }])).resolves.toBe(true);

        expect(clients.loras.downloadModel).toHaveBeenCalledWith(
            8214,
            13304,
            'L:/loras',
            '',
            true,
            expect.any(String),
            'civarchive',
            null,
        );
    });

    it('forwards exact recipe file parameters to the download API', async () => {
        const manager = new BulkMissingLoraDownloadManager();
        manager.pendingRecipes = [];
        const fileParams = {
            fileId: 181973,
            sha256: '7EB8672531ABCDEF',
            type: 'Model',
            format: 'SafeTensor',
            size: 'full',
            fp: 'fp32',
            isPrimary: false,
        };

        await expect(manager.executeDownload([{
            type: MODEL_TYPES.LORA,
            id: 235605,
            modelId: 97744,
            modelName: 'Exact variant',
            fileParams,
        }])).resolves.toBe(true);

        expect(clients.loras.downloadModel).toHaveBeenCalledWith(
            97744,
            235605,
            'L:/loras',
            '',
            true,
            expect.any(String),
            null,
            fileParams,
        );
    });

    it('treats an already installed recipe resource as complete and refreshes its state', async () => {
        clients.loras.downloadModel.mockResolvedValue({
            success: false,
            error: 'Model version already exists in lora library',
        });
        const manager = new BulkMissingLoraDownloadManager();
        manager.pendingRecipes = [];

        await expect(manager.executeDownload([{
            type: MODEL_TYPES.LORA,
            id: 13304,
            modelId: 8214,
            modelName: 'Firekeeper',
            metadataSource: 'civarchive',
        }])).resolves.toBe(true);
    });
});
