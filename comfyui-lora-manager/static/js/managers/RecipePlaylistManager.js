// Recipe playlist ("実行リスト"): an ordered recipe list that can be queued
// into ComfyUI sequentially. Failures are skipped so one broken recipe never
// stops the run — ComfyUI's own queue serializes execution, and the user can
// cancel remaining runs from the ComfyUI UI.

import { fetchRecipeDetails } from '../api/recipeApi.js';
import { showToast } from '../utils/uiHelpers.js';
import { analyzeRecipeReplayCapability } from '../utils/recipeReplayCapability.js';
import {
    addPlaylistEntries,
    loadPlaylist,
    movePlaylistEntry,
    removePlaylistEntry,
    savePlaylist,
} from '../utils/recipePlaylistStore.js';

const STATUS_LABELS = {
    preparing: '準備中…',
    queued: '投入済み',
    skipped: 'スキップ',
};

export async function queuePlaylistEntries(entries, deps) {
    const {
        fetchRecipe,
        analyze,
        jsonFetch,
        uuid,
        onProgress = () => {},
    } = deps;

    const results = [];
    for (const entry of Array.isArray(entries) ? entries : []) {
        const result = {
            id: entry.id,
            title: entry.title,
            status: 'pending',
            reason: null,
            promptId: null,
        };
        results.push(result);
        onProgress(entry.id, 'preparing');
        try {
            const recipe = await fetchRecipe(entry.id);
            const analysis = await analyze(recipe);
            if (!analysis?.built || analysis.level === 'unavailable') {
                throw new Error(
                    (analysis?.reasons || []).join(' / ') || '再現不可レシピです'
                );
            }
            if (recipe.replay_manifest && analysis.audit?.ok !== true) {
                throw new Error('再現監査を通過できませんでした');
            }
            const built = analysis.built;
            const prepared = await jsonFetch('/api/lm/load-recipe-workflow', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: built.prompt,
                    source: built.source,
                    a1111_parameters: built.a1111Parameters,
                    a1111_checkpoint: built.a1111Checkpoint,
                    replay_manifest: built.replayManifest
                        ? {
                              schema: built.replayManifest.schema,
                              version: built.replayManifest.version,
                              manifest_hash: built.replayManifest.manifest_hash,
                          }
                        : null,
                    required_model_inputs:
                        analysis.audit?.required_model_inputs || [],
                    prepare_only: true,
                }),
            });
            if (!prepared?.success || !prepared?.prompt) {
                throw new Error(
                    prepared?.message ||
                        prepared?.error ||
                        'ワークフローを準備できませんでした'
                );
            }
            const promptId = uuid();
            const queued = await jsonFetch('/prompt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Comfy-Usage-Source': 'lora-manager-recipe-playlist',
                },
                body: JSON.stringify({
                    prompt: prepared.prompt,
                    prompt_id: promptId,
                    extra_data: {
                        lora_manager_recipe_playlist: {
                            schema: 'lora-manager.recipe-playlist',
                            version: 1,
                            recipe_id: entry.id,
                        },
                    },
                }),
            });
            result.promptId = queued?.prompt_id || promptId;
            result.status = 'queued';
            onProgress(entry.id, 'queued');
        } catch (error) {
            // Skip and continue — the remaining entries still get queued.
            result.status = 'skipped';
            result.reason = error?.message || String(error);
            onProgress(entry.id, 'skipped', result.reason);
        }
    }
    return results;
}

async function defaultJsonFetch(url, options) {
    const response = await fetch(url, options);
    let payload = null;
    try {
        payload = await response.json();
    } catch {
        payload = null;
    }
    if (!response.ok) {
        throw new Error(
            payload?.message ||
                payload?.error ||
                `リクエストに失敗しました (${response.status})`
        );
    }
    return payload;
}

function defaultUuid() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `playlist-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

class RecipePlaylistManager {
    constructor() {
        this.modal = null;
        this.listElement = null;
        this.runButton = null;
        this.running = false;
        this.statusById = new Map();
    }

    addRecipes(recipes) {
        const { entries, added, skipped } = addPlaylistEntries(
            loadPlaylist(),
            recipes
        );
        savePlaylist(entries);
        this.statusById.clear();
        this.updateBadge();
        this.renderList();
        const skippedNote = skipped > 0 ? `（重複などスキップ${skipped}件)` : '';
        showToast(
            'toast.recipes.playlistAdded',
            { added, skipped },
            added > 0 ? 'success' : 'info',
            `実行リストへ${added}件追加しました${skippedNote}`
        );
    }

    updateBadge() {
        const badge = document.getElementById('recipePlaylistBadge');
        if (!badge) return;
        const count = loadPlaylist().length;
        badge.textContent = count > 0 ? String(count) : '';
        badge.style.display = count > 0 ? '' : 'none';
    }

    open() {
        this.ensureModal();
        this.renderList();
        this.modal.style.display = 'flex';
    }

    close() {
        if (this.modal) this.modal.style.display = 'none';
    }

    ensureModal() {
        if (this.modal) return;

        const modal = document.createElement('div');
        modal.id = 'recipePlaylistModal';
        modal.className = 'modal recipe-playlist-modal';
        modal.style.display = 'none';

        const content = document.createElement('div');
        content.className = 'modal-content recipe-playlist-content';

        const closeButton = document.createElement('button');
        closeButton.className = 'close';
        closeButton.innerHTML = '&times;';
        closeButton.addEventListener('click', () => this.close());

        const header = document.createElement('h2');
        header.textContent = '実行リスト';

        const hint = document.createElement('p');
        hint.className = 'recipe-playlist-hint';
        hint.textContent =
            '上から順にComfyUIキューへ投入します。準備に失敗したレシピはスキップして続行します（停止したい場合はComfyUI側でキューを操作してください）。';

        const list = document.createElement('div');
        list.className = 'recipe-playlist-list';

        const actions = document.createElement('div');
        actions.className = 'recipe-playlist-actions';

        const runButton = document.createElement('button');
        runButton.className = 'primary-btn';
        runButton.innerHTML = '<i class="fas fa-play"></i> 順次投入';
        runButton.addEventListener('click', () => this.run());

        const clearButton = document.createElement('button');
        clearButton.className = 'secondary-btn';
        clearButton.innerHTML = '<i class="fas fa-trash"></i> クリア';
        clearButton.addEventListener('click', () => {
            savePlaylist([]);
            this.statusById.clear();
            this.updateBadge();
            this.renderList();
        });

        actions.append(runButton, clearButton);
        content.append(closeButton, header, hint, list, actions);
        modal.append(content);
        modal.addEventListener('click', event => {
            if (event.target === modal) this.close();
        });
        document.body.append(modal);

        this.modal = modal;
        this.listElement = list;
        this.runButton = runButton;
    }

    renderList() {
        if (!this.listElement) return;
        const entries = loadPlaylist();
        this.listElement.innerHTML = '';

        if (entries.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'recipe-playlist-empty';
            empty.textContent =
                '実行リストは空です。レシピの右クリックメニュー「実行リストへ追加」から追加してください。';
            this.listElement.append(empty);
            if (this.runButton) this.runButton.disabled = true;
            return;
        }
        if (this.runButton) this.runButton.disabled = this.running;

        entries.forEach((entry, index) => {
            const row = document.createElement('div');
            row.className = 'recipe-playlist-row';
            row.dataset.id = entry.id;

            const order = document.createElement('span');
            order.className = 'recipe-playlist-order';
            order.textContent = String(index + 1);

            const info = document.createElement('div');
            info.className = 'recipe-playlist-info';
            const title = document.createElement('div');
            title.className = 'recipe-playlist-title';
            title.textContent = entry.title;
            info.append(title);
            if (entry.checkpointName) {
                const checkpoint = document.createElement('div');
                checkpoint.className = 'recipe-playlist-checkpoint';
                checkpoint.textContent = entry.checkpointName;
                info.append(checkpoint);
            }

            const status = document.createElement('span');
            status.className = 'recipe-playlist-status';
            const statusInfo = this.statusById.get(entry.id);
            if (statusInfo) {
                status.textContent = STATUS_LABELS[statusInfo.status] || '';
                status.title = statusInfo.reason || '';
                status.dataset.status = statusInfo.status;
            }

            const controls = document.createElement('div');
            controls.className = 'recipe-playlist-controls';
            controls.append(
                this.controlButton('fa-arrow-up', '上へ', () =>
                    this.moveEntry(entry.id, -1)
                ),
                this.controlButton('fa-arrow-down', '下へ', () =>
                    this.moveEntry(entry.id, 1)
                ),
                this.controlButton('fa-times', 'リストから削除', () =>
                    this.removeEntry(entry.id)
                )
            );

            row.append(order, info, status, controls);
            this.listElement.append(row);
        });
    }

    controlButton(iconClass, label, onClick) {
        const button = document.createElement('button');
        button.className = 'recipe-playlist-control';
        button.title = label;
        button.innerHTML = `<i class="fas ${iconClass}"></i>`;
        button.addEventListener('click', onClick);
        return button;
    }

    moveEntry(id, offset) {
        savePlaylist(movePlaylistEntry(loadPlaylist(), id, offset));
        this.renderList();
    }

    removeEntry(id) {
        savePlaylist(removePlaylistEntry(loadPlaylist(), id));
        this.statusById.delete(id);
        this.updateBadge();
        this.renderList();
    }

    async run() {
        if (this.running) return;
        const entries = loadPlaylist();
        if (entries.length === 0) return;

        this.running = true;
        this.statusById.clear();
        if (this.runButton) this.runButton.disabled = true;
        try {
            const results = await queuePlaylistEntries(entries, {
                fetchRecipe: id => fetchRecipeDetails(id, { variant: 'active' }),
                analyze: recipe => analyzeRecipeReplayCapability(recipe),
                jsonFetch: defaultJsonFetch,
                uuid: defaultUuid,
                onProgress: (id, status, reason) => {
                    this.statusById.set(id, { status, reason: reason || '' });
                    this.renderList();
                },
            });
            const queued = results.filter(item => item.status === 'queued').length;
            const skipped = results.filter(item => item.status === 'skipped').length;
            showToast(
                'toast.recipes.playlistQueued',
                { queued, skipped },
                skipped > 0 ? 'warning' : 'success',
                `実行リスト: ${queued}件をキューへ投入しました` +
                    (skipped > 0 ? `（スキップ${skipped}件）` : '')
            );
        } finally {
            this.running = false;
            if (this.runButton) this.runButton.disabled = false;
            this.renderList();
        }
    }
}

export const recipePlaylistManager = new RecipePlaylistManager();
