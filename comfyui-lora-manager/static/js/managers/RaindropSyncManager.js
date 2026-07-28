import { modalManager } from './ModalManager.js';
import { showToast } from '../utils/uiHelpers.js';
import { translate } from '../utils/i18nHelpers.js';
import { state } from '../state/index.js';

const POLL_INTERVAL_MS = 1500;

/**
 * Raindrop → Civitai → レシピ の同期を画面から起動する。
 *
 * 同期の実体はサーバ側が起動する別プロセス（配布ツリーの civitai-recipe-sync/）。
 * ここは開始・進捗の取得・中断の3つのAPIを叩くだけで、同期ロジックは持たない。
 */
export class RaindropSyncManager {
    constructor() {
        this.pollingInterval = null;
        this.lastProgress = null;
        this.refreshOnClose = false;
    }

    showModal() {
        this.updateIntroState();
        modalManager.showModal('raindropSyncModal');
        // すでに走っている同期があれば進捗画面へ切り替える
        this.fetchProgress().then((progress) => {
            if (progress && progress.status === 'running') {
                this.showStep('raindropSyncProgressStep');
                this.applyProgress(progress);
                this.startPolling();
            }
        });
    }

    updateIntroState() {
        const settings = (state.global && state.global.settings) || {};
        const tokenState = document.getElementById('raindropSyncTokenState');
        const collectionState = document.getElementById('raindropSyncCollectionState');
        const hint = document.getElementById('raindropSyncIntroHint');
        const startBtn = document.getElementById('raindropSyncStartBtn');

        const hasToken = !!settings.raindrop_token_set;
        const collectionId = (settings.raindrop_collection_id || '').toString().trim();

        if (tokenState) {
            tokenState.textContent = hasToken
                ? translate('recipes.raindropSync.configured', {}, 'Configured')
                : translate('recipes.raindropSync.notConfigured', {}, 'Not configured');
        }
        if (collectionState) {
            collectionState.textContent = collectionId || '-';
        }

        const ready = hasToken && !!collectionId;
        if (hint) {
            hint.textContent = ready
                ? ''
                : translate('recipes.raindropSync.needsSettings', {},
                    'Set the Raindrop token and collection ID in Settings first.');
        }
        if (startBtn) {
            startBtn.disabled = !ready;
        }
    }

    showStep(stepId) {
        ['raindropSyncIntroStep', 'raindropSyncProgressStep', 'raindropSyncResultsStep']
            .forEach((id) => {
                const el = document.getElementById(id);
                if (el) el.style.display = id === stepId ? '' : 'none';
            });
    }

    async startSync() {
        try {
            const response = await fetch('/api/lm/recipes/raindrop-sync/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{}',
            });
            const data = await response.json().catch(() => ({}));

            if (!response.ok || data.success === false) {
                const message = data.error || `HTTP ${response.status}`;
                showToast('toast.recipes.raindropSyncStartFailed', { message }, 'error');
                return;
            }

            this.refreshOnClose = false;
            this.showStep('raindropSyncProgressStep');
            this.applyProgress(data.progress || {});
            this.startPolling();
        } catch (error) {
            showToast('toast.recipes.raindropSyncStartFailed',
                { message: error.message }, 'error');
        }
    }

    async cancelSync() {
        const cancelBtn = document.getElementById('raindropSyncCancelBtn');
        if (cancelBtn) cancelBtn.disabled = true;
        try {
            await fetch('/api/lm/recipes/raindrop-sync/cancel', { method: 'POST' });
        } catch (error) {
            showToast('toast.recipes.raindropSyncCancelFailed',
                { message: error.message }, 'error');
            if (cancelBtn) cancelBtn.disabled = false;
        }
    }

    startPolling() {
        this.stopPolling();
        this.pollingInterval = setInterval(async () => {
            const progress = await this.fetchProgress();
            if (!progress) return;
            this.applyProgress(progress);
            if (progress.status !== 'running') {
                this.stopPolling();
                this.showResults(progress);
            }
        }, POLL_INTERVAL_MS);
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    async fetchProgress() {
        try {
            const response = await fetch('/api/lm/recipes/raindrop-sync/progress');
            if (!response.ok) return null;
            const data = await response.json();
            if (data.success === false) return null;
            this.lastProgress = data.progress || null;
            return this.lastProgress;
        } catch (_) {
            return null;
        }
    }

    applyProgress(progress) {
        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        const percent = Number(progress.progress_percent || 0);
        const bar = document.getElementById('raindropSyncProgressBar');
        if (bar) bar.style.width = `${percent}%`;
        setText('raindropSyncPercent', `${Math.round(percent)}%`);
        setText('raindropSyncTotal', progress.total || 0);
        setText('raindropSyncSuccess', progress.success || 0);
        setText('raindropSyncFailed', progress.failed || 0);
        setText('raindropSyncSkipped', progress.already_synced || 0);
        setText('raindropSyncCurrent', progress.current_image_id || '-');

        const statusText = document.getElementById('raindropSyncStatusText');
        if (statusText) {
            statusText.textContent = progress.status === 'running'
                ? translate('recipes.raindropSync.running', {}, 'Syncing…')
                : translate('recipes.raindropSync.completed', {}, 'Finished');
        }

        this.renderExcluded(progress);

        const logEl = document.getElementById('raindropSyncLog');
        if (logEl && Array.isArray(progress.log)) {
            const wasAtBottom =
                logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
            logEl.textContent = progress.log.join('\n');
            if (wasAtBottom) logEl.scrollTop = logEl.scrollHeight;
        }
    }

    showResults(progress) {
        this.refreshOnClose = (progress.success || 0) > 0;
        this.showStep('raindropSyncResultsStep');

        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        setText('raindropSyncResultTotal', progress.total || 0);
        setText('raindropSyncResultSuccess', progress.success || 0);
        setText('raindropSyncResultFailed', progress.failed || 0);

        let title;
        if (progress.status === 'completed') {
            title = translate('recipes.raindropSync.completed', {}, 'Sync finished');
        } else if (progress.status === 'cancelled') {
            title = translate('recipes.raindropSync.cancelled', {}, 'Sync cancelled');
        } else {
            title = translate('recipes.raindropSync.failedTitle', {}, 'Sync failed');
        }
        setText('raindropSyncResultsTitle', title);
        setText('raindropSyncResultMessage', progress.message || '');

        const failedIds = Array.isArray(progress.failed_ids) ? progress.failed_ids : [];
        const block = document.getElementById('raindropSyncFailedIdsBlock');
        const list = document.getElementById('raindropSyncFailedIds');
        if (block) block.style.display = failedIds.length ? '' : 'none';
        if (list) {
            list.innerHTML = '';
            failedIds.forEach((id) => {
                const row = document.createElement('div');
                const link = document.createElement('a');
                link.href = `https://civitai.com/images/${encodeURIComponent(id)}`;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = String(id);
                row.appendChild(link);
                list.appendChild(row);
            });
        }
    }

    /**
     * ブックマーク数と対象数がずれたとき、その差の内訳を出す。
     * 「330件あるのに325件しか処理されない」を画面だけで説明できるようにするため。
     */
    renderExcluded(progress) {
        const block = document.getElementById('raindropSyncExcludedBlock');
        const summary = document.getElementById('raindropSyncExcludedSummary');
        const list = document.getElementById('raindropSyncExcluded');
        if (!block || !summary || !list) return;

        const ex = progress.excluded || {};
        const bookmarks = progress.bookmarks || 0;
        const total = progress.total || 0;
        const gap = bookmarks - total;

        if (!bookmarks || gap <= 0) {
            block.style.display = 'none';
            return;
        }

        block.style.display = '';
        summary.textContent = translate(
            'recipes.raindropSync.excludedSummary',
            { bookmarks, total, gap },
            `ブックマーク ${bookmarks} 件のうち ${gap} 件は対象外`
        );

        const rows = [
            ['recipes.raindropSync.excludedNotCivitaiImage', 'Civitai画像URLでない', ex.not_civitai_image],
            ['recipes.raindropSync.excludedDuplicate', '同じ画像IDが重複', ex.duplicate],
            ['recipes.raindropSync.excludedAlreadySynced', '同期済み', ex.already_synced],
            ['recipes.raindropSync.excludedNoLink', 'リンクが空', ex.no_link],
        ];

        list.innerHTML = '';
        rows.forEach(([key, fallback, count]) => {
            if (!count) return;
            const row = document.createElement('div');
            row.textContent = `${translate(key, {}, fallback)}: ${count}`;
            list.appendChild(row);
        });

        const samples = Array.isArray(ex.not_civitai_image_samples)
            ? ex.not_civitai_image_samples : [];
        samples.forEach((url) => {
            const row = document.createElement('div');
            const link = document.createElement('a');
            link.href = url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = url;
            row.appendChild(document.createTextNode('  → '));
            row.appendChild(link);
            list.appendChild(row);
        });

        const dupes = Array.isArray(ex.duplicate_image_ids) ? ex.duplicate_image_ids : [];
        dupes.forEach((id) => {
            const row = document.createElement('div');
            row.textContent = `  → ${translate('recipes.raindropSync.excludedDuplicate', {}, '同じ画像IDが重複')}: ${id}`;
            list.appendChild(row);
        });
    }

    toggleExcluded() {
        this._toggleBlock('raindropSyncExcluded', 'raindropSyncExcludedToggleIcon');
    }

    toggleLog() {
        this._toggleBlock('raindropSyncLog', 'raindropSyncLogToggleIcon');
    }

    toggleFailedIds() {
        this._toggleBlock('raindropSyncFailedIds', 'raindropSyncFailedToggleIcon');
    }

    _toggleBlock(blockId, iconId) {
        const block = document.getElementById(blockId);
        const icon = document.getElementById(iconId);
        if (!block) return;
        const hidden = block.style.display === 'none';
        block.style.display = hidden ? '' : 'none';
        if (icon) icon.classList.toggle('expanded', hidden);
    }

    closeAndReset() {
        this.stopPolling();
        modalManager.closeModal('raindropSyncModal');
        this.showStep('raindropSyncIntroStep');
        const cancelBtn = document.getElementById('raindropSyncCancelBtn');
        if (cancelBtn) cancelBtn.disabled = false;

        if (this.refreshOnClose) {
            this.refreshOnClose = false;
            // 追加されたレシピを一覧へ反映する
            if (window.recipeManager && window.recipeManager.pageControls) {
                window.recipeManager.pageControls.refreshModels(false);
            }
        }
    }
}

export const raindropSyncManager = new RaindropSyncManager();
