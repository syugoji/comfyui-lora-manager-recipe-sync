import { analyzeRecipeReplayCapability } from '../utils/recipeReplayCapability.js';

const STORAGE_PREFIX = 'lm_recipe_trial_v1:';
const JOB_TTL_MS = 24 * 60 * 60 * 1000;
const DEFAULT_POLL_MS = 1500;
const DEFAULT_TIMEOUT_MS = 2 * 60 * 60 * 1000;
const TERMINAL_CANDIDATE_STATES = new Set([
    'succeeded', 'failed', 'canceled', 'not_submitted', 'submission_unknown',
]);

function safeIntegerSeed(value) {
    const seed = Number(value);
    if (!Number.isSafeInteger(seed) || seed < 0 || seed >= Number.MAX_SAFE_INTEGER) return null;
    return seed;
}

function browserRandomSeed() {
    const values = new Uint32Array(2);
    globalThis.crypto.getRandomValues(values);
    return (values[0] & 0xfffff) * 0x100000000 + values[1];
}

function browserUuid() {
    if (typeof globalThis.crypto?.randomUUID === 'function') {
        return globalThis.crypto.randomUUID();
    }
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map(value => value.toString(16).padStart(2, '0'));
    return `${hex.slice(0, 4).join('')}-${hex.slice(4, 6).join('')}-${hex.slice(6, 8).join('')}-${hex.slice(8, 10).join('')}-${hex.slice(10).join('')}`;
}

export function createTrialSeeds(originalSeed, randomSeed = browserRandomSeed) {
    const original = safeIntegerSeed(originalSeed);
    const used = new Set();
    const result = [];
    if (original !== null) {
        used.add(original);
        result.push({ seed: original, origin: 'original' });
    }
    while (result.length < 4) {
        let candidate = null;
        for (let attempt = 0; attempt < 64; attempt += 1) {
            const value = safeIntegerSeed(randomSeed());
            if (value !== null && !used.has(value)) {
                candidate = value;
                break;
            }
        }
        if (candidate === null) {
            candidate = 0;
            while (used.has(candidate)) candidate += 1;
        }
        used.add(candidate);
        result.push({ seed: candidate, origin: 'random' });
    }
    return result;
}

function queuePromptIds(items) {
    return new Set((Array.isArray(items) ? items : [])
        .map(item => Array.isArray(item) ? item[1] : null)
        .filter(value => typeof value === 'string'));
}

function historyImages(entry) {
    const images = [];
    const visit = (value, outputNodeId = null) => {
        if (Array.isArray(value)) {
            for (const item of value) visit(item, outputNodeId);
            return;
        }
        if (!value || typeof value !== 'object') return;
        if (typeof value.filename === 'string') {
            const image = value;
            const normalized = {
                filename: image.filename,
                subfolder: typeof image.subfolder === 'string' ? image.subfolder : '',
                type: typeof image.type === 'string' ? image.type : 'output',
                output_node_id: outputNodeId,
                image_index: images.filter(item => item.output_node_id === outputNodeId).length,
            };
            const params = new URLSearchParams(normalized);
            normalized.url = `/view?${params.toString()}`;
            images.push(normalized);
            return;
        }
        for (const nested of Object.values(value)) visit(nested, outputNodeId);
    };
    for (const [nodeId, output] of Object.entries(entry?.outputs || {})) {
        visit(output, nodeId);
    }
    return images;
}

function historyFailureMessage(entry) {
    const messages = entry?.status?.messages;
    if (!Array.isArray(messages)) return 'ComfyUIで候補生成に失敗しました。';
    for (let index = messages.length - 1; index >= 0; index -= 1) {
        const item = messages[index];
        const detail = Array.isArray(item) ? item[1] : null;
        const message = detail?.exception_message || detail?.message || detail?.error;
        if (typeof message === 'string' && message.trim()) return message.trim();
    }
    return 'ComfyUIで候補生成に失敗しました。';
}

function cloneJob(job) {
    return job ? JSON.parse(JSON.stringify(job)) : null;
}

export class RecipeTrialManager {
    constructor({
        fetchImpl = (...args) => fetch(...args),
        storage = globalThis.localStorage,
        analyze = analyzeRecipeReplayCapability,
        now = () => Date.now(),
        sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds)),
        randomSeed = browserRandomSeed,
        uuid = browserUuid,
        pollIntervalMs = DEFAULT_POLL_MS,
        timeoutMs = DEFAULT_TIMEOUT_MS,
    } = {}) {
        this.fetchImpl = fetchImpl;
        this.storage = storage;
        this.analyze = analyze;
        this.now = now;
        this.sleep = sleep;
        this.randomSeed = randomSeed;
        this.uuid = uuid;
        this.pollIntervalMs = pollIntervalMs;
        this.timeoutMs = timeoutMs;
        this.currentJob = null;
        this.onUpdate = null;
        this.running = false;
        this.cancelRequested = false;
    }

    storageKey(recipeId) {
        return `${STORAGE_PREFIX}${encodeURIComponent(String(recipeId))}`;
    }

    readStoredJob(recipeId) {
        let job = null;
        try {
            const raw = this.storage?.getItem(this.storageKey(recipeId));
            job = raw ? JSON.parse(raw) : null;
        } catch {
            return null;
        }
        if (!job || job.schema !== 'lora-manager.recipe-trial' || job.version !== 1) {
            return null;
        }
        if (Number(job.expires_at) <= this.now()) {
            try { this.storage?.removeItem(this.storageKey(recipeId)); } catch { /* no-op */ }
            return null;
        }
        return job;
    }

    persist(job) {
        this.storage?.setItem(this.storageKey(job.recipe_id), JSON.stringify(job));
    }

    emit(job = this.currentJob) {
        if (typeof this.onUpdate === 'function') this.onUpdate(cloneJob(job));
    }

    async jsonRequest(url, options = {}, label = 'ComfyUI request') {
        const response = await this.fetchImpl(url, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            const detail = payload?.message
                || payload?.error?.message
                || payload?.error
                || `${label} failed (${response.status})`;
            throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        }
        return payload;
    }

    async queueState() {
        return this.jsonRequest('/queue', {}, 'Queue check');
    }

    async requireEmptyQueue() {
        const queue = await this.queueState();
        if ((queue.queue_running || []).length > 0 || (queue.queue_pending || []).length > 0) {
            throw new Error('ComfyUIのキューが空ではありません。既存生成の完了後に再実行してください。');
        }
    }

    createJob(recipe, draft) {
        const createdAt = this.now();
        const seeds = createTrialSeeds(recipe?.gen_params?.seed, this.randomSeed);
        return {
            schema: 'lora-manager.recipe-trial',
            version: 1,
            job_id: this.uuid(),
            recipe_id: String(recipe.id || recipe.recipe_id),
            recipe_title: String(recipe.title || recipe.id || ''),
            draft_hash: String(draft.draft_hash || ''),
            manifest_hash: String(draft.manifest_hash || ''),
            source_etag: String(recipe.source_etag || recipe.etag || ''),
            draft_snapshot: cloneJob(draft),
            prompt_source: 'lm_studio',
            created_at: createdAt,
            expires_at: createdAt + JOB_TTL_MS,
            status: 'running',
            active_index: null,
            error: null,
            candidates: seeds.map((item, index) => ({
                index,
                candidate_id: `${String(recipe.id || recipe.recipe_id)}:${createdAt}:${index}`,
                seed: item.seed,
                seed_origin: item.origin,
                status: 'pending',
                prompt_id: null,
                attempted_at: null,
                images: [],
                error: null,
            })),
        };
    }

    trialRecipe(recipe, draft, seed) {
        return {
            ...recipe,
            gen_params: {
                ...(recipe.gen_params || {}),
                prompt: draft.proposed_prompt,
                negative_prompt: draft.negative_prompt,
                seed,
            },
        };
    }

    async prepareWorkflow(recipe, draft, seed) {
        const trialRecipe = this.trialRecipe(recipe, draft, seed);
        const analysis = await this.analyze(trialRecipe);
        if (analysis?.level === 'unavailable' || !analysis?.built) {
            throw new Error((analysis?.reasons || []).join(' / ') || '候補workflowを構築できません。');
        }
        if (trialRecipe.replay_manifest && analysis?.audit?.ok !== true) {
            const detail = (analysis?.audit?.failures || [])
                .map(item => item?.message)
                .filter(Boolean)
                .join(' / ');
            throw new Error(detail || '必須LoRAの送信前監査に失敗しました。');
        }
        const built = analysis.built;
        const prepared = await this.jsonRequest('/api/lm/load-recipe-workflow', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: built.prompt,
                source: built.source,
                a1111_parameters: built.a1111Parameters,
                a1111_checkpoint: built.a1111Checkpoint,
                replay_manifest: built.replayManifest ? {
                    schema: built.replayManifest.schema,
                    version: built.replayManifest.version,
                    manifest_hash: built.replayManifest.manifest_hash,
                } : null,
                required_model_inputs: analysis.audit?.required_model_inputs || [],
                prepare_only: true,
            }),
        }, 'Workflow preparation');
        if (!prepared?.success || !prepared?.prompt) {
            throw new Error(prepared?.message || prepared?.error || '候補workflowを準備できません。');
        }
        return prepared.prompt;
    }

    async submitCandidate(job, candidate, prompt) {
        candidate.prompt_id = this.uuid();
        candidate.attempted_at = this.now();
        candidate.status = 'submitting';
        job.active_index = candidate.index;
        this.persist(job);
        this.emit(job);

        let payload;
        try {
            payload = await this.jsonRequest('/prompt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Comfy-Usage-Source': 'lora-manager-recipe-trial',
                },
                body: JSON.stringify({
                    prompt,
                    prompt_id: candidate.prompt_id,
                    extra_data: {
                        lora_manager_recipe_trial: {
                            schema: 'lora-manager.recipe-trial',
                            version: 1,
                            job_id: job.job_id,
                            recipe_id: job.recipe_id,
                            source_etag: job.source_etag,
                            draft_hash: job.draft_hash,
                            manifest_hash: job.manifest_hash,
                            candidate_id: candidate.candidate_id,
                            candidate_index: candidate.index,
                            seed: candidate.seed,
                            prompt_source: 'lm_studio',
                        },
                    },
                }),
            }, 'Prompt submission');
        } catch (error) {
            candidate.status = 'submission_unknown';
            candidate.error = `${error?.message || error} 同じ候補は自動再送しません。`;
            this.persist(job);
            this.emit(job);
            throw error;
        }
        if (payload.prompt_id !== candidate.prompt_id) {
            candidate.status = 'submission_unknown';
            candidate.error = 'ComfyUIが異なるprompt IDを返したため、自動再送せず停止しました。';
            this.persist(job);
            this.emit(job);
            throw new Error(candidate.error);
        }
        candidate.status = 'queued';
        this.persist(job);
        this.emit(job);
    }

    applyHistory(candidate, entry) {
        if (!entry) return false;
        const completed = entry?.status?.completed === true;
        const status = String(entry?.status?.status_str || '').toLowerCase();
        if (!completed && status !== 'error') return false;
        if (status === 'error') {
            candidate.status = 'failed';
            candidate.error = historyFailureMessage(entry);
            return true;
        }
        const images = historyImages(entry);
        if (images.length === 0) {
            candidate.status = 'failed';
            candidate.error = '生成履歴に画像出力がありません。';
            return true;
        }
        candidate.status = 'succeeded';
        candidate.images = images;
        candidate.error = null;
        return true;
    }

    async historyEntry(promptId) {
        const history = await this.jsonRequest(
            `/history/${encodeURIComponent(promptId)}`,
            {},
            'History check'
        );
        return history?.[promptId] || null;
    }

    async pollCandidate(job, candidate) {
        const startedAt = this.now();
        while (!this.cancelRequested && this.now() - startedAt < this.timeoutMs) {
            const entry = await this.historyEntry(candidate.prompt_id);
            if (this.applyHistory(candidate, entry)) {
                this.persist(job);
                this.emit(job);
                return candidate.status === 'succeeded';
            }
            candidate.status = 'running';
            this.persist(job);
            this.emit(job);
            await this.sleep(this.pollIntervalMs);
        }
        if (this.cancelRequested) {
            candidate.status = 'canceled';
            candidate.error = 'ユーザー操作で停止しました。';
        } else {
            candidate.status = 'failed';
            candidate.error = '候補生成の監視がタイムアウトしました。同じ候補は自動再送しません。';
        }
        this.persist(job);
        this.emit(job);
        return false;
    }

    finalizeJob(job) {
        const succeeded = job.candidates.filter(item => item.status === 'succeeded').length;
        const unfinished = job.candidates.filter(item => !TERMINAL_CANDIDATE_STATES.has(item.status));
        if (this.cancelRequested || job.status === 'canceled') {
            job.status = 'canceled';
        } else if (unfinished.length > 0) {
            job.status = succeeded > 0 ? 'partial' : 'failed';
        } else if (succeeded === 4) {
            job.status = 'completed';
        } else if (succeeded > 0) {
            job.status = 'partial';
        } else {
            job.status = 'failed';
        }
        job.active_index = null;
        this.persist(job);
        this.emit(job);
        return job;
    }

    async start({ recipe, draft, onUpdate = null }) {
        if (this.running) throw new Error('別の候補生成を処理中です。');
        if (!recipe?.id && !recipe?.recipe_id) throw new Error('Recipe IDがありません。');
        if (!draft?.draft_hash || !draft?.proposed_prompt) throw new Error('AI下書きがありません。');
        if (!/^[a-f0-9]{64}$/u.test(String(recipe?.source_etag || ''))) {
            throw new Error('保存用ETagを取得できません。レシピ詳細を再読み込みしてください。');
        }
        const currentManifest = recipe?.replay_manifest?.manifest_hash;
        if (!currentManifest || currentManifest !== draft.manifest_hash) {
            throw new Error('レシピまたは再現manifestが更新されています。AI下書きを作り直してください。');
        }
        const stored = this.readStoredJob(recipe.id || recipe.recipe_id);
        if (stored && !['completed', 'failed', 'partial', 'canceled'].includes(stored.status)) {
            throw new Error('24時間以内の未完了候補があります。履歴を復旧してから確認してください。');
        }

        this.running = true;
        this.cancelRequested = false;
        this.onUpdate = onUpdate;
        const job = this.createJob(recipe, draft);
        this.currentJob = job;
        // Fail before any queue mutation if durable recovery is unavailable.
        this.persist(job);
        this.emit(job);
        try {
            await this.requireEmptyQueue();
            for (const candidate of job.candidates) {
                if (this.cancelRequested) break;
                let prompt;
                try {
                    prompt = await this.prepareWorkflow(recipe, draft, candidate.seed);
                    await this.requireEmptyQueue();
                    await this.submitCandidate(job, candidate, prompt);
                    await this.pollCandidate(job, candidate);
                } catch (error) {
                    if (!TERMINAL_CANDIDATE_STATES.has(candidate.status)) {
                        candidate.status = 'failed';
                        candidate.error = error?.message || String(error);
                    }
                    job.error = candidate.error || error?.message || String(error);
                    for (const remaining of job.candidates.slice(candidate.index + 1)) {
                        if (remaining.status === 'pending') remaining.status = 'not_submitted';
                    }
                    break;
                }
            }
            for (const candidate of job.candidates) {
                if (candidate.status === 'pending') candidate.status = 'not_submitted';
            }
            return cloneJob(this.finalizeJob(job));
        } catch (error) {
            job.error = error?.message || String(error);
            for (const candidate of job.candidates) {
                if (candidate.status === 'pending') candidate.status = 'not_submitted';
            }
            this.finalizeJob(job);
            throw error;
        } finally {
            this.running = false;
        }
    }

    async recover(recipeId, { onUpdate = null } = {}) {
        const job = this.readStoredJob(recipeId);
        if (!job) return null;
        this.currentJob = job;
        this.onUpdate = onUpdate;
        this.emit(job);
        for (const candidate of job.candidates || []) {
            if (!candidate.prompt_id || candidate.status === 'succeeded') continue;
            try {
                const entry = await this.historyEntry(candidate.prompt_id);
                this.applyHistory(candidate, entry);
            } catch {
                // Recovery is best-effort. Keep the recorded prompt id for the
                // next check and never submit it again.
            }
        }
        const queue = await this.queueState().catch(() => null);
        let ownQueueActive = false;
        if (queue) {
            const runningIds = queuePromptIds(queue.queue_running);
            const pendingIds = queuePromptIds(queue.queue_pending);
            for (const candidate of job.candidates || []) {
                if (runningIds.has(candidate.prompt_id)) {
                    candidate.status = 'running';
                    ownQueueActive = true;
                } else if (pendingIds.has(candidate.prompt_id)) {
                    candidate.status = 'queued';
                    ownQueueActive = true;
                } else if (candidate.prompt_id && !TERMINAL_CANDIDATE_STATES.has(candidate.status)) {
                    candidate.status = 'submission_unknown';
                    candidate.error = '履歴とキューにprompt IDがありません。同じ候補は自動再送しません。';
                }
            }
        }
        for (const candidate of job.candidates || []) {
            if (!candidate.prompt_id && !TERMINAL_CANDIDATE_STATES.has(candidate.status)) {
                candidate.status = 'not_submitted';
            }
        }
        if (ownQueueActive) {
            job.status = 'recovering';
            job.active_index = (job.candidates || []).find(item => ['running', 'queued'].includes(item.status))?.index ?? null;
            this.persist(job);
            this.emit(job);
        } else {
            this.finalizeJob(job);
        }
        return cloneJob(job);
    }

    async cancel() {
        const job = this.currentJob;
        if (!job) return null;
        this.cancelRequested = true;
        job.status = 'canceled';
        const queue = await this.queueState().catch(() => ({ queue_running: [], queue_pending: [] }));
        const runningIds = queuePromptIds(queue.queue_running);
        const pendingIds = queuePromptIds(queue.queue_pending);
        const deleteIds = [];
        for (const candidate of job.candidates || []) {
            if (!candidate.prompt_id) {
                if (candidate.status === 'pending') candidate.status = 'canceled';
                continue;
            }
            if (pendingIds.has(candidate.prompt_id)) deleteIds.push(candidate.prompt_id);
            if (runningIds.has(candidate.prompt_id)) {
                await this.fetchImpl('/interrupt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt_id: candidate.prompt_id }),
                }).catch(() => null);
            }
            if (!TERMINAL_CANDIDATE_STATES.has(candidate.status)) candidate.status = 'canceled';
        }
        if (deleteIds.length > 0) {
            await this.fetchImpl('/queue', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ delete: deleteIds }),
            }).catch(() => null);
        }
        this.persist(job);
        this.emit(job);
        return cloneJob(job);
    }
}

export const recipeTrialManager = new RecipeTrialManager();
