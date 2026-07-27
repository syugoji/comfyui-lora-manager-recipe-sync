import { fetchRecipeDetails } from '../api/recipeApi.js';
import { showToast } from '../utils/uiHelpers.js';
import { analyzeRecipeReplayCapability } from '../utils/recipeReplayCapability.js';
import { createRecipeWorkflowName } from '../utils/recipeWorkflowName.js';
import { bulkMissingLoraDownloadManager } from './BulkMissingLoraDownloadManager.js';

const ACTIVE_COMFY_CLIENT_KEY = 'lm_last_active_comfy_client';

function replayFailureMessage(recipe, analysis) {
    const detail = (analysis?.reasons || []).join(' / ') || '必須情報を確認できませんでした。';
    if (!recipe?.replay_manifest) return detail;
    return `再現監査で停止しました: ${detail} ComfyUIへ送信していません。元レシピは変更していません。`;
}

class RecipeWorkflowReplayManager {
    async getRecipe(recipeOrId) {
        const recipeId = typeof recipeOrId === 'object'
            ? (recipeOrId.id || recipeOrId.recipe_id)
            : recipeOrId;
        if (!recipeId) throw new Error('Recipe ID is missing');
        return await fetchRecipeDetails(recipeId, { variant: 'active' });
    }

    async ensureResources(recipe) {
        const requiredLoras = recipe?.replay_manifest?.required_resources
            ?.filter(item => item?.required === true && item?.kind === 'lora')
            .map(item => item?.resource)
            .filter(Boolean);
        const resourceRecipe = requiredLoras
            ? { ...recipe, loras: requiredLoras }
            : recipe;
        const stats = bulkMissingLoraDownloadManager.collectMissingResources([resourceRecipe]);
        if (stats.uniqueCount === 0) return true;

        return await bulkMissingLoraDownloadManager.downloadMissingResources([resourceRecipe]);
    }

    async replay(recipeOrId) {
        let recipe = await this.getRecipe(recipeOrId);
        let analysis = await analyzeRecipeReplayCapability(recipe);
        // A red/unavailable recipe is not guaranteed to become reproducible by
        // downloading assets (it may lack prompt, checkpoint identity, nodes,
        // or an output graph).  Do not spend disk/network automatically.  The
        // bulk download action remains available as an explicit user choice.
        if (analysis.level === 'unavailable') {
            throw new Error(replayFailureMessage(recipe, analysis));
        }
        const resourcesReady = await this.ensureResources(recipe);
        if (!resourcesReady) return false;

        // The recipe scan performed by the downloader refreshes local paths and
        // in-library state. Always fetch the latest copy before building nodes.
        recipe = await this.getRecipe(recipe);
        analysis = await analyzeRecipeReplayCapability(recipe);
        if (analysis.level === 'unavailable') {
            throw new Error(replayFailureMessage(recipe, analysis));
        }
        if (recipe.replay_manifest && analysis.audit?.ok !== true) {
            throw new Error(
                '再現監査の結果を確認できないため停止しました。ComfyUIへ送信していません。元レシピは変更していません。'
            );
        }
        const built = analysis.built;
        const workflowName = createRecipeWorkflowName(recipe);
        let targetClientId = null;
        try {
            targetClientId = localStorage.getItem(ACTIVE_COMFY_CLIENT_KEY);
        } catch {
            // The backend can still select the only connected ComfyUI tab.
        }

        const response = await fetch('/api/lm/load-recipe-workflow', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: built.prompt,
                title: recipe.title || recipe.id,
                workflow_name: workflowName,
                source: built.source,
                a1111_parameters: built.a1111Parameters,
                a1111_checkpoint: built.a1111Checkpoint,
                replay_manifest: built.replayManifest ? {
                    schema: built.replayManifest.schema,
                    version: built.replayManifest.version,
                    manifest_hash: built.replayManifest.manifest_hash,
                } : null,
                required_model_inputs: analysis.audit?.required_model_inputs || [],
                target_client_id: targetClientId || null,
            }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.success) {
            throw new Error(data.message || data.error || `Workflow load failed (${response.status})`);
        }

        if (Array.isArray(data.unresolved_models) && data.unresolved_models.length > 0) {
            const missingNames = [...new Set(
                data.unresolved_models
                    .map(item => item?.requested)
                    .filter(Boolean)
            )];
            const preview = missingNames.slice(0, 4).join('、');
            const remainder = missingNames.length > 4 ? ` ほか${missingNames.length - 4}件` : '';
            built.warnings.unshift(
                `未解決モデル: ${preview}${remainder}。赤いローダーノードで実際のファイルを選択してください。`
            );
        }
        if (Array.isArray(data.skipped_models) && data.skipped_models.length > 0) {
            const skippedNames = [...new Set(
                data.skipped_models.map(item => item?.requested).filter(Boolean)
            )];
            built.warnings.unshift(
                `未導入LoRA ${skippedNames.slice(0, 4).join('、')} を安全に迂回しました。`
            );
        }
        const fallbackModels = Array.isArray(data.resolved_models)
            ? data.resolved_models.filter(item => item?.reason === 'fallback to installed upscale model')
            : [];
        if (fallbackModels.length > 0) {
            const replacements = fallbackModels
                .map(item => `${item.requested} → ${item.resolved}`)
                .join('、');
            built.warnings.unshift(`未導入アップスケーラーを導入済みモデルへ置換しました: ${replacements}`);
        }

        // The a1111 path is ComfyUI's native importer converting the metadata
        // into a new graph — the card classifies it as 互換再構築 (yellow), so
        // the toast must not imply the original workflow was loaded as-is.
        const sourceLabel = built.source === 'embedded'
            ? '元画像のワークフロー'
            : built.source === 'a1111'
                ? '元画像のA1111生成データから互換再構築したワークフロー'
            : built.source === 'checkpoint-template'
                ? '互換テンプレートから再構築したワークフロー'
                : '標準構成から再構築したワークフロー';
        const warning = built.warnings.length > 0 ? built.warnings[0] : '';
        showToast(
            'toast.recipes.workflowReplayed',
            { source: sourceLabel, warning },
            built.warnings.length > 0 ? 'warning' : 'success',
            `${sourceLabel}を新しいComfyUIワークフローに読み込みました。${warning}`
        );
        return true;
    }
}

export const recipeWorkflowReplayManager = new RecipeWorkflowReplayManager();
