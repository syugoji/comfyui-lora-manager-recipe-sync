import { buildRecipeWorkflow, getResourceFilename } from './recipeWorkflowBuilder.js';

const MODEL_INPUTS = new Set([
    'ckpt_name', 'lora_name', 'unet_name', 'clip_name', 'clip_name1',
    'clip_name2', 'vae_name', 'control_net_name', 'model_name',
]);
let objectInfoPromise = null;

export function getRecipePromptStatus(recipe) {
    const genParams = recipe?.gen_params || {};
    const prompt = String(genParams.prompt || '').trim();
    const provenance = String(
        genParams.prompt_source || recipe?.prompt_source || ''
    ).trim().toLowerCase();
    const revisionProvenance = String(
        recipe?.revision_summary?.prompt_source || ''
    ).trim().toLowerCase();
    if (recipe?.revision_summary?.active === true && revisionProvenance === 'lm_studio') {
        return 'generated';
    }
    if (prompt && ['lm_studio', 'ai', 'generated'].includes(provenance)) return 'generated';
    return prompt ? 'source' : 'missing';
}

export function getComfyObjectInfo({ force = false } = {}) {
    if (force || !objectInfoPromise) {
        objectInfoPromise = fetch('/object_info')
            .then(response => {
                if (!response.ok) throw new Error(`object_info request failed (${response.status})`);
                return response.json();
            })
            .catch(error => {
                objectInfoPromise = null;
                throw error;
            });
    }
    return objectInfoPromise;
}

function normalizedPath(value) {
    return String(value || '').replaceAll('\\', '/').replace(/^\.\//, '').toLowerCase();
}

function basename(value) {
    return normalizedPath(value).split('/').at(-1) || '';
}

function stem(value) {
    return basename(value).replace(/\.[^.]+$/, '');
}

function compactStem(value) {
    return stem(value).replace(/[^a-z0-9]+/g, '');
}

function compatibleFamilyMatches(inputName, requested, choices) {
    const compact = compactStem(requested);
    if (inputName === 'vae_name') {
        const sdxlAliases = new Set([
            'sdxlvae', 'sdxlvaefixed', 'fixfp16errorssdxllowermemoryusev10',
        ]);
        if (sdxlAliases.has(compact)) {
            return choices.filter(choice => compactStem(choice) === 'sdxlvae');
        }
    }
    if (['clip_name', 'clip_name1', 'clip_name2'].includes(inputName) && compact.startsWith('t5xxl')) {
        return choices.filter(choice => compactStem(choice).startsWith('t5xxl'));
    }
    return [];
}

function modelMatches(inputName, requested, choices) {
    const exact = choices.filter(choice => normalizedPath(choice) === normalizedPath(requested));
    if (exact.length === 1) return exact;
    const byBasename = choices.filter(choice => basename(choice) === basename(requested));
    if (byBasename.length === 1) return byBasename;
    const byStem = choices.filter(choice => stem(choice) === stem(requested));
    if (byStem.length === 1) return byStem;
    return compatibleFamilyMatches(inputName, requested, choices);
}

function inspectBuiltWorkflow(built, objectInfo) {
    const fatal = [];
    const compatible = [];
    const outputIds = Object.entries(built.prompt || {})
        .filter(([, node]) => objectInfo?.[node?.class_type]?.output_node === true)
        .map(([id]) => id);
    const reachable = new Set();
    const pending = [...outputIds];
    while (pending.length > 0) {
        const id = String(pending.pop());
        if (reachable.has(id) || !built.prompt?.[id]) continue;
        reachable.add(id);
        for (const value of Object.values(built.prompt[id]?.inputs || {})) {
            if (Array.isArray(value) && value.length >= 2 && built.prompt[String(value[0])]) {
                pending.push(String(value[0]));
            }
        }
    }

    for (const [id, node] of Object.entries(built.prompt || {})) {
        if (!reachable.has(id)) continue;
        const info = objectInfo?.[node?.class_type];
        if (!info) {
            fatal.push(`不足ノード: ${node?.class_type || 'Unknown'}`);
            continue;
        }
        const inputs = node.inputs || {};
        for (const inputName of Object.keys(info?.input?.required || {})) {
            if (!(inputName in inputs) || inputs[inputName] === null || inputs[inputName] === undefined) {
                fatal.push(`必須入力なし: ${node.class_type}.${inputName}`);
            }
        }
        const specs = { ...(info?.input?.optional || {}), ...(info?.input?.required || {}) };
        for (const [inputName, value] of Object.entries(inputs)) {
            if (!MODEL_INPUTS.has(inputName) || typeof value !== 'string' || !value.trim()) continue;
            const spec = specs?.[inputName];
            const choices = Array.isArray(spec?.[0])
                ? spec[0]
                : (spec?.[0] === 'COMBO' && Array.isArray(spec?.[1]?.options)
                    ? spec[1].options
                    : null);
            if (!choices || !choices.every(choice => typeof choice === 'string')) continue;
            if (modelMatches(inputName, value, choices).length === 1) continue;
            const reason = `未導入モデル: ${value}`;
            fatal.push(reason);
        }
    }

    if (outputIds.length === 0) fatal.push('画像出力がありません（Prompt has no outputs）');
    return {
        fatal: [...new Set(fatal)],
        compatible: [...new Set(compatible)],
    };
}

function result(level, reasons, built = null, audit = null) {
    const metadata = {
        exact: {
            label: '完全ワークフロー', iconClass: 'fas fa-check-circle',
            title: '元画像の実行グラフと必要モデルを確認済みです。ComfyUI・拡張・モデルの版差によるピクセル差は残る場合があります。',
        },
        compatible: {
            label: '互換再構築', iconClass: 'fas fa-tools',
            title: '実行可能ですが、補完・置換・標準再構築を含むため、質感や構図が変わる可能性があります。',
        },
        unavailable: {
            label: '再現不可', iconClass: 'fas fa-ban',
            title: '再現に不可欠な生成情報・ノード・モデルのいずれかが不足しています。',
        },
    }[level];
    const detail = reasons.length ? `${metadata.title}\n${reasons.join('\n')}` : metadata.title;
    return { level, ...metadata, title: detail, reasons, built, audit };
}

function normalizedHash(value) {
    const hash = String(value || '').trim().toLowerCase();
    return /^[a-f0-9]{8,64}$/.test(hash) ? hash : '';
}

function embeddedCheckpointHash(recipe) {
    const direct = normalizedHash(recipe?.gen_params?.model_hash || recipe?.gen_params?.modelHash);
    if (direct) return direct;
    const parameters = [
        recipe?.a1111_parameters,
        recipe?.metadata?.a1111_parameters,
        recipe?.raw_metadata?.parameters,
    ].find(value => typeof value === 'string' && value.trim());
    const match = parameters?.match(/(?:^|[,\n]\s*)Model hash\s*:\s*([a-f0-9]{8,64})/i);
    return normalizedHash(match?.[1]);
}

function resolvedCheckpointHash(recipe) {
    const checkpoint = recipe?.checkpoint || {};
    const direct = normalizedHash(checkpoint.hash || checkpoint.sha256);
    if (direct) return direct;
    const files = checkpoint?.civitai?.files;
    if (!Array.isArray(files)) return '';
    for (const file of files) {
        const hash = normalizedHash(file?.hashes?.SHA256 || file?.hashes?.sha256);
        if (hash) return hash;
    }
    return '';
}

function hashesConflict(left, right) {
    return Boolean(left && right && !left.startsWith(right) && !right.startsWith(left));
}

function loraChoices(objectInfo) {
    const choices = [];
    for (const [classType, info] of Object.entries(objectInfo || {})) {
        const type = String(classType).replace(/[^a-z0-9]+/gi, '').toLowerCase();
        if (!type.startsWith('loraloader') && !type.startsWith('loadlora')) continue;
        const spec = {
            ...(info?.input?.optional || {}),
            ...(info?.input?.required || {}),
        }.lora_name;
        const values = Array.isArray(spec?.[0])
            ? spec[0]
            : (spec?.[0] === 'COMBO' && Array.isArray(spec?.[1]?.options)
                ? spec[1].options
                : []);
        choices.push(...values.filter(value => typeof value === 'string'));
    }
    return [...new Set(choices)];
}

function reachableWorkflowNodes(prompt, objectInfo) {
    const roots = Object.entries(prompt || {})
        .filter(([, node]) => objectInfo?.[node?.class_type]?.output_node === true
            || ['saveimage', 'previewimage', 'saveanimatedwebp', 'saveanimatedpng']
                .includes(String(node?.class_type || '').replace(/[^a-z0-9]+/gi, '').toLowerCase()))
        .map(([id]) => String(id));
    const reachable = new Set();
    const pending = [...roots];
    while (pending.length > 0) {
        const id = String(pending.pop());
        if (reachable.has(id) || !prompt?.[id]) continue;
        reachable.add(id);
        for (const value of Object.values(prompt[id]?.inputs || {})) {
            if (Array.isArray(value) && value.length >= 2 && prompt[String(value[0])]) {
                pending.push(String(value[0]));
            }
        }
    }
    return reachable;
}

function consumedOutputSlots(prompt, reachable, nodeId) {
    const slots = new Set();
    for (const consumerId of reachable) {
        for (const value of Object.values(prompt?.[consumerId]?.inputs || {})) {
            if (Array.isArray(value) && value.length >= 2 && String(value[0]) === String(nodeId)) {
                slots.add(Number(value[1]));
            }
        }
    }
    return slots;
}

export function auditReplayManifest(recipe, built, objectInfo) {
    const manifest = built?.replayManifest || recipe?.replay_manifest || null;
    if (!manifest) {
        return { ok: true, mode: 'legacy', failures: [], required_model_inputs: [] };
    }

    const failures = [];
    const requiredModelInputs = [];
    const fail = (code, requirementId, message) => {
        failures.push({ code, requirement_id: requirementId, message });
    };
    if (manifest.schema !== 'lora-manager.replay-manifest' || manifest.version !== 1) {
        fail('MANIFEST_VERSION_UNSUPPORTED', '', '対応していない再現manifestです。');
        return { ok: false, mode: 'strict', failures, required_model_inputs: [] };
    }
    for (const error of manifest.errors || []) {
        fail(error?.code || 'MANIFEST_ERROR', '', error?.message || '再現manifestにエラーがあります。');
    }

    const prompt = built?.prompt || {};
    const reachable = reachableWorkflowNodes(prompt, objectInfo);
    const choices = loraChoices(objectInfo);
    const loraNodes = Object.entries(prompt).filter(([, node]) => {
        const type = String(node?.class_type || '').replace(/[^a-z0-9]+/gi, '').toLowerCase();
        return type.startsWith('loraloader') || type.startsWith('loadlora');
    });

    for (const requirement of manifest.required_resources || []) {
        if (requirement?.required !== true || requirement?.kind !== 'lora') continue;
        const requirementId = String(requirement.requirement_id || '');
        const expectedFilename = getResourceFilename(requirement.resource);
        const expectedChoices = expectedFilename
            ? modelMatches('lora_name', expectedFilename, choices)
            : [];
        const canonicalChoice = expectedChoices.length === 1 ? expectedChoices[0] : null;
        if (!canonicalChoice) {
            fail(
                'UNRESOLVED_REQUIRED_RESOURCE',
                requirementId,
                `必須LoRAをComfyUIのローカル素材へ一意に解決できません: ${expectedFilename || requirementId}`
            );
        }

        const evidenceNodeIds = new Set(
            (requirement.evidence || [])
                .map(item => item?.node_id)
                .filter(value => value !== undefined && value !== null)
                .map(String)
        );
        const candidates = new Set();
        for (const [nodeId, node] of loraNodes) {
            const metaId = String(node?._meta?.replay_requirement?.id || '');
            if (metaId && metaId === requirementId) candidates.add(String(nodeId));
            if (evidenceNodeIds.has(String(nodeId))) candidates.add(String(nodeId));
            if (!canonicalChoice) continue;
            const actual = node?.inputs?.lora_name;
            const actualChoices = typeof actual === 'string'
                ? modelMatches('lora_name', actual, choices)
                : [];
            if (actualChoices.length === 1
                && normalizedPath(actualChoices[0]) === normalizedPath(canonicalChoice)) {
                candidates.add(String(nodeId));
            }
        }

        if (candidates.size === 0) {
            fail(
                'REQUIRED_LORA_MISSING',
                requirementId,
                `必須LoRAがワークフローにありません: ${expectedFilename || requirementId}`
            );
            continue;
        }
        if (candidates.size > 1) {
            fail(
                'REQUIRED_LORA_DUPLICATE',
                requirementId,
                `必須LoRAが重複適用されています: ${expectedFilename || requirementId}`
            );
            continue;
        }

        const nodeId = [...candidates][0];
        const node = prompt[nodeId];
        const type = String(node?.class_type || '').replace(/[^a-z0-9]+/gi, '').toLowerCase();
        if (type !== 'loraloader') {
            fail(
                'UNSUPPORTED_LORA_LOADER',
                requirementId,
                `v1監査で扱えないLoRAノードです: ${node?.class_type || 'Unknown'}`
            );
            continue;
        }
        if (!reachable.has(nodeId)) {
            fail(
                'REQUIRED_LORA_DISCONNECTED',
                requirementId,
                `必須LoRAが画像出力へ接続されていません: ${expectedFilename || requirementId}`
            );
        }
        if (node?.mode === 2 || node?.mode === 4
            || ['bypass', 'mute', 'never'].includes(String(node?.mode || '').toLowerCase())) {
            fail(
                'REQUIRED_LORA_BYPASSED',
                requirementId,
                `必須LoRAが無効化されています: ${expectedFilename || requirementId}`
            );
        }

        const modelInput = node?.inputs?.model;
        const clipInput = node?.inputs?.clip;
        if (!Array.isArray(modelInput) || !prompt[String(modelInput[0])]
            || !Array.isArray(clipInput) || !prompt[String(clipInput[0])]) {
            fail(
                'REQUIRED_LORA_INPUT_DISCONNECTED',
                requirementId,
                `必須LoRAのMODEL/CLIP入力が接続されていません: ${expectedFilename || requirementId}`
            );
        }
        const slots = consumedOutputSlots(prompt, reachable, nodeId);
        if (!slots.has(0) || !slots.has(1)) {
            fail(
                'REQUIRED_LORA_OUTPUT_DISCONNECTED',
                requirementId,
                `必須LoRAのMODEL/CLIP出力が両方とも利用されていません: ${expectedFilename || requirementId}`
            );
        }

        const actualModel = Number(node?.inputs?.strength_model);
        const actualClip = Number(node?.inputs?.strength_clip);
        const expectedModel = Number(requirement?.expected?.strength_model);
        const expectedClip = Number(requirement?.expected?.strength_clip);
        if (![actualModel, actualClip, expectedModel, expectedClip].every(Number.isFinite)) {
            fail(
                'LORA_STRENGTH_NON_FINITE',
                requirementId,
                `必須LoRAの強度が有限値ではありません: ${expectedFilename || requirementId}`
            );
        } else if (Math.abs(actualModel - expectedModel) > 1e-9
            || Math.abs(actualClip - expectedClip) > 1e-9) {
            fail(
                'LORA_STRENGTH_MISMATCH',
                requirementId,
                `必須LoRAの強度が元データと一致しません: ${expectedFilename || requirementId}`
            );
        }

        requiredModelInputs.push({
            node_id: nodeId,
            widget_name: 'lora_name',
            requirement_id: requirementId,
        });
    }

    return {
        ok: failures.length === 0,
        mode: 'strict',
        failures,
        required_model_inputs: requiredModelInputs,
    };
}

export async function analyzeRecipeReplayCapability(recipe, { objectInfo = null } = {}) {
    try {
        const resolvedObjectInfo = objectInfo || await getComfyObjectInfo();
        const built = buildRecipeWorkflow(recipe, { objectInfo: resolvedObjectInfo });
        const inspected = inspectBuiltWorkflow(built, resolvedObjectInfo);
        const audit = auditReplayManifest(recipe, built, resolvedObjectInfo);
        inspected.fatal.push(...audit.failures.map(failure => failure.message));
        const sourceCheckpointHash = embeddedCheckpointHash(recipe);
        const localCheckpointHash = resolvedCheckpointHash(recipe);
        if (hashesConflict(sourceCheckpointHash, localCheckpointHash)) {
            inspected.fatal.push(
                `元画像のチェックポイントSHAと導入済みモデルが一致しません: ${sourceCheckpointHash} / ${localCheckpointHash.slice(0, 12)}`
            );
        }
        if (recipe?.checkpoint?.inLibrary === false) {
            inspected.fatal.push(
                `未導入または破損したチェックポイント: ${recipe.checkpoint.file_name || recipe.checkpoint.name || 'Unknown'}`
            );
        }
        const manifestRequiredResources = built.replayManifest
            ? built.replayManifest.required_resources
                .filter(item => item?.required === true)
                .map(item => item?.resource)
                .filter(Boolean)
            : null;
        const resourcesToCheck = manifestRequiredResources
            ? [...manifestRequiredResources, ...(recipe?.embeddings || [])]
            : [...(recipe?.loras || []), ...(recipe?.embeddings || [])];
        for (const resource of resourcesToCheck) {
            if (!resource?.exclude && resource?.inLibrary === false) {
                inspected.fatal.push(
                    `未導入または破損した素材: ${resource.file_name || resource.name || resource.modelName || 'Unknown'}`
                );
            }
            if (resource?.exclude || resource?.inLibrary || !resource?.isDeleted) continue;
            inspected.fatal.push(
                `配布終了または取得不能な素材: ${resource.file_name || resource.name || resource.modelName || 'Unknown'}`
            );
        }
        inspected.fatal = [...new Set(inspected.fatal)];
        if (inspected.fatal.length > 0) return result('unavailable', inspected.fatal, built, audit);

        const reasons = [...built.warnings, ...inspected.compatible];
        const exact = built.source === 'embedded' && reasons.length === 0;
        return result(exact ? 'exact' : 'compatible', reasons, built, audit);
    } catch (error) {
        return result('unavailable', [error?.message || String(error)]);
    }
}
