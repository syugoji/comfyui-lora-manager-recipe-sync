import { resolveSamplerScheduler } from './genParamsMapper.js';

const WORKFLOW_CONTAINER_KEYS = ['comfy', 'comfy_workflow', 'workflow'];
const LORA_TAG_PATTERN = /<lora:([^:>]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*>/gi;
const REPLAY_MANIFEST_SCHEMA = 'lora-manager.replay-manifest';
const REPLAY_MANIFEST_VERSION = 1;

function clone(value) {
    return JSON.parse(JSON.stringify(value));
}

function parseJsonObject(value) {
    if (!value) return null;
    if (typeof value === 'object' && !Array.isArray(value)) return value;
    if (typeof value !== 'string') return null;

    try {
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
    } catch {
        return null;
    }
}

function normalizePromptContainer(value) {
    const parsed = parseJsonObject(value);
    if (!parsed) return null;

    if (parsed.prompt && typeof parsed.prompt === 'object' && !Array.isArray(parsed.prompt)) {
        return clone(parsed.prompt);
    }

    const nodes = Object.values(parsed);
    if (nodes.length > 0 && nodes.every(node => node && typeof node === 'object' && node.class_type)) {
        return clone(parsed);
    }

    return null;
}

function findEmbeddedPrompt(recipe) {
    const candidates = [
        recipe?.comfy,
        recipe?.comfy_prompt,
        recipe?.workflow,
        recipe?.metadata?.comfy,
        recipe?.metadata?.workflow,
        recipe?.raw_metadata?.comfy,
        recipe?.raw_metadata?.workflow,
    ];

    for (const candidate of candidates) {
        const prompt = normalizePromptContainer(candidate);
        if (prompt) return prompt;
    }

    return null;
}

function findA1111Parameters(recipe) {
    const candidates = [
        recipe?.a1111_parameters,
        recipe?.metadata?.a1111_parameters,
        recipe?.raw_metadata?.parameters,
    ];

    return candidates.find(value => typeof value === 'string' && value.trim()) || null;
}

function findCheckpointTemplate(recipe) {
    const images = recipe?.checkpoint?.civitai?.images;
    if (!Array.isArray(images)) return null;

    for (const image of images) {
        const meta = image?.meta;
        if (!meta) continue;

        for (const key of WORKFLOW_CONTAINER_KEYS) {
            const prompt = normalizePromptContainer(meta[key]);
            if (prompt) return prompt;
        }
    }

    return null;
}

function basename(path) {
    if (typeof path !== 'string') return '';
    const parts = path.replaceAll('\\', '/').split('/');
    return parts[parts.length - 1] || '';
}

function workflowRelativePath(path, preferredType = null) {
    if (typeof path !== 'string') return '';

    const normalized = path.replaceAll('\\', '/');
    const lower = normalized.toLowerCase();
    const markers = preferredType === 'Diffusion Model'
        ? ['/models/diffusion_models/', '/models/unet/']
        : preferredType === 'Model'
            ? ['/models/stable-diffusion/', '/models/checkpoints/']
            : ['/models/lora/', '/models/loras/', '/models/lycoris/'];

    for (const marker of markers) {
        const index = lower.lastIndexOf(marker);
        if (index !== -1) return normalized.slice(index + marker.length);
    }

    return basename(normalized);
}

function loraLookupName(value) {
    return basename(String(value || ''))
        .replace(/\.(?:safetensors|ckpt|pt|pth)$/i, '')
        .trim()
        .toLowerCase();
}

function loraCompactName(value) {
    return loraLookupName(value).replace(/[^a-z0-9]+/g, '');
}

function loraNameTokens(value) {
    const genericTokens = new Set([
        'lora', 'locon', 'style', 'model', 'version', 'sd', 'sdxl', 'xl',
        'pony', 'illustrious', 'safetensors', 'safetensor', 'checkpoint',
    ]);
    return basename(String(value || ''))
        .replace(/\.(?:safetensors|ckpt|pt|pth)$/i, '')
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter(token => token.length >= 2
            && !genericTokens.has(token)
            && !/^v?\d+(?:\.\d+)?$/.test(token));
}

function bigramDice(left, right) {
    if (left === right) return 1;
    if (left.length < 2 || right.length < 2) return 0;

    const counts = new Map();
    for (let index = 0; index < left.length - 1; index += 1) {
        const pair = left.slice(index, index + 2);
        counts.set(pair, (counts.get(pair) || 0) + 1);
    }

    let intersection = 0;
    for (let index = 0; index < right.length - 1; index += 1) {
        const pair = right.slice(index, index + 2);
        const count = counts.get(pair) || 0;
        if (count > 0) {
            intersection += 1;
            counts.set(pair, count - 1);
        }
    }
    return (2 * intersection) / (left.length + right.length - 2);
}

function loraNameSimilarity(left, right) {
    const leftCompact = loraCompactName(left);
    const rightCompact = loraCompactName(right);
    if (!leftCompact || !rightCompact) return 0;
    if (leftCompact === rightCompact) return 1;

    const shorter = leftCompact.length <= rightCompact.length ? leftCompact : rightCompact;
    const longer = shorter === leftCompact ? rightCompact : leftCompact;
    const lengthRatio = shorter.length / longer.length;
    let score = 0;
    if (shorter.length >= 6 && longer.includes(shorter)) {
        score = 0.82 + (0.16 * lengthRatio);
    }

    const leftTokens = new Set(loraNameTokens(left));
    const rightTokens = new Set(loraNameTokens(right));
    if (leftTokens.size > 0 && rightTokens.size > 0) {
        const common = [...leftTokens].filter(token => rightTokens.has(token)).length;
        const containment = common / Math.min(leftTokens.size, rightTokens.size);
        const union = new Set([...leftTokens, ...rightTokens]).size;
        const jaccard = union ? common / union : 0;
        score = Math.max(score, (0.72 * containment) + (0.28 * jaccard));
    }

    return Math.max(score, bigramDice(leftCompact, rightCompact) * 0.9);
}

function loraCandidateNames(lora) {
    const aliases = [
        ...(Array.isArray(lora?.aliases) ? lora.aliases : []),
        ...(Array.isArray(lora?.promptAliases) ? lora.promptAliases : []),
    ];
    const civitai = lora?.civitai || {};
    const apiFiles = Array.isArray(civitai.files) ? civitai.files.map(file => file?.name) : [];
    return [
        lora?.file_name,
        lora?.filename,
        lora?.name,
        lora?.modelName,
        lora?.modelVersionName,
        civitai?.name,
        civitai?.model?.name,
        ...apiFiles,
        ...aliases,
    ].filter(Boolean);
}

function getLoraStrength(lora, fallback = 1) {
    const candidates = [lora?.weight, lora?.strength];
    const value = candidates.find(candidate => Number.isFinite(Number(candidate)));
    return value === undefined ? fallback : Number(value);
}

function getLoraStrengths(lora, fallback = 1) {
    const shared = getLoraStrength(lora, fallback);
    const model = Number.isFinite(Number(lora?.strength_model))
        ? Number(lora.strength_model)
        : shared;
    const clip = Number.isFinite(Number(lora?.strength_clip))
        ? Number(lora.strength_clip)
        : shared;
    return { model, clip };
}

function extractPromptLoras(prompt) {
    if (typeof prompt !== 'string') return { text: prompt, loras: [] };

    const loras = [];
    const text = prompt.replace(LORA_TAG_PATTERN, (tag, rawName, rawStrength) => {
        const name = String(rawName || '').trim();
        const strength = Number(rawStrength);
        if (name && Number.isFinite(strength)) loras.push({ name, strength });
        return '';
    }).replace(/\s{2,}/g, ' ').trim();

    return { text, loras };
}

function cleanPromptText(value) {
    if (typeof value !== 'string') return value;
    return value.replace(/(?:\u200b|â)/g, '').trim();
}

function getReplayManifest(recipe, options = {}) {
    const manifest = options.replayManifest ?? recipe?.replay_manifest ?? null;
    if (!manifest) return null;
    if (manifest.schema !== REPLAY_MANIFEST_SCHEMA || manifest.version !== REPLAY_MANIFEST_VERSION) {
        throw new Error('対応していない再現manifestです。レシピ詳細を再読み込みしてください。');
    }
    if (!Array.isArray(manifest.required_resources)
        || !Array.isArray(manifest.advisory_resources)
        || !Array.isArray(manifest.errors)) {
        throw new Error('再現manifestの形式が不正です。ComfyUIを再起動して再スキャンしてください。');
    }
    if (manifest.errors.length > 0) {
        const detail = manifest.errors
            .map(error => error?.message || error?.code)
            .filter(Boolean)
            .join(' / ');
        throw new Error(`再現manifestを確定できません: ${detail || 'Unknown manifest error'}`);
    }
    return manifest;
}

function requiredManifestLoras(manifest) {
    if (!manifest) return null;
    const required = manifest.required_resources.filter(
        item => item?.required === true && item?.kind === 'lora'
    );
    const seenIds = new Set();
    const seenFilenames = new Set();
    return required.map(item => {
        const id = String(item?.requirement_id || '').trim();
        const status = String(item?.resolution?.status || '');
        const resource = item?.resource;
        const model = Number(item?.expected?.strength_model);
        const clip = Number(item?.expected?.strength_clip);
        if (!id || seenIds.has(id)) {
            throw new Error(`再現manifestの必須LoRA IDが重複しています: ${id || 'Unknown'}`);
        }
        if (!['recipe_match', 'inline_only'].includes(status)
            || !resource || typeof resource !== 'object') {
            throw new Error(`必須LoRAを保存素材へ解決できません: ${id}`);
        }
        if (!Number.isFinite(model) || !Number.isFinite(clip)) {
            throw new Error(`必須LoRAの強度が不正です: ${id}`);
        }
        const workflowFilename = getResourceFilename(resource);
        const filenameKey = workflowFilename.replaceAll('\\', '/').toLowerCase();
        if (!workflowFilename || seenFilenames.has(filenameKey)) {
            throw new Error(`必須LoRAのファイルを一意に確定できません: ${workflowFilename || id}`);
        }
        seenIds.add(id);
        seenFilenames.add(filenameKey);
        return {
            ...resource,
            weight: model,
            strength: model,
            strength_model: model,
            strength_clip: clip,
            _replayRequirement: {
                schema_version: REPLAY_MANIFEST_VERSION,
                required: true,
                id,
                manifest_hash: manifest.manifest_hash || '',
            },
        };
    });
}

function mergePromptLoras(recipeLoras, promptLoras, { promptAuthoritative = false } = {}) {
    const result = Array.isArray(recipeLoras) ? recipeLoras.map(lora => ({ ...lora })) : [];
    const structuredCount = result.length;
    const matchedStructuredIndexes = new Set();
    const byName = new Map();
    const fuzzyClaimedIndexes = new Set();
    result.forEach((lora, index) => {
        for (const candidate of loraCandidateNames(lora)) {
            const key = loraCompactName(candidate);
            if (key) byName.set(key, index);
        }
    });

    for (const tagged of promptLoras) {
        const key = loraCompactName(tagged.name);
        let existingIndex = byName.get(key);
        if (existingIndex === undefined && key.length >= 6) {
            // Fuzzy matching is only safe against structured recipe resources.
            // Prompt tags added earlier in this loop are independent inputs;
            // similarly named tags (for example Korean/Taiwan Doll Likeness)
            // must not collapse into one loader.
            const ranked = result.slice(0, structuredCount)
                .map((lora, index) => ({
                    index,
                    score: Math.max(0, ...loraCandidateNames(lora)
                        .map(candidate => loraNameSimilarity(tagged.name, candidate))),
                }))
                .filter(candidate => !fuzzyClaimedIndexes.has(candidate.index))
                .sort((left, right) => right.score - left.score);
            const best = ranked[0];
            const runnerUp = ranked[1];
            if (best?.score >= 0.62 && (!runnerUp || best.score - runnerUp.score >= 0.12)) {
                existingIndex = best.index;
                fuzzyClaimedIndexes.add(existingIndex);
            }
        }
        if (existingIndex !== undefined) {
            if (existingIndex < structuredCount) matchedStructuredIndexes.add(existingIndex);
            // The inline tag is the explicit per-image setting, so it wins over
            // a generic resource strength stored in the recipe.
            result[existingIndex].weight = tagged.strength;
            const aliases = new Set(result[existingIndex].promptAliases || []);
            aliases.add(tagged.name);
            result[existingIndex].promptAliases = [...aliases];
            byName.set(key, existingIndex);
            continue;
        }

        result.push({
            name: tagged.name,
            file_name: tagged.name,
            weight: tagged.strength,
        });
        byName.set(key, result.length - 1);
    }

    if (promptAuthoritative && promptLoras.length > 0) {
        return result.filter((_, index) => (
            index >= structuredCount || matchedStructuredIndexes.has(index)
        ));
    }
    return result;
}

function civitaiFile(resource, preferredType = null) {
    const files = resource?.civitai?.files;
    if (!Array.isArray(files) || files.length === 0) return null;

    if (preferredType) {
        const typed = files.find(file => String(file?.type || '').toLowerCase() === preferredType.toLowerCase());
        if (typed) return typed;
    }

    return files.find(file => file?.primary) || files[0];
}

export function getResourceFilename(resource, preferredType = null) {
    if (!resource) return '';

    const apiFile = civitaiFile(resource, preferredType);
    const candidates = [
        resource.inLibrary ? resource.localPath : null,
        resource.file_name,
        resource.filename,
        resource.localPath,
        resource.file_path,
        apiFile?.name,
    ];

    let filename = '';
    for (const candidate of candidates) {
        filename = workflowRelativePath(candidate, preferredType);
        if (filename) break;
    }

    if (filename && !/\.[a-z0-9]{2,16}$/i.test(filename)) {
        const apiName = basename(apiFile?.name);
        if (apiName && apiName.toLowerCase().startsWith(filename.toLowerCase())) {
            filename = apiName;
        } else {
            filename += '.safetensors';
        }
    }

    return filename;
}

function parseSize(size) {
    if (typeof size === 'string') {
        const match = size.match(/(\d+)\s*[xX\u00d7,]\s*(\d+)/);
        if (match) return { width: Number(match[1]), height: Number(match[2]) };
    }
    if (Array.isArray(size) && size.length >= 2) {
        return { width: Number(size[0]), height: Number(size[1]) };
    }
    if (size && typeof size === 'object') {
        return { width: Number(size.width), height: Number(size.height) };
    }
    return null;
}

function normalizedClassType(value) {
    return String(value || '').replace(/[^a-z0-9]+/gi, '').toLowerCase();
}

function filenameFromName(value) {
    const name = basename(String(value || '').trim());
    if (!name || /^none|automatic$/i.test(name)) return '';
    return /\.[a-z0-9]{2,16}$/i.test(name) ? name : `${name}.safetensors`;
}

function parameterValue(parameters, key) {
    if (typeof parameters !== 'string') return '';
    const match = parameters.match(new RegExp(`(?:^|[,\\n]\\s*)${key}\\s*:\\s*([^,\\r\\n]+)`, 'i'));
    return match?.[1]?.trim() || '';
}

function applyA1111LoraWeights(loras, parameters) {
    const resources = Array.isArray(loras) ? loras : [];
    if (!parameters || resources.length === 0) return resources;

    const match = parameters.match(/(?:^|[,\n]\s*)lora\s*weights\s*:\s*"([^"]+)"/i);
    if (!match) return resources;
    const weights = match[1].split(',').map(value => Number(value.trim()));
    if (weights.length !== resources.length || weights.some(value => !Number.isFinite(value))) {
        return resources;
    }
    return resources.map((lora, index) => ({ ...lora, weight: weights[index] }));
}

function recipeVaeName(recipe) {
    const gen = recipe?.gen_params || {};
    return filenameFromName(
        gen.vae || gen.vae_name || parameterValue(findA1111Parameters(recipe), 'VAE')
    );
}

function isFluxRecipe(recipe) {
    const checkpoint = recipe?.checkpoint || {};
    const declaredFamily = [
        recipe?.base_model,
        checkpoint.baseModel,
        checkpoint.base_model,
    ].find(value => typeof value === 'string' && value.trim());
    if (declaredFamily) return declaredFamily.toLowerCase().includes('flux');

    const fallbackIdentity = [
        checkpoint.name,
        checkpoint.localPath,
        recipe?.gen_params?.model,
    ].filter(Boolean).join(' ').toLowerCase();
    return fallbackIdentity.includes('flux');
}

function requiresStructuredA1111(parameters) {
    if (typeof parameters !== 'string' || !parameters.trim()) return false;
    const size = parameterValue(parameters, 'Size');
    if (!parseSize(size)) return true;
    return [
        /\bVersion\s*:\s*ComfyUI\b/i,
        /\bVAE\s*:/i,
        /\bHires (?:upscale|upscaler|steps)\s*:/i,
        /\bADetailer\b/i,
        /\bTiled Diffusion\b/i,
        /\b(?:FreeU|Refiner|AutomaticVAE|LoRA\s*weights|MultiDiffusion|PAG|Segment)\b/i,
        /<segment\b/i,
        /\bworkflow\s*:/i,
        /\(None,?\)x\(None,?\)/i,
    ].some(pattern => pattern.test(parameters));
}

function a1111CompatibilityFeatures(parameters) {
    const patterns = [
        ['VAE', /\bVAE\s*:/i],
        ['hires', /\bHires (?:upscale|upscaler|steps)\s*:/i],
        ['ADetailer', /\bADetailer\b/i],
        ['Tiled Diffusion', /\bTiled Diffusion\b/i],
        ['FreeU', /\bFreeU\b/i],
        ['Refiner', /\bRefiner\b/i],
        ['LoRA weights', /\bLoRA\s*weights\b/i],
        ['Segment', /(?:\bSegment\b|<segment\b)/i],
        ['PAG', /\bPAG\b/i],
    ];
    return patterns.filter(([, pattern]) => pattern.test(parameters || '')).map(([label]) => label);
}

function vaeDecodeInputs(samples, vae, pixelCount) {
    if (pixelCount <= 2_500_000) return { inputs: { samples, vae }, class_type: 'VAEDecode' };
    return {
        inputs: {
            samples,
            vae,
            tile_size: 512,
            overlap: 64,
            temporal_size: 64,
            temporal_overlap: 8,
        },
        class_type: 'VAEDecodeTiled',
    };
}

function hiresUpscalerName(recipe) {
    return String(
        recipe?.gen_params?.hires_upscaler
        || parameterValue(findA1111Parameters(recipe), 'Hires upscaler')
        || ''
    ).trim();
}

function usesPixelHiresUpscaler(name) {
    if (!name || /^(?:none|latent|nearest|nearest-exact|bilinear|bicubic|area|bislerp)$/i.test(name)) {
        return false;
    }
    return /(?:4x|esrgan|realesr|ultrasharp|remacri|swinir|upscal)/i.test(name);
}

function installedUpscalerAlias(name) {
    if (/remacri/i.test(name)) return 'remacri_original.pth';
    return name;
}

function embeddedPromptNeedsRebuild(prompt, recipe) {
    if (!isFluxRecipe(recipe)) return false;
    const hasCheckpointLoader = Object.values(prompt).some(
        node => normalizedClassType(node?.class_type) === 'checkpointloadersimple'
    );
    const path = String(recipe?.checkpoint?.localPath || '').replaceAll('\\', '/').toLowerCase();
    return hasCheckpointLoader && (path.includes('/diffusion_models/') || path.includes('/unet/'));
}

function inlineLegacyConstants(prompt, warnings) {
    const replacements = new Map();
    for (const [id, node] of Object.entries(prompt)) {
        const type = normalizedClassType(node?.class_type);
        if (!['int', 'float', 'string'].includes(type)) continue;
        const raw = node?.inputs?.Number ?? node?.inputs?.number ?? node?.inputs?.value
            ?? node?.inputs?.String ?? node?.inputs?.string;
        let value = raw;
        if (type === 'int') value = Number.parseInt(raw, 10);
        if (type === 'float') value = Number.parseFloat(raw);
        if ((type === 'int' || type === 'float') && !Number.isFinite(value)) continue;
        replacements.set(String(id), value);
    }
    if (replacements.size === 0) return;
    for (const node of Object.values(prompt)) {
        for (const [key, value] of Object.entries(node?.inputs || {})) {
            if (Array.isArray(value) && replacements.has(String(value[0]))) {
                node.inputs[key] = replacements.get(String(value[0]));
            }
        }
    }
    for (const id of replacements.keys()) delete prompt[id];
    warnings.push(`旧式の定数ノード${replacements.size}件を標準入力値へ変換しました。`);
}

// Civitai/A1111 use -1 to mean "random seed".  ComfyUI validates sampler
// seeds as unsigned integers, so pass a safe non-negative value instead.
function normalizeSeed(value, fallback = 0) {
    const seed = Number(value);
    if (!Number.isFinite(seed) || seed < 0) return fallback;
    return Math.trunc(seed);
}

function standardPrompt(recipe) {
    const gen = recipe?.gen_params || {};
    const size = parseSize(gen.size) || { width: 1024, height: 1024 };
    const diffusionName = getResourceFilename(recipe?.checkpoint, 'Diffusion Model')
        || filenameFromName(gen.model);
    const checkpointName = getResourceFilename(recipe?.checkpoint, 'Model')
        || filenameFromName(gen.model);
    const vaeName = recipeVaeName(recipe);
    const steps = Number.isFinite(Number(gen.steps)) ? Number(gen.steps) : 20;
    const cfg = Number.isFinite(Number(gen.cfg_scale)) ? Number(gen.cfg_scale) : 7;

    if (!checkpointName && !diffusionName) {
        throw new Error('再現に必要なチェックポイント情報がありません。元画像にモデル情報を含む生成データが必要です。');
    }
    if (recipe?.generation_source === 'reconstructed' && !String(gen.prompt || '').trim()) {
        throw new Error('元画像にプロンプト／生成パラメータがなく、このレシピは正しく再現できません。');
    }

    if (isFluxRecipe(recipe)) {
        const fluxVae = vaeName || 'ae.safetensors';
        const decode = vaeDecodeInputs(['8', 0], ['3', 0], size.width * size.height);
        return {
            '1': {
                inputs: { unet_name: diffusionName, weight_dtype: 'default' },
                class_type: 'UNETLoader',
                _meta: { title: 'Load Diffusion Model' },
            },
            '2': {
                inputs: {
                    clip_name1: 't5xxl_fp8_e4m3fn_scaled.safetensors',
                    clip_name2: 'clip_l.safetensors',
                    type: 'flux',
                },
                class_type: 'DualCLIPLoader',
                _meta: { title: 'DualCLIPLoader' },
            },
            '3': {
                inputs: { vae_name: fluxVae },
                class_type: 'VAELoader',
                _meta: { title: 'Load VAE' },
            },
            '4': {
                inputs: { text: gen.prompt || '', clip: ['2', 0] },
                class_type: 'CLIPTextEncode',
                _meta: { title: 'CLIP Text Encode (Positive Prompt)' },
            },
            '5': {
                inputs: { text: gen.negative_prompt || '', clip: ['2', 0] },
                class_type: 'CLIPTextEncode',
                _meta: { title: 'CLIP Text Encode (Negative Prompt)' },
            },
            '6': {
                inputs: { conditioning: ['4', 0], guidance: cfg || 3.5 },
                class_type: 'FluxGuidance',
                _meta: { title: 'Flux Guidance' },
            },
            '7': {
                inputs: { width: size.width, height: size.height, batch_size: 1 },
                class_type: 'EmptySD3LatentImage',
                _meta: { title: 'Empty SD3 Latent Image' },
            },
            '8': {
                inputs: {
                    seed: normalizeSeed(gen.seed), steps, cfg: 1,
                    sampler_name: 'euler', scheduler: 'normal', denoise: 1,
                    model: ['1', 0], positive: ['6', 0], negative: ['5', 0],
                    latent_image: ['7', 0],
                },
                class_type: 'KSampler',
                _meta: { title: 'KSampler' },
            },
            '9': {
                ...decode,
                _meta: { title: decode.class_type === 'VAEDecodeTiled' ? 'VAE Decode (Tiled)' : 'VAE Decode' },
            },
            '10': {
                inputs: { filename_prefix: `Recipe_${recipe?.title || recipe?.id || 'ComfyUI'}`, images: ['9', 0] },
                class_type: 'SaveImage',
                _meta: { title: 'Save Image' },
            },
        };
    }

    const decode = vaeDecodeInputs(['5', 0], ['1', 2], size.width * size.height);
    const prompt = {
        '1': {
            inputs: { ckpt_name: checkpointName },
            class_type: 'CheckpointLoaderSimple',
            _meta: { title: 'Load Checkpoint' },
        },
        '2': {
            inputs: { text: gen.prompt || '', clip: ['1', 1] },
            class_type: 'CLIPTextEncode',
            _meta: { title: 'CLIP Text Encode (Positive Prompt)' },
        },
        '3': {
            inputs: { text: gen.negative_prompt || '', clip: ['1', 1] },
            class_type: 'CLIPTextEncode',
            _meta: { title: 'CLIP Text Encode (Negative Prompt)' },
        },
        '4': {
            inputs: { width: size.width, height: size.height, batch_size: 1 },
            class_type: 'EmptyLatentImage',
            _meta: { title: 'Empty Latent Image' },
        },
        '5': {
            inputs: {
                seed: normalizeSeed(gen.seed),
                steps,
                cfg,
                sampler_name: 'euler',
                scheduler: 'normal',
                // This reconstructed graph is txt2img and starts from an empty
                // latent. A1111's denoising strength usually belongs to a
                // hires/img2img pass and produces a flat image here when < 1.
                denoise: 1,
                model: ['1', 0],
                positive: ['2', 0],
                negative: ['3', 0],
                latent_image: ['4', 0],
            },
            class_type: 'KSampler',
            _meta: { title: 'KSampler' },
        },
        '6': {
            ...decode,
            _meta: { title: decode.class_type === 'VAEDecodeTiled' ? 'VAE Decode (Tiled)' : 'VAE Decode' },
        },
        '7': {
            inputs: { filename_prefix: `Recipe_${recipe?.title || recipe?.id || 'ComfyUI'}`, images: ['6', 0] },
            class_type: 'SaveImage',
            _meta: { title: 'Save Image' },
        },
    };

    if (vaeName) {
        prompt['8'] = {
            inputs: { vae_name: vaeName },
            class_type: 'VAELoader',
            _meta: { title: 'Load VAE' },
        };
        prompt['6'].inputs.vae = ['8', 0];
    }

    const hiresScale = Number(gen.hires_upscale);
    if (Number.isFinite(hiresScale) && hiresScale > 1) {
        const upscalerName = hiresUpscalerName(recipe);
        const hiresSamplerInputs = {
            ...prompt['5'].inputs,
            seed: normalizeSeed(gen.seed) + 1,
            steps: Number.isFinite(Number(gen.hires_steps)) ? Number(gen.hires_steps) : steps,
            cfg: Number.isFinite(Number(gen.hires_cfg_scale)) ? Number(gen.hires_cfg_scale) : cfg,
            denoise: Number.isFinite(Number(gen.denoising_strength)) ? Number(gen.denoising_strength) : 0.35,
        };

        if (usesPixelHiresUpscaler(upscalerName)) {
            const vaeReference = [...prompt['6'].inputs.vae];
            const modelId = nextNodeId(prompt);
            const upscaleId = String(Number(modelId) + 1);
            const resizeId = String(Number(modelId) + 2);
            const encodeId = String(Number(modelId) + 3);
            const samplerId = String(Number(modelId) + 4);
            const decodeId = String(Number(modelId) + 5);
            const targetWidth = Math.max(8, Math.round((size.width * hiresScale) / 8) * 8);
            const targetHeight = Math.max(8, Math.round((size.height * hiresScale) / 8) * 8);
            const finalDecode = vaeDecodeInputs(
                [samplerId, 0],
                vaeReference,
                targetWidth * targetHeight
            );
            prompt[modelId] = {
                inputs: { model_name: installedUpscalerAlias(upscalerName) },
                class_type: 'UpscaleModelLoader',
                _meta: { title: `Load Hires Upscaler: ${upscalerName}` },
            };
            prompt[upscaleId] = {
                inputs: { upscale_model: [modelId, 0], image: ['6', 0] },
                class_type: 'ImageUpscaleWithModel',
                _meta: { title: 'Image Hires Upscale (Model)' },
            };
            prompt[resizeId] = {
                inputs: {
                    image: [upscaleId, 0], upscale_method: 'lanczos',
                    width: targetWidth, height: targetHeight, crop: 'disabled',
                },
                class_type: 'ImageScale',
                _meta: { title: 'Resize to Hires Target' },
            };
            prompt[encodeId] = {
                inputs: { pixels: [resizeId, 0], vae: vaeReference },
                class_type: 'VAEEncode',
                _meta: { title: 'VAE Encode (Hires)' },
            };
            prompt[samplerId] = {
                inputs: { ...hiresSamplerInputs, latent_image: [encodeId, 0] },
                class_type: 'KSampler',
                _meta: { title: 'KSampler (Hires pass)' },
            };
            prompt[decodeId] = {
                ...finalDecode,
                _meta: {
                    title: finalDecode.class_type === 'VAEDecodeTiled'
                        ? 'VAE Decode (Hires Tiled)'
                        : 'VAE Decode (Hires)',
                },
            };
            prompt['7'].inputs.images = [decodeId, 0];
        } else {
            prompt['9'] = {
                inputs: { samples: ['5', 0], upscale_method: 'bislerp', scale_by: hiresScale },
                class_type: 'LatentUpscaleBy',
                _meta: { title: 'Latent Hires Upscale' },
            };
            prompt['10'] = {
                inputs: { ...hiresSamplerInputs, latent_image: ['9', 0] },
                class_type: 'KSampler',
                _meta: { title: 'KSampler (Hires pass)' },
            };
            prompt['6'].inputs.samples = ['10', 0];
        }
    }

    const clipSkip = Number(gen.clip_skip);
    if (Number.isInteger(clipSkip) && clipSkip > 1) {
        const id = nextNodeId(prompt);
        prompt[id] = {
            inputs: { clip: ['1', 1], stop_at_clip_layer: -clipSkip },
            class_type: 'CLIPSetLastLayer',
            _meta: { title: `CLIP Skip ${clipSkip}` },
        };
        prompt['2'].inputs.clip = [id, 0];
        prompt['3'].inputs.clip = [id, 0];
    }

    return prompt;
}

function samplerUsesEmptyLatent(prompt, inputs) {
    const latentReference = inputs?.latent_image;
    if (!Array.isArray(latentReference) || latentReference.length === 0) return false;

    const latentNode = prompt[String(latentReference[0])];
    return /Empty.*LatentImage/i.test(latentNode?.class_type || '');
}

function patchGenerationParameters(prompt, recipe) {
    const gen = recipe?.gen_params || {};
    const promptLoras = extractPromptLoras(gen.prompt);
    const entries = Object.entries(prompt);
    const textNodes = entries.filter(([, node]) => normalizedClassType(node?.class_type) === 'cliptextencode');
    let positiveNode = textNodes.find(([, node]) => {
        const title = String(node?._meta?.title || '').toLowerCase();
        return title.includes('positive') && !title.includes('negative');
    });
    let negativeNode = textNodes.find(([, node]) => String(node?._meta?.title || '').toLowerCase().includes('negative'));

    positiveNode ||= textNodes[0];
    negativeNode ||= textNodes.find(entry => entry !== positiveNode) || textNodes[1];

    if (positiveNode && typeof promptLoras.text === 'string') positiveNode[1].inputs.text = promptLoras.text;
    if (negativeNode && typeof gen.negative_prompt === 'string') negativeNode[1].inputs.text = gen.negative_prompt;

    const schedulerHint = gen.scheduler || parameterValue(findA1111Parameters(recipe), 'Schedule type');
    const resolvedSampler = resolveSamplerScheduler(
        [gen.sampler, schedulerHint].filter(Boolean).join(' ')
    );
    for (const [, node] of entries) {
        const inputs = node?.inputs;
        if (!inputs || typeof inputs !== 'object') continue;

        if (/KSampler/i.test(node.class_type || '')) {
            const isHiresPass = String(node?._meta?.title || '').toLowerCase().includes('hires');
            if ('seed' in inputs) {
                const sourceSeed = Number.isFinite(Number(gen.seed)) ? gen.seed : inputs.seed;
                inputs.seed = normalizeSeed(sourceSeed) + (isHiresPass ? 1 : 0);
            }
            const requestedSteps = isHiresPass && Number.isFinite(Number(gen.hires_steps))
                ? gen.hires_steps : gen.steps;
            const requestedCfg = isHiresPass && Number.isFinite(Number(gen.hires_cfg_scale))
                ? gen.hires_cfg_scale : gen.cfg_scale;
            if (Number.isFinite(Number(requestedSteps)) && 'steps' in inputs) inputs.steps = Number(requestedSteps);
            if (Number.isFinite(Number(requestedCfg)) && 'cfg' in inputs && !isFluxRecipe(recipe)) {
                inputs.cfg = Number(requestedCfg);
            }
            if ('denoise' in inputs) {
                if (samplerUsesEmptyLatent(prompt, inputs)) {
                    inputs.denoise = 1;
                } else if (Number.isFinite(Number(gen.denoising_strength))) {
                    inputs.denoise = Number(gen.denoising_strength);
                }
            }
            if (resolvedSampler.sampler && 'sampler_name' in inputs) inputs.sampler_name = resolvedSampler.sampler;
            if (resolvedSampler.scheduler && 'scheduler' in inputs) inputs.scheduler = resolvedSampler.scheduler;
        }

        if (normalizedClassType(node.class_type) === 'randomnoise' && 'noise_seed' in inputs) {
            const sourceSeed = Number.isFinite(Number(gen.seed)) ? gen.seed : inputs.noise_seed;
            inputs.noise_seed = normalizeSeed(sourceSeed);
        }
    }

    const size = parseSize(gen.size);
    if (size) {
        for (const [, node] of entries) {
            if (!/Empty.*LatentImage/i.test(node?.class_type || '')) continue;
            if ('width' in node.inputs) node.inputs.width = size.width;
            if ('height' in node.inputs) node.inputs.height = size.height;
        }
    }
}

function patchCheckpoint(prompt, checkpoint) {
    const checkpointFilename = getResourceFilename(checkpoint, 'Model');
    const diffusionFilename = getResourceFilename(checkpoint, 'Diffusion Model') || checkpointFilename;
    if (!checkpointFilename && !diffusionFilename) return;

    for (const node of Object.values(prompt)) {
        if (!node?.inputs) continue;
        const type = normalizedClassType(node.class_type);
        if (type === 'checkpointloadersimple' && checkpointFilename) {
            node.inputs.ckpt_name = checkpointFilename;
        } else if (type === 'unetloader' && diffusionFilename) {
            node.inputs.unet_name = diffusionFilename;
        }
    }
}

function nextNodeId(prompt) {
    const numericIds = Object.keys(prompt).map(Number).filter(Number.isFinite);
    return String((numericIds.length ? Math.max(...numericIds) : 0) + 1);
}

function sameReference(value, reference) {
    return Array.isArray(value)
        && value.length >= 2
        && String(value[0]) === String(reference[0])
        && Number(value[1]) === Number(reference[1]);
}

function replaceReferences(prompt, oldReference, newReference) {
    for (const node of Object.values(prompt)) {
        for (const [key, value] of Object.entries(node?.inputs || {})) {
            if (sameReference(value, oldReference)) node.inputs[key] = [...newReference];
        }
    }
}

function findLoaderReferences(prompt) {
    const entries = Object.entries(prompt);
    const checkpointLoader = entries.find(([, node]) => normalizedClassType(node?.class_type) === 'checkpointloadersimple');
    if (checkpointLoader) {
        const clipLayer = entries.find(([, node]) => normalizedClassType(node?.class_type) === 'clipsetlastlayer');
        return {
            model: [checkpointLoader[0], 0],
            clip: clipLayer ? [clipLayer[0], 0] : [checkpointLoader[0], 1],
        };
    }

    const modelLoader = entries.find(([, node]) => ['unetloader', 'modelloader'].includes(normalizedClassType(node?.class_type)));
    const clipLoader = entries.find(([, node]) => ['cliploader', 'dualcliploader'].includes(normalizedClassType(node?.class_type)));
    return {
        model: modelLoader ? [modelLoader[0], 0] : null,
        clip: clipLoader ? [clipLoader[0], 0] : null,
    };
}

function isLoraLoaderClass(value) {
    const type = normalizedClassType(value);
    return type.startsWith('loraloader') || type.startsWith('loadlora');
}

function insertLoras(prompt, loras, warnings) {
    const candidates = (Array.isArray(loras) ? loras : [])
        .map(lora => ({ ...lora, workflowFilename: getResourceFilename(lora) }))
        .filter(lora => lora.workflowFilename
            && (!lora.isDeleted || lora.inLibrary || lora._replayRequirement?.required === true));
    const availableLoras = [];
    const seenResources = new Set();
    for (const lora of candidates) {
        const key = loraCompactName(lora.workflowFilename);
        if (!key || seenResources.has(key)) continue;
        seenResources.add(key);
        availableLoras.push(lora);
    }
    if (availableLoras.length === 0) return;

    const existing = Object.values(prompt).filter(node => isLoraLoaderClass(node?.class_type));
    const pendingLoras = [];
    for (const lora of availableLoras) {
        const names = loraCandidateNames(lora).map(loraCompactName).filter(Boolean);
        names.push(loraCompactName(lora.workflowFilename));
        const matched = existing.find(node => {
            const current = loraCompactName(node?.inputs?.lora_name);
            return current && names.includes(current);
        });
        if (!matched) {
            pendingLoras.push(lora);
            continue;
        }
        const strengths = getLoraStrengths(lora);
        matched.inputs.lora_name = lora.workflowFilename;
        if ('strength_model' in matched.inputs) matched.inputs.strength_model = strengths.model;
        if ('strength_clip' in matched.inputs) matched.inputs.strength_clip = strengths.clip;
        matched._meta ||= {};
        matched._meta.lora_aliases = [...new Set(loraCandidateNames(lora).map(String))];
        if (lora._replayRequirement) {
            matched._meta.replay_requirement = { ...lora._replayRequirement };
        }
    }
    if (pendingLoras.length === 0) return;

    const references = findLoaderReferences(prompt);
    if (!references.model || !references.clip) {
        warnings.push('LoRAを接続できるMODEL/CLIPローダーがテンプレート内に見つかりませんでした。');
        return;
    }

    let nextId = nextNodeId(prompt);
    let modelReference = references.model;
    let clipReference = references.clip;
    const originalModelReference = [...references.model];
    const originalClipReference = [...references.clip];

    for (const lora of pendingLoras) {
        const id = nextId;
        nextId = String(Number(nextId) + 1);
        const strengths = getLoraStrengths(lora);
        prompt[id] = {
            inputs: {
                model: [...modelReference],
                clip: [...clipReference],
                lora_name: lora.workflowFilename,
                strength_model: strengths.model,
                strength_clip: strengths.clip,
            },
            class_type: 'LoraLoader',
            _meta: {
                title: `Load LoRA: ${lora.name || lora.workflowFilename}`,
                // Preserve every known recipe-side name through the backend so
                // A1111 tags can be rewritten to the exact ComfyUI library path.
                lora_aliases: [...new Set(loraCandidateNames(lora).map(String))],
                ...(lora._replayRequirement
                    ? { replay_requirement: { ...lora._replayRequirement } }
                    : {}),
            },
        };
        modelReference = [id, 0];
        clipReference = [id, 1];
    }

    const insertedIds = new Set(pendingLoras.map((_, index) => String(Number(nextId) - pendingLoras.length + index)));
    const originalNodes = Object.fromEntries(Object.entries(prompt).filter(([id]) => !insertedIds.has(id)));
    replaceReferences(originalNodes, originalModelReference, modelReference);
    replaceReferences(originalNodes, originalClipReference, clipReference);
}

function objectInfoOutputs(prompt, objectInfo) {
    return Object.entries(prompt).filter(([, node]) => {
        const info = objectInfo?.[node?.class_type];
        return info?.output_node === true
            || ['saveimage', 'previewimage'].includes(normalizedClassType(node?.class_type));
    });
}

function imageSinkCandidates(prompt, objectInfo) {
    return Object.entries(prompt).filter(([, node]) => {
        if (!Array.isArray(node?.inputs?.images)) return false;
        const type = String(node?.class_type || '');
        return !objectInfo?.[type] || /(?:save|saver|output|prompt)/i.test(type);
    });
}

function collectReachableNodeIds(prompt, roots) {
    const reachable = new Set();
    const pending = roots.map(String);
    while (pending.length > 0) {
        const id = pending.pop();
        if (reachable.has(id) || !prompt[id]) continue;
        reachable.add(id);
        for (const value of Object.values(prompt[id]?.inputs || {})) {
            if (Array.isArray(value) && value.length >= 2 && prompt[String(value[0])]) {
                pending.push(String(value[0]));
            }
        }
    }
    return reachable;
}

function embeddedGraphProblems(prompt, objectInfo, rootIds) {
    const reachable = collectReachableNodeIds(prompt, rootIds);
    const missingNodes = new Set();
    const missingInputs = new Set();
    for (const id of reachable) {
        const node = prompt[id];
        const info = objectInfo?.[node?.class_type];
        if (!info) {
            missingNodes.add(String(node?.class_type || 'Unknown'));
            continue;
        }
        const required = info?.input?.required || {};
        for (const key of Object.keys(required)) {
            if (!(key in (node.inputs || {})) || node.inputs[key] === undefined || node.inputs[key] === null) {
                missingInputs.add(`${node.class_type}.${key}`);
            }
        }
    }
    return { reachable, missingNodes: [...missingNodes], missingInputs: [...missingInputs] };
}

function canBuildStandardRecipe(recipe) {
    const hasPrompt = String(recipe?.gen_params?.prompt || '').trim().length > 0;
    const hasModel = Boolean(
        getResourceFilename(recipe?.checkpoint, 'Model')
        || getResourceFilename(recipe?.checkpoint, 'Diffusion Model')
        || filenameFromName(recipe?.gen_params?.model)
    );
    return hasPrompt && hasModel;
}

function isFourChannelCheckpointRecipe(recipe) {
    const family = [
        recipe?.base_model,
        recipe?.checkpoint?.baseModel,
        recipe?.checkpoint?.base_model,
        recipe?.gen_params?.model,
    ].filter(Boolean).join(' ').toLowerCase();
    return /sdxl|illustrious|noobai|pony/.test(family);
}

function repairAmbiguousAeVae(prompt, recipe, warnings) {
    if (!isFourChannelCheckpointRecipe(recipe)) return;
    const checkpointLoaders = Object.entries(prompt).filter(([, node]) =>
        /checkpointloader/i.test(String(node?.class_type || ''))
    );
    if (checkpointLoaders.length !== 1) return;
    const checkpointVae = [checkpointLoaders[0][0], 2];

    for (const [vaeId, node] of Object.entries(prompt)) {
        if (normalizedClassType(node?.class_type) !== 'vaeloader') continue;
        const compactName = String(node?.inputs?.vae_name || '')
            .replace(/\\/g, '/')
            .split('/').at(-1)
            .replace(/\.[^.]+$/, '')
            .replace(/[^a-z0-9]+/gi, '')
            .toLowerCase();
        if (compactName !== 'ae') continue;

        let replaced = false;
        for (const otherNode of Object.values(prompt)) {
            for (const [key, value] of Object.entries(otherNode?.inputs || {})) {
                if (!sameReference(value, [vaeId, 0])) continue;
                otherNode.inputs[key] = [...checkpointVae];
                replaced = true;
            }
        }
        if (!replaced) continue;
        delete prompt[vaeId];
        warnings.push(
            'SDXL系の4ch潜在に対して16chのFlux系 ae.safetensors が選ばれる曖昧なVAE指定を検出し、チェックポイント内蔵VAEへ置換しました。'
        );
    }
}

function validateOrRepairEmbeddedPrompt(prompt, recipe, objectInfo, warnings) {
    if (!objectInfo || typeof objectInfo !== 'object') return { prompt, rebuilt: false };

    let roots = objectInfoOutputs(prompt, objectInfo).map(([id]) => id);
    const sinks = imageSinkCandidates(prompt, objectInfo);
    const unusableSinks = sinks.filter(([, node]) => objectInfo?.[node.class_type]?.output_node !== true);
    let addedOutput = false;
    if (roots.length === 0 || unusableSinks.length > 0) {
        const candidate = (unusableSinks.at(-1) || sinks.at(-1));
        if (candidate) {
            const id = nextNodeId(prompt);
            prompt[id] = {
                inputs: {
                    filename_prefix: `Recipe_${recipe?.title || recipe?.id || 'ComfyUI'}`,
                    images: [...candidate[1].inputs.images],
                },
                class_type: 'SaveImage',
                _meta: { title: 'Save Image (repaired output)' },
            };
            roots = [id];
            addedOutput = true;
        }
    }

    if (roots.length === 0) {
        throw new Error('元ワークフローに画像出力がなく、出力元も特定できません（Prompt has no outputs）。');
    }

    const problems = embeddedGraphProblems(prompt, objectInfo, roots);
    if (problems.missingNodes.length > 0 || problems.missingInputs.length > 0) {
        const details = [
            problems.missingNodes.length ? `不足ノード: ${problems.missingNodes.join('、')}` : '',
            problems.missingInputs.length ? `不足入力: ${problems.missingInputs.join('、')}` : '',
        ].filter(Boolean).join(' / ');
        if (!canBuildStandardRecipe(recipe)) {
            throw new Error(`元ワークフローを実行できません。${details}`);
        }
        warnings.push(`${details} を検出したため、保存済み生成条件から標準構成へ再構築しました。`);
        return { prompt: standardPrompt(recipe), rebuilt: true };
    }

    if (addedOutput) {
        for (const id of Object.keys(prompt)) {
            if (!problems.reachable.has(id)) delete prompt[id];
        }
        warnings.push('元ワークフローの画像出力ノードが無効だったため、標準SaveImage出力を補完しました。');
    }
    return { prompt, rebuilt: false };
}

function optimizeSingleBatchSlice(prompt, warnings) {
    for (const [sliceId, sliceNode] of Object.entries(prompt)) {
        if (normalizedClassType(sliceNode?.class_type) !== 'latentfrombatch') continue;
        const sourceRef = sliceNode?.inputs?.samples;
        const batchIndex = Number(sliceNode?.inputs?.batch_index);
        const length = Number(sliceNode?.inputs?.length ?? 1);
        if (!Array.isArray(sourceRef) || !Number.isInteger(batchIndex) || batchIndex < 0 || length !== 1) continue;
        const sourceNode = prompt[String(sourceRef[0])];
        if (!sourceNode || !/Empty.*LatentImage/i.test(sourceNode.class_type || '')) continue;
        if (Number(sourceNode?.inputs?.batch_size) <= 1) continue;

        for (const node of Object.values(prompt)) {
            for (const [key, value] of Object.entries(node?.inputs || {})) {
                if (!sameReference(value, [sliceId, 0])) continue;
                node.inputs[key] = [...sourceRef];
                if (/KSampler/i.test(node.class_type || '')) {
                    if (Number.isFinite(Number(node.inputs.seed))) node.inputs.seed = normalizeSeed(node.inputs.seed) + batchIndex;
                    if (Number.isFinite(Number(node.inputs.noise_seed))) node.inputs.noise_seed = normalizeSeed(node.inputs.noise_seed) + batchIndex;
                }
            }
        }
        sourceNode.inputs.batch_size = 1;
        delete prompt[sliceId];
        warnings.push(`バッチ${batchIndex + 1}枚目だけを使う構成を単一バッチへ最適化し、同じシード系列を維持しました。`);
    }
}

export function buildRecipeWorkflow(recipe, options = {}) {
    if (!recipe || typeof recipe !== 'object') throw new Error('Recipe data is required');

    const warnings = [];
    const replayManifest = getReplayManifest(recipe, options);
    const rawA1111Parameters = findA1111Parameters(recipe);
    const promptLoras = extractPromptLoras(recipe?.gen_params?.prompt);
    const manifestLoras = requiredManifestLoras(replayManifest);
    const effectiveRecipe = {
        ...recipe,
        gen_params: {
            ...(recipe.gen_params || {}),
            prompt: cleanPromptText(promptLoras.text),
            negative_prompt: cleanPromptText(recipe?.gen_params?.negative_prompt),
        },
        loras: manifestLoras ?? mergePromptLoras(
            applyA1111LoraWeights(recipe.loras, rawA1111Parameters),
            promptLoras.loras,
            { promptAuthoritative: Boolean(rawA1111Parameters) }
        ),
    };
    let source = 'standard';
    let prompt = findEmbeddedPrompt(effectiveRecipe);
    let a1111Parameters = null;

    if (prompt) {
        if (embeddedPromptNeedsRebuild(prompt, effectiveRecipe)) {
            prompt = standardPrompt(effectiveRecipe);
            warnings.push('Fluxの拡散モデルをCheckpoint Loaderへ接続した互換性のない構成を検出し、標準Flux構成へ再構築しました。');
        } else {
            source = 'embedded';
            inlineLegacyConstants(prompt, warnings);
            repairAmbiguousAeVae(prompt, effectiveRecipe, warnings);
            const validated = validateOrRepairEmbeddedPrompt(
                prompt,
                effectiveRecipe,
                options.objectInfo,
                warnings
            );
            prompt = validated.prompt;
            if (validated.rebuilt) source = 'standard';
        }
    } else if (rawA1111Parameters) {
        const features = a1111CompatibilityFeatures(rawA1111Parameters);
        const originalPrompt = effectiveRecipe.gen_params.prompt;
        effectiveRecipe.gen_params.prompt = String(originalPrompt || '')
            .replace(/<segment\b[^>]*>/gi, ' ')
            .replace(/\s{2,}/g, ' ')
            .trim();
        prompt = standardPrompt(effectiveRecipe);
        if (isFluxRecipe(effectiveRecipe) || requiresStructuredA1111(rawA1111Parameters)) {
            const detected = features.length ? `（${features.join('、')}）` : '';
            warnings.push(`A1111固有情報${detected}を検出しました。VAE・hires・CLIP Skipは可能な範囲で再構築し、未対応拡張は除外したため、完全再現ではなく質感や構図が変わる可能性があります。`);
        } else {
            // Clean, complete A1111 metadata can use ComfyUI's native importer.
            source = 'a1111';
            a1111Parameters = rawA1111Parameters;
        }
    } else {
        prompt = findCheckpointTemplate(effectiveRecipe);
        if (prompt) {
            source = 'checkpoint-template';
            warnings.push('元画像の完全なワークフローがないため、チェックポイントの互換テンプレートから再構築しました。補助CLIP/VAEが未所持の場合は該当ノードが赤表示になります。');
        } else {
            prompt = standardPrompt(effectiveRecipe);
            warnings.push('元画像の完全なワークフローがないため、標準のtxt2img構成から再構築しました。');
        }
    }

    patchCheckpoint(prompt, effectiveRecipe.checkpoint);
    patchGenerationParameters(prompt, effectiveRecipe);
    // A manifest-backed embedded graph is evidence, not a template. Never
    // inject a new branch; strict audit will reject missing/disconnected LoRAs.
    if (!(replayManifest && source === 'embedded')) {
        insertLoras(prompt, effectiveRecipe.loras, warnings);
    }
    // Standard/A1111 reconstruction can also create a VAELoader from an
    // ambiguous `VAE: ae` value. Run the 4ch checkpoint guard after every
    // construction path, not only when an embedded Comfy prompt was found.
    repairAmbiguousAeVae(prompt, effectiveRecipe, warnings);
    optimizeSingleBatchSlice(prompt, warnings);

    return {
        prompt,
        source,
        warnings,
        a1111Parameters,
        a1111Checkpoint: a1111Parameters
            ? (getResourceFilename(effectiveRecipe.checkpoint, 'Model') || null)
            : null,
        replayManifest,
    };
}
