import { resolveSamplerScheduler } from '/loras_static/js/utils/genParamsMapper.js';

const OPTION_LABELS = [
    'Steps', 'Sampler', 'Schedule type', 'CFG scale', 'Seed', 'Size', 'Model',
    'Model hash', 'VAE', 'VAE hash', 'Clip skip', 'Denoising strength',
    'Hires CFG Scale', 'Hires CFG scale', 'Hires upscale', 'Hires resize',
    'Hires steps', 'Hires upscaler', 'Version',
];

function metadataTextScore(text) {
    const markerScore = Number(text.includes('\nSteps:')) + Number(text.includes('\nNegative prompt:'));
    let ascii = 0;
    let suspicious = 0;
    for (const char of text) {
        const code = char.charCodeAt(0);
        if ((code >= 32 && code < 127) || '\r\n\t'.includes(char)) ascii += 1;
        if (code === 0 || code > 0x2fff) suspicious += 1;
    }
    return markerScore * 1_000_000 + ascii - suspicious;
}

function normalizeA1111Text(value) {
    if (typeof value !== 'string' || !value) return value;
    const candidates = [value, value.replaceAll('\0', '')];
    let swapped = '';
    for (const char of value) {
        const code = char.charCodeAt(0);
        swapped += String.fromCharCode(((code & 0xff) << 8) | (code >>> 8));
    }
    candidates.push(swapped, swapped.replaceAll('\0', ''));
    return candidates.reduce((best, candidate) => (
        metadataTextScore(candidate) > metadataTextScore(best) ? candidate : best
    ), value);
}

function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function parseA1111Options(parameters) {
    const normalized = normalizeA1111Text(parameters);
    if (typeof normalized !== 'string') return {};
    const stepsIndex = normalized.lastIndexOf('\nSteps:');
    if (stepsIndex < 0) return {};
    const section = normalized.slice(stepsIndex + 1);
    const labelsPattern = OPTION_LABELS.map(escapeRegExp).join('|');
    const result = {};
    for (const label of OPTION_LABELS) {
        const pattern = new RegExp(
            `(?:^|,\\s*)${escapeRegExp(label)}\\s*:\\s*(.*?)(?=,\\s*(?:${labelsPattern})\\s*:|$)`,
            'i',
        );
        const match = section.match(pattern);
        if (match) result[label.toLowerCase()] = match[1].trim();
    }
    return result;
}

function parseSize(value) {
    if (typeof value !== 'string') return null;
    const match = value.match(/(\d+)\s*x\s*(\d+)/i);
    if (!match) return null;
    const width = Number(match[1]);
    const height = Number(match[2]);
    return width > 0 && height > 0 ? { width, height } : null;
}

function graphNodes(graph) {
    if (Array.isArray(graph?._nodes)) return graph._nodes;
    if (Array.isArray(graph?.nodes)) return graph.nodes;
    return [];
}

function nodeClass(node) {
    return node?.comfyClass || node?.type || node?.constructor?.comfyClass || '';
}

function setWidget(node, names, value) {
    if (value === undefined || value === null || value === '') return false;
    const widget = node?.widgets?.find(item => names.includes(String(item?.name || '').toLowerCase()));
    if (!widget) return false;
    widget.value = value;
    widget.callback?.(value, node, widget);
    return true;
}

function numeric(value) {
    if (value === undefined || value === null || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function patchA1111Graph(graph, parameters, { checkpointName = null } = {}) {
    const normalized = normalizeA1111Text(parameters);
    const options = parseA1111Options(normalized);
    const nodes = graphNodes(graph);

    const samplerText = options.sampler;
    const scheduleText = options['schedule type'];
    const samplerResolved = resolveSamplerScheduler(samplerText);
    const schedulerResolved = resolveSamplerScheduler(scheduleText);
    const resolved = {
        sampler: samplerResolved.sampler,
        scheduler: schedulerResolved.scheduler || samplerResolved.scheduler,
    };
    const samplers = nodes.filter(node => nodeClass(node) === 'KSampler');
    samplers.forEach((node, index) => {
        setWidget(node, ['sampler_name', 'sampler'], resolved.sampler);
        setWidget(node, ['scheduler'], resolved.scheduler);
        setWidget(node, ['seed', 'noise_seed'], numeric(options.seed));
        const hiresSteps = numeric(options['hires steps']);
        setWidget(node, ['steps'], index > 0 && hiresSteps !== null ? hiresSteps : numeric(options.steps));
        const hiresCfg = numeric(options['hires cfg scale']);
        setWidget(node, ['cfg'], index > 0 && hiresCfg !== null ? hiresCfg : numeric(options['cfg scale']));
        if (index > 0) {
            setWidget(node, ['denoise'], numeric(options['denoising strength']));
        }
    });

    const baseSize = parseSize(options.size);
    if (baseSize) {
        for (const node of nodes.filter(item => nodeClass(item) === 'EmptyLatentImage')) {
            setWidget(node, ['width'], baseSize.width);
            setWidget(node, ['height'], baseSize.height);
        }
    }

    let targetSize = parseSize(options['hires resize']);
    const upscale = numeric(options['hires upscale']);
    if (!targetSize && baseSize && upscale !== null) {
        targetSize = {
            width: Math.round(baseSize.width * upscale),
            height: Math.round(baseSize.height * upscale),
        };
    }
    if (targetSize) {
        for (const node of nodes.filter(item => ['ImageScale', 'LatentUpscale'].includes(nodeClass(item)))) {
            setWidget(node, ['width'], targetSize.width);
            setWidget(node, ['height'], targetSize.height);
        }
    }

    const modelName = checkpointName || options.model;
    if (modelName) {
        for (const node of nodes.filter(item => ['CheckpointLoaderSimple', 'UNETLoader'].includes(nodeClass(item)))) {
            setWidget(node, ['ckpt_name', 'unet_name'], modelName);
        }
    }

    return { parameters: normalized, options, baseSize, targetSize, sampler: resolved };
}

export { normalizeA1111Text, parseA1111Options, patchA1111Graph };
