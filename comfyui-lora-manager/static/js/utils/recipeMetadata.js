const REPLAY_METADATA_KEYS = [
    'a1111_parameters',
    'comfy_prompt',
    'comfy_workflow',
    'generation_metadata',
    'generation_source',
    'generation_source_policy',
];

function buildCompleteRecipeMetadata(recipeData = {}, sourcePath = null) {
    const metadata = {
        base_model: recipeData.base_model || '',
        loras: Array.isArray(recipeData.loras) ? recipeData.loras : [],
        embeddings: Array.isArray(recipeData.embeddings) ? recipeData.embeddings : [],
        gen_params: recipeData.gen_params || {},
        raw_metadata: recipeData.raw_metadata || {},
    };

    for (const key of REPLAY_METADATA_KEYS) {
        const value = recipeData[key];
        if (value !== undefined && value !== null && value !== '' && value !== false) {
            metadata[key] = value;
        }
    }

    const checkpoint = recipeData.checkpoint || recipeData.model || recipeData.gen_params?.checkpoint;
    if (checkpoint && typeof checkpoint === 'object') metadata.checkpoint = checkpoint;
    if (typeof sourcePath === 'string' && sourcePath.trim()) metadata.source_path = sourcePath.trim();

    const nsfwLevel = recipeData.preview_nsfw_level;
    if (nsfwLevel !== undefined && nsfwLevel !== null) {
        metadata.preview_nsfw_level = nsfwLevel;
    }
    return metadata;
}

export { REPLAY_METADATA_KEYS, buildCompleteRecipeMetadata };
