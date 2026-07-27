function sanitizeWorkflowName(value) {
    return String(value || '')
        .replace(/[^a-zA-Z0-9_-]+/g, '_')
        .replace(/^_+|_+$/g, '');
}

export function createRecipeWorkflowName(recipe) {
    const title = String(recipe?.title || recipe?.name || '');
    const civitaiRecipe = title.match(/Civitai[\s_-]*Recipe[\s_-]*(\d+)/i);
    if (civitaiRecipe) return `Civitai_Recipe_${civitaiRecipe[1]}`;

    const civitaiImageId = [
        recipe?.civitai_image_id,
        recipe?.civitaiImageId,
        recipe?.image_id,
        recipe?.imageId,
    ].find(value => /^\d+$/.test(String(value || '')));
    if (civitaiImageId) return `Civitai_Recipe_${civitaiImageId}`;

    const titleName = sanitizeWorkflowName(title);
    if (titleName) return titleName;

    const identifier = sanitizeWorkflowName(recipe?.id || recipe?.recipe_id);
    return identifier ? `Recipe_${identifier}` : 'Recipe_workflow';
}
