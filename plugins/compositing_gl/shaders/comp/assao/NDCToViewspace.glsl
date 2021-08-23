vec3 NDCToViewspace( vec2 pos, float viewspaceDepth )
{
    vec3 ret;

    //ret.xy = (g_ASSAOConsts.NDCToViewMul * pos + g_ASSAOConsts.NDCToViewAdd) * viewspaceDepth;
    ret.xy = g_ASSAOConsts.NDCToViewMul * pos * (-viewspaceDepth);

    ret.z = viewspaceDepth;

    return ret;
}
