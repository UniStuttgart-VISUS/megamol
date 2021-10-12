///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

// CAUTION: this contains an implicit conversion from normalized screenCoordinates to NDCCoordinates
// hidden in NDCToViewMul and NDCToViewAdd
vec3 NDCToViewspace( vec2 pos, float viewspaceDepth )
{
    vec3 ret;

    //ret.xy = (g_ASSAOConsts.NDCToViewMul * pos + g_ASSAOConsts.NDCToViewAdd) * viewspaceDepth;
    ret.xy = (g_ASSAOConsts.NDCToViewMul * pos + g_ASSAOConsts.NDCToViewAdd) * (-viewspaceDepth);

    //ret.xy = g_ASSAOConsts.NDCToViewMul * pos * (-viewspaceDepth);

    ret.z = viewspaceDepth;

    return ret;
}
