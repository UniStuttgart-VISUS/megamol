uniform sampler2D g_DepthSource;
uniform sampler2D g_DepthSourcePointClamp;
// halfDepthsMipViews
// -----------------------
layout(r16f, binding = 0) uniform writeonly image2D g_HalfDepthsMipView0;
layout(r16f, binding = 3) uniform writeonly image2D g_HalfDepthsMipView3;
// -----------------------

// unused when normals are provided by module
layout(rgba8, binding = 4) uniform writeonly image2D g_NormalsOutputUAV;
