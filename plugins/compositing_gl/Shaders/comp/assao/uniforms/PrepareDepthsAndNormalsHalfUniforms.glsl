uniform sampler2D g_DepthSource;
// halfDepthsMipViews
// -----------------------
layout(r16f, binding = 0) uniform image2D g_HalfDepthsMipView0;
layout(r16f, binding = 3) uniform image2D g_HalfDepthsMipView3;
// -----------------------

// unused when normals are provided by module
layout(rgba8, binding = 7) uniform image2D      g_NormalsOutputUAV;
