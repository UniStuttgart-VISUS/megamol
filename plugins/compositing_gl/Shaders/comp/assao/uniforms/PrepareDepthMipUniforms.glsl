uniform sampler2D g_ViewspaceDepthSource;
uniform sampler2D g_ViewspaceDepthSource1;
uniform sampler2D g_ViewspaceDepthSource2;
uniform sampler2D g_ViewspaceDepthSource3;
// halfDepthsMipViews
// -----------------------
layout(r16f, binding = 0) uniform image2D      g_HalfDepthsMipView0;
layout(r16f, binding = 1) uniform image2D      g_HalfDepthsMipView1;
layout(r16f, binding = 2) uniform image2D      g_HalfDepthsMipView2;
layout(r16f, binding = 3) uniform image2D      g_HalfDepthsMipView3;
// -----------------------
