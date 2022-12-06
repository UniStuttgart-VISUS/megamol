uniform sampler2D g_DepthSource;
uniform sampler2D g_NormalmapSource;
uniform sampler2D g_ViewSpaceDepthSourceDepthTapSampler;
uniform sampler2D g_ViewspaceDepthSource;
uniform sampler2D g_ViewspaceDepthSource1;
uniform sampler2D g_ViewspaceDepthSource2;
uniform sampler2D g_ViewspaceDepthSource3;
uniform sampler2D g_ImportanceMap;
uniform sampler2D g_LoadCounter;
uniform sampler2D g_BlurInput;
uniform sampler2DArray g_FinalSSAO;
// halfDepthsMipViews
// -----------------------
layout(r16f, binding = 0) uniform image2D      g_HalfDepthsMipView0;
layout(r16f, binding = 1) uniform image2D      g_HalfDepthsMipView1;
layout(r16f, binding = 2) uniform image2D      g_HalfDepthsMipView2;
layout(r16f, binding = 3) uniform image2D      g_HalfDepthsMipView3;
// -----------------------

// halfDepths
// -----------------------
layout(r16f, binding = 0) uniform image2D      g_HalfDepths0;
layout(r16f, binding = 1) uniform image2D      g_HalfDepths1;
layout(r16f, binding = 2) uniform image2D      g_HalfDepths2;
layout(r16f, binding = 3) uniform image2D      g_HalfDepths3;
// -----------------------

layout(rg8, binding = 4) uniform image2D      g_PingPongHalfResultA;
layout(rg8, binding = 5) uniform image2D      g_PingPongHalfResultB;

layout(r16f, binding = 6) uniform image2D      g_FinalOutput;

// unused when normals are provided by module
layout(rgba8, binding = 7) uniform image2D      g_NormalsOutputUAV;

// only needed for adaptive SSAO
// -----------------------
//layout(r32ui, binding = 8) uniform image1D      g_LoadCounterOutputUAV;
// -----------------------
