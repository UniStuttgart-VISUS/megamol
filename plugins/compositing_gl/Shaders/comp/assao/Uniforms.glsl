uniform sampler2D g_DepthSource;
uniform sampler2D g_NormalmapSource;
uniform sampler2D g_ViewspaceDepthSource;
uniform sampler2D g_ViewspaceDepthSource1;
uniform sampler2D g_ViewspaceDepthSource2;
uniform sampler2D g_ViewspaceDepthSource3;
uniform sampler2D g_ImportanceMap;
uniform sampler2D g_LoadCounter;
uniform sampler2D g_BlurInput;
// TODO: set correct layouts
// halfDepthsMipViews
// -----------------------
layout(rg32f, binding = 0) uniform image2D      g_HalfDepthsMipView0;
layout(rg32f, binding = 1) uniform image2D      g_HalfDepthsMipView1;
layout(rg32f, binding = 2) uniform image2D      g_HalfDepthsMipView2;
layout(rg32f, binding = 3) uniform image2D      g_HalfDepthsMipView3;
// -----------------------

// halfDepths
// -----------------------
layout(rg32f, binding = 0) uniform image2D      g_HalfDepths0;
layout(rg32f, binding = 1) uniform image2D      g_HalfDepths1;
layout(rg32f, binding = 2) uniform image2D      g_HalfDepths2;
layout(rg32f, binding = 3) uniform image2D      g_HalfDepths3;
// -----------------------

layout(rg32f, binding = 4) uniform image2D      g_PingPongHalfResultA;
layout(rg32f, binding = 5) uniform image2D      g_PingPongHalfResultB;

layout(rg32f, binding = 6) uniform image2D      g_FinalOutput;
layout(rg32f, binding = 7) uniform image2DArray g_FinalSSAO;

// only needed for adaptive SSAO
// -----------------------
layout(rg32f, binding = 8) uniform image2D      g_NormalsOutputUAV;             // unsigned normalized --> [0, 1]
layout(r32ui, binding = 9) uniform image1D      g_LoadCounterOutputUAV;
// -----------------------
