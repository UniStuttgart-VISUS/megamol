uniform sampler2D g_DepthSource;
uniform sampler2D g_NormalmapSource;
uniform sampler2D g_ViewspaceDepthSource;
uniform sampler2D g_ViewspaceDepthSource1;
uniform sampler2D g_ViewspaceDepthSource2;
uniform sampler2D g_ViewspaceDepthSource3;
uniform sampler2D g_ImportanceMap;
uniform sampler2D g_LoadCounter;
uniform sampler2D g_BlurInput;
uniform sampler2D g_FinalSSAO;
layout(rg32f, binding = 0) uniform image2D   g_NormalsOutputUAV;             // unsigned normalized --> [0, 1]
layout(r32ui, binding = 1) uniform image1D   g_LoadCounterOutputUAV;
