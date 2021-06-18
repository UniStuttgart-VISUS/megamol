uniform sampler2D g_ViewspaceDepthSource;
uniform sampler2D g_ViewSpaceDepthSourceDepthTapSampler;
uniform sampler2D g_NormalmapSource;

layout(rg8, binding = 0) uniform image2D g_PingPongHalfResultA;
