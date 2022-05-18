uniform sampler2D g_ViewSpaceDepthSource;
uniform sampler2D g_NormalmapSource;
uniform sampler2D g_ViewSpaceDepthSourceDepthTapSampler;

layout(rg8, binding = 0) uniform writeonly image2D g_PingPongHalfResultA;
