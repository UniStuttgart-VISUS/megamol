uniform sampler2D g_ViewspaceDepthSource;
uniform sampler2D g_ViewSpaceDepthSourceDepthTapSampler;
uniform sampler2D g_NormalmapSource;

layout(rgba8, binding = 0) uniform writeonly image2D g_PingPongHalfResultA;


// normal texture muss float texture sein, sonst umrechnungsfehler bei loadnormal/decodenormal mit *(-2) + 1
