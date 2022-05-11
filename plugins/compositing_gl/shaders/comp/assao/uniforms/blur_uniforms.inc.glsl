uniform sampler2D g_BlurInput;

layout(rg8, binding = 0) uniform writeonly image2D g_PingPongHalfResultB;
