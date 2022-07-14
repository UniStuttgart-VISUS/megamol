#version 430

// Data passed from previous shader stages
in vec3 world_normal;

// Deferred rendering typically uses a g-buffer with three color attachments:
// RGBA surface color (aka albedo), XYZ surface normal and non-linear depth.
// If you use different render targets this can be different.
layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main(void) {   
   albedo_out = vec4(0.6,0.6,0.6,1.0);
   // Note that deferred lighting and post processing modules of compositing_gl plugin expect world space normals
   normal_out = world_normal;
   // Typically, we store non-linear depth as supplied by built-in gl_FragCoord.z in the g-Buffer
   depth_out = gl_FragCoord.z;
}
