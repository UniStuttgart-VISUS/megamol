#version 440

// Extension is required for using gl_DrawIDARB
#extension GL_ARB_shader_draw_parameters : require

// Include these uniforms for view matrix and projection matrix if needed.
// They are always set by RenderMDIMesh module using these names.
uniform mat4 view_mx;
uniform mat4 proj_mx;

// Any additional data needed in a shader needs to be put into shader storage buffers (SSBOs).
// Each render task binds an SSBO to binding point 0. I recommend using an array of structs for
// this render task data, using a struct that keeps all data needed by each indirect draw executed
// by MultiDrawIndirect in RenderMDIMesh as part of the render task.
struct MeshShaderParams
{
    mat4 transform;
};
layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

// You can store addtional per frame data in multiple additional SSBOs. These will be bound once per frame
// and can be accessed by all render tasks and draw calls. Used binding points are user defined. You have
// to make sure yourself that shader and control code match.
//struct PerFrameData { float d; };
//layout(std430, binding = 1) readonly buffer PerFrameDataBuffer { PerFrameData[] per_frame_data; };

// Vertex shader inputs need to match your mesh data, there is no fixed vertex layout used by mesh plugin
layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;

// Passing values from vertex to fragment shader stage
out vec3 world_normal;

void main()
{
   // Use gl_DrawIDARB to access per draw data, e.g. objects transforms
   mat4 object_transform = mesh_shader_params[gl_DrawIDARB].transform;
   // Deferred rendering and post processing modules of compositing_gl plugin expect world space normals
   world_normal = inverse(transpose(mat3(object_transform))) * v_normal;

   // Standard model view projection transformation of vertices.
   // What you want to do with your shader might differ.
   gl_Position =  proj_mx * view_mx * object_transform * vec4(v_position,1.0);
}
