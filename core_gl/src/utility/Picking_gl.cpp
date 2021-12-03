/*
 * Picking_gl.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "mmcore/utility/Picking.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/RenderUtils.h"

using namespace megamol::core::utility;

#ifdef WITH_GL

#define PICKING_GL_CHECK_ERROR                                                                    \
    {                                                                                             \
        auto err = glGetError();                                                                  \
        if (err != 0)                                                                             \
            megamol::core::utility::log::Log::DefaultLog.WriteError(                              \
                "OpenGL Error: %i. [%s, %s, line %d]\n ", err, __FILE__, __FUNCTION__, __LINE__); \
    }

PickingBuffer::~PickingBuffer() {

    delete this->fbo;
}

bool PickingBuffer::EnableInteraction(glm::vec2 vp_dim) {

    if (this->enabled) {
        log::Log::DefaultLog.WriteError(
            "[GL Picking Buffer] Disable interaction before enabling again. [%s, %s, line %d]\n ", __FILE__,
            __FUNCTION__, __LINE__);
        return true;
    }

    // Enable interaction only if interactions have been added in previous frame
    if (this->available_interactions.empty()) {
        return false;
    }

    // Interactions are processed in ProcessMouseMove() and should be cleared each frame
    this->available_interactions.clear();
    this->enabled = false;
    this->viewport_dim = glm::ivec2(static_cast<int>(vp_dim.x), static_cast<int>(vp_dim.y));

    if (this->fbo == nullptr) {
        try {
            this->fbo = new glowl::FramebufferObject(
                this->viewport_dim.x, this->viewport_dim.y, glowl::FramebufferObject::DepthStencilType::NONE);
        } catch (glowl::FramebufferObjectException& e) {
            log::Log::DefaultLog.WriteError(
                "[GL Picking Buffer] Error during framebuffer object creation: '%s'. [%s, %s, line %d]\n ", e.what(),
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        this->fbo->createColorAttachment(GL_RGBA32F, GL_RGBA, GL_FLOAT); // 0 Output Image
        this->fbo->createColorAttachment(GL_RG32F, GL_RG, GL_FLOAT);     // 1 Object ID(red) and Depth (green)
        PICKING_GL_CHECK_ERROR
    } else if (this->fbo->getWidth() != this->viewport_dim.x || this->fbo->getHeight() != this->viewport_dim.y) {
        this->fbo->resize(this->viewport_dim.x, this->viewport_dim.y);
    }

    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &this->prev_fbo);

    this->fbo->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GLint in[1] = {0};
    glClearBufferiv(GL_COLOR, 1, in);
    PICKING_GL_CHECK_ERROR

    this->enabled = true;
    return this->enabled;
}


bool PickingBuffer::DisableInteraction() {

    if (!this->enabled) {
        // log::Log::DefaultLog.WriteError(
        //    "[GL Picking Buffer] Enable interaction before disabling it. [%s, %s, line %d]\n ", __FILE__,
        //    __FUNCTION__, __LINE__);
        return false;
    }
    this->enabled = false;

    // Clear pending manipulations
    this->pending_manipulations.clear();

    std::shared_ptr<glowl::GLSLProgram> shader;
    // Create FBO sahders if required -----------------------------------------
    if (this->fbo_shader == nullptr) {
        std::string vertex_src = "#version 130 \n "
                                 "out vec2 uv_coord; \n "
                                 "void main() { \n "
                                 "    const vec4 vertices[6] = vec4[6](vec4(-1.0, -1.0, 0.0, 0.0), \n "
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n "
                                 "        vec4(-1.0, 1.0, 0.0, 1.0), \n "
                                 "        vec4(1.0, 1.0, 1.0, 1.0), \n "
                                 "        vec4(-1.0, -1.0, 0.0, 0.0), \n "
                                 "        vec4(1.0, -1.0, 1.0, 0.0)); \n "
                                 "    vec4 vertex = vertices[gl_VertexID]; \n "
                                 "    uv_coord = vertex.zw; \n "
                                 "    gl_Position = vec4(vertex.xy, -1.0, 1.0); \n "
                                 "} ";

        std::string fragment_src = "#version 130  \n "
                                   "#extension GL_ARB_explicit_attrib_location : require \n "
                                   "in vec2 uv_coord; \n "
                                   "uniform sampler2D col_tex; \n "
                                   "uniform sampler2D depth_tex; \n "
                                   "layout(location = 0) out vec4 outFragColor; \n "
                                   "void main() { \n "
                                   "    vec4 color = texture(col_tex, uv_coord).rgba; \n "
                                   "    if (color == vec4(0.0)) discard; \n "
                                   "    float depth = texture(depth_tex, uv_coord).g; \n "
                                   "    gl_FragDepth = depth; \n "
                                   "    outFragColor = color; \n "
                                   "} ";
        if (!megamol::core_gl::utility::RenderUtils::CreateShader(shader, vertex_src, fragment_src))
            return false;
        this->fbo_shader = shader.get();
    }

    PICKING_GL_CHECK_ERROR
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Bind fbo to read buffer for retrieving pixel data
    GLfloat pixel_data[2] = {-1.0f, FLT_MAX};
    this->fbo->bindToRead(1);
    PICKING_GL_CHECK_ERROR
    // Get object id and depth at cursor location from framebuffer's second color attachment
    /// TODO Check if cursor position is within framebuffer pixel range -> ensured by GLFW?
    glReadPixels(static_cast<GLint>(this->cursor_x), this->fbo->getHeight() - static_cast<GLint>(this->cursor_y), 1, 1,
        GL_RG, GL_FLOAT, pixel_data);
    PICKING_GL_CHECK_ERROR
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto id = static_cast<unsigned int>(pixel_data[0]);
    auto depth = pixel_data[1];

    if (id > 0) {
        this->cursor_on_interaction_obj = {true, id, depth};
        this->pending_manipulations.emplace_back(Manipulation{InteractionType::HIGHLIGHT, id, 0.0f, 0.0f, 0.0f, 0.0f});
        /// megamol::core::utility::log::Log::DefaultLog.WriteError("[[[DEBUG]]] ID = %i | Depth = %f", id, depth);
    } else {
        this->cursor_on_interaction_obj = PICKING_INTERACTION_TUPLE_INIT;
    }

    // Draw fbo color buffer as texture because blending is required
    glBindFramebuffer(GL_FRAMEBUFFER, this->prev_fbo);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    this->fbo_shader->use();

    glActiveTexture(GL_TEXTURE0);
    this->fbo->bindColorbuffer(0);

    glActiveTexture(GL_TEXTURE1);
    this->fbo->bindColorbuffer(1);

    this->fbo_shader->setUniform("col_tex", 0);
    this->fbo_shader->setUniform("depth_tex", 1);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}
#endif
