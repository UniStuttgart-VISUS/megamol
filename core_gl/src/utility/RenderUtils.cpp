/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore_gl/utility/RenderUtils.h"

#include "mmcore_gl/utility/ShaderFactory.h"

using megamol::core::utility::log::Log;

namespace megamol::core_gl::utility {

// STATIC functions -------------------------------------------------------

bool RenderUtils::LoadTextureFromFile(std::shared_ptr<glowl::Texture2D>& out_texture_ptr,
    const std::filesystem::path& filename, GLint tex_min_filter, GLint tex_max_filter) {

    if (filename.empty())
        return false;
    bool retval = false;

    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    std::vector<char> buf;
    size_t size = 0;

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (megamol::core::utility::FileUtils::LoadRawFile(filename, buf)) {
        if (pbc.Load(static_cast<void*>(buf.data()), buf.size())) {
            img.Convert(vislib::graphics::BitmapImage::TemplateFloatRGBA);
            retval = RenderUtils::LoadTextureFromData(
                out_texture_ptr, img.Width(), img.Height(), img.PeekDataAs<FLOAT>(), tex_min_filter, tex_max_filter);
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to read texture: %s [%s, %s, line %d]\n", filename.string(), __FILE__, __FUNCTION__, __LINE__);
            retval = false;
        }
    } else {
        retval = false;
    }

    return retval;
}


bool RenderUtils::LoadTextureFromData(std::shared_ptr<glowl::Texture2D>& out_texture_ptr, int width, int height,
    float* data, GLint tex_min_filter, GLint tex_max_filter) {

    if (data == nullptr)
        return false;
    try {
        std::vector<std::pair<GLenum, GLint>> int_parameters;
        int_parameters.push_back({GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE});
        int_parameters.push_back({GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE});
        int_parameters.push_back({GL_TEXTURE_MIN_FILTER, tex_min_filter});
        int_parameters.push_back({GL_TEXTURE_MAG_FILTER, tex_max_filter});
        std::vector<std::pair<GLenum, GLfloat>> float_parameters;
        glowl::TextureLayout tex_layout(
            GL_RGBA32F, width, height, 1, GL_RGBA, GL_FLOAT, 1, int_parameters, float_parameters);
        if (out_texture_ptr == nullptr) {
            out_texture_ptr =
                std::make_shared<glowl::Texture2D>("image", tex_layout, static_cast<GLvoid*>(data), false);
        } else {
            // Reload data
            out_texture_ptr->reload(tex_layout, static_cast<GLvoid*>(data), false);
        }
    } catch (glowl::TextureException& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during texture creation: '%s'. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool RenderUtils::CreateShader(std::shared_ptr<glowl::GLSLProgram>& out_shader_ptr, const std::string& vertex_code,
    const std::string& fragment_code) {

    try {
        if (out_shader_ptr != nullptr)
            out_shader_ptr.reset();
        out_shader_ptr =
            std::make_shared<glowl::GLSLProgram>(RenderUtils::createShaderSource(vertex_code, fragment_code));
    } catch (glowl::GLSLProgramException const& exc) {
        std::string debug_label;
        if (out_shader_ptr != nullptr) {
            debug_label = out_shader_ptr->getDebugLabel();
        }
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error during shader program creation of\"%s\": %s. [%s, %s, line %d]\n ", debug_label.c_str(), exc.what(),
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> RenderUtils::createShaderSource(
    const std::string& vertex_code, const std::string& fragment_code) {

    std::vector<std::pair<glowl::GLSLProgram::ShaderType, std::string>> shader_srcs;
    if (!vertex_code.empty())
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Vertex, vertex_code});
    if (!fragment_code.empty())
        shader_srcs.push_back({glowl::GLSLProgram::ShaderType::Fragment, fragment_code});

    return shader_srcs;
}


// LOCAL functions -------------------------------------------------------


RenderUtils::RenderUtils()
        : smooth(true)
        , init_once(false)
        , vertex_array(0)
        , textures()
        , queues()
        , shaders()
        , buffers() {}


RenderUtils::~RenderUtils() {}


bool RenderUtils::InitPrimitiveRendering(frontend_resources::RuntimeConfig const& runtimeConf) {

    if (this->init_once) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "Primitive rendering has already been initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    // Create shaders
    std::vector<std::pair<GLuint, std::string>> location_name_pairs = {{Buffers::POSITION, "inPosition"},
        {Buffers::COLOR, "inColor"}, {Buffers::TEXTURE_COORD, "inTexture"}, {Buffers::ATTRIBUTES, "inAttributes"}};

    auto const shader_options = core::utility::make_path_shader_options(runtimeConf);

    try {
        this->shaders[Primitives::POINTS] = core::utility::make_glowl_shader(
            "POINTS", shader_options, "core/primitives/points.vert.glsl", "core/primitives/points.frag.glsl");
        this->shaders[Primitives::POINTS]->bindAttribLocations(location_name_pairs);

        this->shaders[Primitives::LINES] = core::utility::make_glowl_shader(
            "LINES", shader_options, "core/primitives/lines.vert.glsl", "core/primitives/lines.frag.glsl");
        this->shaders[Primitives::LINES]->bindAttribLocations(location_name_pairs);

        this->shaders[Primitives::QUADS] = core::utility::make_glowl_shader(
            "QUADS", shader_options, "core/primitives/quads.vert.glsl", "core/primitives/quads.frag.glsl");
        this->shaders[Primitives::QUADS]->bindAttribLocations(location_name_pairs);

        this->shaders[Primitives::COLOR_TEXTURE] = core::utility::make_glowl_shader("COLOR_TEXTURE", shader_options,
            "core/primitives/color_texture.vert.glsl", "core/primitives/color_texture.frag.glsl");
        this->shaders[Primitives::COLOR_TEXTURE]->bindAttribLocations(location_name_pairs);

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("RenderUtils: " + std::string(e.what())).c_str());
        return false;
    }

    // Create buffers
    this->buffers[Buffers::POSITION] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->buffers[Buffers::COLOR] = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->buffers[Buffers::TEXTURE_COORD] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->buffers[Buffers::ATTRIBUTES] =
        std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &this->vertex_array);
    glBindVertexArray(this->vertex_array);

    this->buffers[Buffers::POSITION]->bind();
    glEnableVertexAttribArray(Buffers::POSITION);
    glVertexAttribPointer(Buffers::POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    this->buffers[Buffers::COLOR]->bind();
    glEnableVertexAttribArray(Buffers::COLOR);
    glVertexAttribPointer(Buffers::COLOR, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    this->buffers[Buffers::TEXTURE_COORD]->bind();
    glEnableVertexAttribArray(Buffers::TEXTURE_COORD);
    glVertexAttribPointer(Buffers::TEXTURE_COORD, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    this->buffers[Buffers::ATTRIBUTES]->bind();
    glEnableVertexAttribArray(Buffers::ATTRIBUTES);
    glVertexAttribPointer(Buffers::ATTRIBUTES, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(Buffers::POSITION);
    glDisableVertexAttribArray(Buffers::COLOR);
    glDisableVertexAttribArray(Buffers::TEXTURE_COORD);
    glDisableVertexAttribArray(Buffers::ATTRIBUTES);

    this->init_once = true;

    return true;
}


bool RenderUtils::LoadTextureFromFile(GLuint& out_texture_id, const std::filesystem::path& filename, bool reload) {

    if (reload) {
        auto texture_iter = std::find_if(
            this->textures.begin(), this->textures.end(), [&out_texture_id](std::shared_ptr<glowl::Texture2D> tex_ptr) {
                return (tex_ptr->getName() == out_texture_id);
            });
        if (texture_iter != this->textures.end()) {
            if (RenderUtils::LoadTextureFromFile(*texture_iter, filename)) {
                return true;
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to reload texture. Texture with given id does not exist. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
        }
    } else {
        this->textures.push_back(nullptr);
        if (RenderUtils::LoadTextureFromFile(this->textures.back(), filename)) {
            out_texture_id = this->textures.back()->getName();
            return true;
        } else {
            this->textures.pop_back();
        }
    }

    out_texture_id = 0;
    return false;
}


void RenderUtils::PushPointPrimitive(const glm::vec3& pos_center, float size, const glm::vec3& cam_view,
    const glm::vec3& cam_pos, const glm::vec4& color, bool sort) {

    glm::vec3 distance = (pos_center - cam_pos);
    if (glm::dot(cam_view, distance) < 0.0f)
        return;

    float d = glm::length(distance);
    float radius = size / 2.0f;
    glm::vec3 rad = pos_center + glm::normalize(this->arbitraryPerpendicular(distance)) * radius;
    glm::vec4 attributes = {rad.x, rad.y, rad.z, d};
    this->pushShaderData(Primitives::POINTS, 0, pos_center, color, glm::vec2(0.0f, 0.0f), attributes);

    if (sort) {
        this->sortPrimitiveQueue(Primitives::POINTS);
    }
}


void RenderUtils::PushLinePrimitive(const glm::vec3& pos_start, const glm::vec3& pos_end, float line_width,
    const glm::vec3& cam_view, const glm::vec3& cam_pos, const glm::vec4& color, bool sort) {

    glm::vec3 normal = cam_view * (-1.0f);
    glm::vec3 linedir = (pos_start - pos_end);
    glm::vec3 w = glm::normalize(glm::cross(normal, linedir));
    glm::vec3 p1 = w * (line_width / 2.0f);
    glm::vec3 pos_bottom_left = pos_start - p1;
    glm::vec3 pos_upper_left = pos_start + p1;
    glm::vec3 pos_upper_right = pos_end + p1;
    glm::vec3 pos_bottom_right = pos_end - p1;
    glm::vec4 attributes = {0.0f, 0.0f, 0.0f, 0.0f};
    this->pushQuad(RenderUtils::Primitives::LINES, 0, pos_bottom_left, pos_upper_left, pos_upper_right,
        pos_bottom_right, color, attributes);

    if (sort) {
        this->sortPrimitiveQueue(Primitives::LINES);
    }
}


void RenderUtils::PushQuadPrimitive(const glm::vec3& pos_center, float width, float height, const glm::vec3& cam_view,
    const glm::vec3& cam_up, const glm::vec4& color, bool sort) {

    glm::vec3 normal = cam_view * (-1.0f);
    glm::vec3 p1 = glm::normalize(cam_up);
    glm::vec3 p2 = glm::cross(normal, p1);
    p1 = glm::normalize(p1) * (height / 2.0f);
    p2 = glm::normalize(p2) * (width / 2.0f);
    glm::vec3 pos_bottom_left = pos_center - p1 - p2;
    glm::vec3 pos_upper_left = pos_center + p1 - p2;
    glm::vec3 pos_upper_right = pos_center + p1 + p2;
    glm::vec3 pos_bottom_right = pos_center - p1 + p2;
    glm::vec4 attributes = {0.0f, 0.0f, 0.0f, 0.0f};
    this->pushQuad(RenderUtils::Primitives::QUADS, 0, pos_bottom_left, pos_upper_left, pos_upper_right,
        pos_bottom_right, color, attributes);

    if (sort) {
        this->sortPrimitiveQueue(Primitives::QUADS);
    }
}


void RenderUtils::PushQuadPrimitive(const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left,
    const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, const glm::vec4& color, bool sort) {

    glm::vec4 attributes = {0.0f, 0.0f, 0.0f, 0.0f};
    this->pushQuad(RenderUtils::Primitives::QUADS, 0, pos_bottom_left, pos_upper_left, pos_upper_right,
        pos_bottom_right, color, attributes);

    if (sort) {
        this->sortPrimitiveQueue(Primitives::QUADS);
    }
}


void RenderUtils::Push2DColorTexture(GLuint texture_id, const glm::vec3& pos_bottom_left,
    const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y,
    const glm::vec4& color, bool force_opaque) {

    glm::vec3 pbl = pos_bottom_left;
    glm::vec3 pul = pos_upper_left;
    glm::vec3 pur = pos_upper_right;
    glm::vec3 pbr = pos_bottom_right;
    if (flip_y) {
        pbl.y = pos_upper_left.y;
        pul.y = pos_bottom_left.y;
        pur.y = pos_bottom_right.y;
        pbr.y = pos_upper_right.y;
    }
    glm::vec4 attributes = {((force_opaque) ? (1.0f) : (0.0f)), 0.0f, 0.0f, 0.0f};
    this->pushQuad(RenderUtils::Primitives::COLOR_TEXTURE, texture_id, pbl, pul, pur, pbr, color, attributes);
}


unsigned int RenderUtils::GetTextureWidth(GLuint texture_id) const {

    unsigned int texture_width = 0;
    auto texture_iter = std::find_if(this->textures.begin(), this->textures.end(),
        [&texture_id](std::shared_ptr<glowl::Texture2D> tex_ptr) { return (tex_ptr->getName() == texture_id); });
    if (texture_iter != this->textures.end()) {
        texture_width = (*texture_iter)->getWidth();
    }
    return texture_width;
}


unsigned int RenderUtils::GetTextureHeight(GLuint texture_id) const {

    unsigned int texture_height = 0;
    auto texture_iter = std::find_if(this->textures.begin(), this->textures.end(),
        [&texture_id](std::shared_ptr<glowl::Texture2D> tex_ptr) { return (tex_ptr->getName() == texture_id); });
    if (texture_iter != this->textures.end()) {
        texture_height = (*texture_iter)->getHeight();
    }
    return texture_height;
}


void RenderUtils::drawPrimitives(RenderUtils::Primitives primitive, const glm::mat4& mat_mvp, glm::vec2 dim_vp) {

    if (!this->init_once) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Primitive rendering must be initialized before drawing. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return;
    }
    GLsizei count = static_cast<GLsizei>(this->queues[primitive].position.size() / 3);
    if (count == 0)
        return;

    this->sortPrimitiveQueue(primitive);

    auto texture_id = this->queues[primitive].texture_id;
    this->buffers[Buffers::POSITION]->rebuffer(this->queues[primitive].position);
    this->buffers[Buffers::COLOR]->rebuffer(this->queues[primitive].color);
    this->buffers[Buffers::TEXTURE_COORD]->rebuffer(this->queues[primitive].texture_coord);
    this->buffers[Buffers::ATTRIBUTES]->rebuffer(this->queues[primitive].attributes);

    // Set OpenGL state ----------------------------------------------------
    // Blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_CULL_FACE);
    // Smoothing
    if (this->smooth) {
        glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_NICEST);
    }

    // Draw ----------------------------------------------------------------
    GLenum mode = GL_TRIANGLES;
    if (primitive == Primitives::POINTS) {
        mode = GL_POINTS;
    }

    glBindVertexArray(this->vertex_array);
    this->shaders[primitive]->use();

    if (texture_id != 0) {
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glUniform1i(this->shaders[primitive]->getUniformLocation("tex"), static_cast<GLint>(0));
    }

    glUniform1i(this->shaders[primitive]->getUniformLocation("apply_smooth"), static_cast<GLint>(this->smooth));
    glUniform2fv(this->shaders[primitive]->getUniformLocation("viewport"), 1, glm::value_ptr(dim_vp));
    glUniformMatrix4fv(this->shaders[primitive]->getUniformLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mat_mvp));

    glDrawArrays(mode, 0, count);

    if (texture_id != 0) {
        glBindTexture(GL_TEXTURE_1D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    glUseProgram(0);
    glBindVertexArray(0);

    // Reset OpenGL state --------------------------------------------------
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}


void RenderUtils::sortPrimitiveQueue(Primitives primitive) {

    // Sort primitives by distance (stored in attribute.w) for correct blending.

    const size_t dim_pos = 3;
    const size_t dim_col = 4;
    const size_t dim_txc = 2;
    const size_t dim_atr = 4;

    float d;

    switch (primitive) {
    case (Primitives::POINTS): {

        auto size_pos = this->queues[primitive].position.size() / dim_pos;
        auto size_col = this->queues[primitive].color.size() / dim_col;
        auto size_txc = this->queues[primitive].texture_coord.size() / dim_txc;
        auto size_atr = this->queues[primitive].attributes.size() / dim_atr;

        if (!((size_pos == size_col) && (size_col == size_txc) && (size_txc == size_atr))) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Primitive sorting fails due to inconsitent data list count - BUG. [%s, %s, line %d]\n", __FILE__,
                __FUNCTION__, __LINE__);
            return;
        }

        // for (size_t i = 0; i < size_atr; ++i) {
        //    d = this->queues[primitive].attributes[i*dim_atr + 3];


        //}


    } break;
    default:
        break;
    }
}


void RenderUtils::pushQuad(Primitives primitive, GLuint texture_id, const glm::vec3& pos_bottom_left,
    const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right,
    const glm::vec4& color, const glm::vec4& attributes) {

    // First triangle
    this->pushShaderData(primitive, texture_id, pos_bottom_left, color, glm::vec2(0.0f, 0.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_upper_right, color, glm::vec2(1.0f, 1.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_upper_left, color, glm::vec2(0.0f, 1.0f), attributes);

    // Second triangle
    this->pushShaderData(primitive, texture_id, pos_bottom_left, color, glm::vec2(0.0f, 0.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_bottom_right, color, glm::vec2(1.0f, 0.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_upper_right, color, glm::vec2(1.0f, 1.0f), attributes);
}


void RenderUtils::pushShaderData(Primitives primitive, GLuint texture_id, const glm::vec3& position,
    const glm::vec4& color, const glm::vec2& texture_coord, const glm::vec4& attributes) {

    this->queues[primitive].texture_id = texture_id;
    this->pushQueue(this->queues[primitive].position, position);
    this->pushQueue(this->queues[primitive].color, color);
    this->pushQueue(this->queues[primitive].texture_coord, texture_coord);
    this->pushQueue(this->queues[primitive].attributes, attributes);
}


void RenderUtils::clearQueue(Primitives primitive) {

    this->queues[primitive].texture_id = 0;
    this->queues[primitive].position.clear();
    this->queues[primitive].color.clear();
    this->queues[primitive].texture_coord.clear();
    this->queues[primitive].attributes.clear();
}


void RenderUtils::pushQueue(std::vector<float>& d, float v, UINT cnt) {

    d.emplace_back(v);
}


void RenderUtils::pushQueue(std::vector<float>& d, glm::vec2 v, UINT cnt) {

    for (unsigned int i = 0; i < cnt; ++i) {
        d.emplace_back(v.x);
        d.emplace_back(v.y);
    }
}


void RenderUtils::pushQueue(std::vector<float>& d, glm::vec3 v, UINT cnt) {

    for (unsigned int i = 0; i < cnt; ++i) {
        d.emplace_back(v.x);
        d.emplace_back(v.y);
        d.emplace_back(v.z);
    }
}


void RenderUtils::pushQueue(std::vector<float>& d, glm::vec4 v, UINT cnt) {

    for (unsigned int i = 0; i < cnt; ++i) {
        d.emplace_back(v.x);
        d.emplace_back(v.y);
        d.emplace_back(v.z);
        d.emplace_back(v.w);
    }
}


glm::vec3 RenderUtils::arbitraryPerpendicular(glm::vec3 in) {

    if ((in.x == 0.0f) && (in.y == 0.0f) && (in.z == 0.0f)) {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    } else if (in.x == 0.0f) {
        return glm::vec3(1.0f, 0.0f, 0.0f);
    } else if (in.y == 0.0f) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    } else if (in.z == 0.0f) {
        return glm::vec3(0.0f, 0.0f, 1.0f);
    } else {
        return glm::vec3(1.0f, 1.0f, -1.0f * (in.x + in.y) / in.z);
    }
}

} // namespace megamol::core_gl::utility
