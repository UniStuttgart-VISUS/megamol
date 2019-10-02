/*
* RenderUtils.cpp
*
* Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CinematicUtils.h"


using namespace megamol::cinematic;


// ##### RenderUtils ######################################################## //

RenderUtils::RenderUtils()
    : smooth(true)
    , init_once(false)
    , vertex_array(0)
    , textures()
    , queues()
    , shaders()
    , buffers() {

}


RenderUtils::~RenderUtils() {

}


bool RenderUtils::InitPrimitiveRendering(megamol::core::utility::ShaderSourceFactory& factory) {

    if (this->init_once) {
        vislib::sys::Log::DefaultLog.WriteWarn("Primitive rendering has already been initialized. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    // Create shaders
    if (!this->createShader(this->shaders[Primitives::POINTS], &this->getShaderCode(factory, "primitives::points::vertex"), &this->getShaderCode(factory, "primitives::points::fragment"))) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to create point shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->createShader(this->shaders[Primitives::LINES], &this->getShaderCode(factory, "primitives::lines::vertex"), &this->getShaderCode(factory, "primitives::lines::fragment"))) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to create line shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->createShader(this->shaders[Primitives::QUADS], &this->getShaderCode(factory, "primitives::quads::vertex"), &this->getShaderCode(factory, "primitives::quads::fragment"))) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to create quad shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->createShader(this->shaders[Primitives::COLOR_TEXTURE], &this->getShaderCode(factory, "primitives::color_texture::vertex"), &this->getShaderCode(factory, "primitives::color_texture::fragment"))) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to create color texture shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->createShader(this->shaders[Primitives::DEPTH_TEXTURE], &this->getShaderCode(factory, "primitives::depth_texture::vertex"), &this->getShaderCode(factory, "primitives::depth_texture::fragment"))) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to create depth texture shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Create buffers
    this->buffers[Buffers::POSITION] = std::make_unique<glowl::BufferObject >(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->buffers[Buffers::COLOR] = std::make_unique<glowl::BufferObject >(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->buffers[Buffers::TEXTURE_COORD] = std::make_unique<glowl::BufferObject >(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    this->buffers[Buffers::ATTRIBUTES] = std::make_unique<glowl::BufferObject >(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

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


bool RenderUtils::LoadTextureFromFile(std::wstring filename, GLuint& out_texture_id) {

    out_texture_id = 0;

    this->textures.emplace_back(std::make_unique<TextureType>());
    auto texture = this->textures.back();

    vislib::graphics::BitmapImage img;
    sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    BYTE *buf = nullptr;
    size_t size = 0;

    //megamol::core::utility::ResourceWrapper::LoadResource(this->GetCoreInstance()->Configuration(), filename, (void**)(&buf)))

    if ((size = this->loadRawFile(filename, &buf)) <= 0) {
        vislib::sys::Log::DefaultLog.WriteError("Could not find texture \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(buf);
        return false;
    }

    if (pbc.Load((void*)buf, size)) {

        img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);

        if (texture->Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
            vislib::sys::Log::DefaultLog.WriteError("Could not load texture \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
            ARY_SAFE_DELETE(buf);
            return false;
        }

        // Additional texture options
        texture->Bind();
        texture->SetFilter(GL_LINEAR, GL_LINEAR);
        texture->SetWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        ARY_SAFE_DELETE(buf);
        out_texture_id = texture->GetId();
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("Could not read texture \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(buf);
        return false;
    }
    
    return true;
}


void RenderUtils::PushPointPrimitive(const glm::vec3& pos_center, float size, const glm::vec3& cam_pos, const glm::vec4& color) {

    glm::vec3 distance = (cam_pos - pos_center);
    //float d = glm::length(distance);
    float radius = size / 2.0f;
    glm::vec3 rad = pos_center + glm::normalize(this->arbitraryPerpendicular(distance)) * radius;
    glm::vec4 attributes = { rad.x, rad.y, rad.z, 0.0f };
    this->pushShaderData(Primitives::POINTS, 0, pos_center, color, glm::vec2(0.0f, 0.0f), attributes);
}


void RenderUtils::PushLinePrimitive(const glm::vec3& pos_start, const glm::vec3& pos_end, float line_width, const glm::vec3& normal, const glm::vec4& color) {

    glm::vec3 linedir = (pos_start - pos_end);
    glm::vec3 w = glm::normalize(glm::cross(normal, linedir));
    glm::vec3 p1 = w * (line_width / 2.0f);
    glm::vec3 pos_bottom_left = pos_start - p1;
    glm::vec3 pos_upper_left = pos_start + p1;
    glm::vec3 pos_upper_right = pos_end + p1;
    glm::vec3 pos_bottom_right = pos_end - p1;
    glm::vec4 attributes = { 0.0f, 0.0f, 0.0f, 0.0f };
    this->pushQuad(RenderUtils::Primitives::QUADS, 0, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, color, attributes);
}


void RenderUtils::PushQuadPrimitive(const glm::vec3& pos_center, float width, float height, const glm::vec3& normal, const glm::vec3& up, const glm::vec4& color) {

    glm::vec3 p1 = glm::normalize(up);
    glm::vec3 p2 = glm::cross(normal, p1);
    p1 = glm::normalize(p1) * (height / 2.0f);
    p2 = glm::normalize(p2) * (width / 2.0f);
    glm::vec3 pos_bottom_left = pos_center - p1 - p2;
    glm::vec3 pos_upper_left = pos_center + p1 - p2;
    glm::vec3 pos_upper_right = pos_center + p1 + p2;
    glm::vec3 pos_bottom_right = pos_center - p1 + p2;
    glm::vec4 attributes = { 0.0f, 0.0f, 0.0f, 0.0f };
    this->pushQuad(RenderUtils::Primitives::QUADS, 0, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, color, attributes);
}

void RenderUtils::PushQuadPrimitive(const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, const glm::vec4& color) {

    glm::vec4 attributes = { 0.0f, 0.0f, 0.0f, 0.0f };
    this->pushQuad(RenderUtils::Primitives::QUADS, 0, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, color, attributes);
}


void RenderUtils::Push2DColorTexture(GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y, const glm::vec4& color) {

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
    glm::vec4 attributes = { 0.0f, 0.0f, 0.0f, 0.0f };
    this->pushQuad(RenderUtils::Primitives::COLOR_TEXTURE, texture_id, pbl, pul, pur, pbr, color, attributes);
}


void RenderUtils::Push2DDepthTexture(GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y, const glm::vec4& color) {

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
    glm::vec4 attributes = { 0.0f, 0.0f, 0.0f, 0.0f };
    this->pushQuad(RenderUtils::Primitives::DEPTH_TEXTURE, texture_id, pbl, pul, pur, pbr, color, attributes);
}


void RenderUtils::drawPrimitives(RenderUtils::Primitives primitive, glm::mat4& mat_mvp, glm::vec2 dim_vp) {

    if (!this->init_once) {
        vislib::sys::Log::DefaultLog.WriteError("Primitive rendering must be initialized before drawing. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }
    GLsizei count = static_cast<GLsizei>(this->queues[primitive].position.size() / 3);
    if (count == 0) return;

    auto texture_id = this->queues[primitive].texture_id;
    this->buffers[Buffers::POSITION]->rebuffer<std::vector<float>>(this->queues[primitive].position);
    this->buffers[Buffers::COLOR]->rebuffer<std::vector<float>>(this->queues[primitive].color);
    this->buffers[Buffers::TEXTURE_COORD]->rebuffer<std::vector<float>>(this->queues[primitive].texture_coord);
    this->buffers[Buffers::ATTRIBUTES]->rebuffer<std::vector<float>>(this->queues[primitive].attributes);

    // Set OpenGL state ----------------------------------------------------
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);
    if (!blendEnabled) {
        glEnable(GL_BLEND);
    }
    GLint blendSrc;
    GLint blendDst;
    glGetIntegerv(GL_BLEND_SRC, &blendSrc);
    glGetIntegerv(GL_BLEND_DST, &blendDst);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    if (!depthEnabled) {
        glEnable(GL_DEPTH_TEST);
    }
    GLboolean cullEnabled = glIsEnabled(GL_CULL_FACE);
    if (cullEnabled) {
        glDisable(GL_CULL_FACE);
    }
    if (this->smooth) {
        glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_NICEST);
    }
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // Draw ----------------------------------------------------------------
    GLenum mode = GL_TRIANGLES;
    if (primitive == Primitives::POINTS) {
        mode = GL_POINTS;
    }

    glBindVertexArray(this->vertex_array);
    this->shaders[primitive].Enable();

    if (texture_id != 0) {
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glUniform1i(this->shaders[primitive].ParameterLocation("tex"), static_cast<GLint>(0));
    }

    glUniform1i(this->shaders[primitive].ParameterLocation("apply_smooth"), static_cast<GLint>(this->smooth));
    glUniform2fv(this->shaders[primitive].ParameterLocation("viewport"), 1, glm::value_ptr(dim_vp));
    glUniformMatrix4fv(this->shaders[primitive].ParameterLocation("mvp"), 1, GL_FALSE, glm::value_ptr(mat_mvp));

    glDrawArrays(mode, 0, count);

    if (texture_id != 0) {
        glBindTexture(GL_TEXTURE_1D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    this->shaders[primitive].Disable();
    glBindVertexArray(0);

    // Reset OpenGL state --------------------------------------------------
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    if (cullEnabled) {
        glEnable(GL_CULL_FACE);
    }
    if (!depthEnabled) {
        glDisable(GL_DEPTH_TEST);
    }
    glBlendFunc(blendSrc, blendDst);
    if (!blendEnabled) {
        glDisable(GL_BLEND);
    }
}


bool RenderUtils::createShader(vislib::graphics::gl::GLSLShader& shader, const std::string * const vertex_code, const std::string * const fragment_code) {

    shader.Release();
    if (!shader.Compile(vertex_code->c_str(), fragment_code->c_str())) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    shader.BindAttribute(Buffers::POSITION, "inPosition");
    shader.BindAttribute(Buffers::COLOR, "inColor");
    shader.BindAttribute(Buffers::TEXTURE_COORD, "inTexture");
    shader.BindAttribute(Buffers::ATTRIBUTES, "inAttributes");
    if (!shader.Link()) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to link shader. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


const std::string RenderUtils::getShaderCode(megamol::core::utility::ShaderSourceFactory& factory, std::string snippet_name) {

    vislib::graphics::gl::ShaderSource source;
    if (!factory.MakeShaderSource(snippet_name.c_str(), source)) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to make vertex shader source. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return std::string("");
    }
    return std::string(source.WholeCode().PeekBuffer());
}


size_t RenderUtils::loadRawFile(std::wstring filename, BYTE **outData) {

    // Reset out data
    *outData = nullptr;

    vislib::StringW name = static_cast<vislib::StringW>(filename.c_str());
    if (name.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteError(" Unable to load texture file. No name given. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }
    if (!vislib::sys::File::Exists(name)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load not existing file \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    size_t size = static_cast<size_t>(vislib::sys::File::GetSize(name));
    if (size < 1) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to load empty file \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(name, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to open file \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    *outData = new BYTE[size];
    size_t num = static_cast<size_t>(f.Read(*outData, size));
    if (num != size) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to read whole file \"%s\". [%s, %s, line %d)]\n", filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}


void RenderUtils::pushQuad(Primitives primitive, GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right,
    const glm::vec3& pos_bottom_right, const glm::vec4& color, const glm::vec4& attributes) {

    // First triangle
    this->pushShaderData(primitive, texture_id, pos_bottom_left, color, glm::vec2(0.0f, 0.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_upper_right, color, glm::vec2(1.0f, 1.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_upper_left, color, glm::vec2(0.0f, 1.0f), attributes);

    // Second triangle
    this->pushShaderData(primitive, texture_id, pos_bottom_left, color, glm::vec2(0.0f, 0.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_bottom_right, color, glm::vec2(1.0f, 0.0f), attributes);
    this->pushShaderData(primitive, texture_id, pos_upper_right, color, glm::vec2(1.0f, 1.0f), attributes);
}


void RenderUtils::pushShaderData(Primitives primitive, GLuint texture_id, const glm::vec3& position, const glm::vec4& color, const glm::vec2& texture_coord, const glm::vec4& attributes) {

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


void RenderUtils::pushQueue(DataType& d, float v, UINT cnt) {

    d.emplace_back(v);
}


void RenderUtils::pushQueue(DataType& d, glm::vec2 v, UINT cnt) {

    for (unsigned int i = 0; i < cnt; ++i) {
        d.emplace_back(v.x);
        d.emplace_back(v.y);
    }
}


void RenderUtils::pushQueue(DataType& d, glm::vec3 v, UINT cnt) {

    for (unsigned int i = 0; i < cnt; ++i) {
        d.emplace_back(v.x);
        d.emplace_back(v.y);
        d.emplace_back(v.z);
    }
}


void RenderUtils::pushQueue(DataType& d, glm::vec4 v, UINT cnt) {

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
    }
    else if (in.x == 0.0f) {
        return glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else if (in.y == 0.0f) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (in.z == 0.0f) {
        return glm::vec3(0.0f, 0.0f, 1.0f);
    }
    else {
        return glm::vec3(1.0f, 1.0f, -1.0f * (in.x + in.y) / in.z);
    }
}


// ##### CinematicUtils ######################################################## //


CinematicUtils::CinematicUtils(void) : megamol::cinematic::RenderUtils()
    , font(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS)
    , font_size(20.0f)
    , init_once(false)
    , background_color(0.0f, 0.0f, 0.0f, 1.0f) {

}


CinematicUtils::~CinematicUtils(void) {

}


bool CinematicUtils::Initialise(megamol::core::CoreInstance* core_instance) {

    if (this->init_once) {
        vislib::sys::Log::DefaultLog.WriteWarn("Primitive rendering has already been initialized. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    // Initialise font
    if (!this->font.Initialise(core_instance)) {
        vislib::sys::Log::DefaultLog.WriteError("Couldn't initialize the font. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->font.SetBatchDrawMode(true);

    // Initialise rendering
    if (!this->InitPrimitiveRendering(core_instance->ShaderSourceFactory())) {
        vislib::sys::Log::DefaultLog.WriteError("Couldn't initialize primitive rendering. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    this->init_once = true;

    return true;
}


const glm::vec4 CinematicUtils::Color(CinematicUtils::Colors c) const {

    glm::vec4 color = { 0.0f, 0.0f, 0.0f, 0.0f };

    switch (c) {
    case (CinematicUtils::Colors::BACKGROUND):
        color = this->background_color;
        break;
    case (CinematicUtils::Colors::FOREGROUND): {
        glm::vec4 foreground = { 1.0f, 1.0f, 1.0f, 1.0f };
        color = this->background_color;
        for (unsigned int i = 0; i < 3; i++) {
            foreground[i] -= color[i];
        }
        color = foreground;
    } break;
    case (CinematicUtils::Colors::KEYFRAME):
        color = { 0.7f, 0.7f, 1.0f, 1.0f };
        break;
    case (CinematicUtils::Colors::KEYFRAME_DRAGGED):
        color = { 0.5f, 0.5f, 1.0f, 1.0f };
        break;
    case (CinematicUtils::Colors::KEYFRAME_SELECTED):
        color = { 0.2f, 0.2f, 1.0f, 1.0f };
        break;
    case (CinematicUtils::Colors::KEYFRAME_SPLINE):
        color = { 0.4f, 0.4f, 1.0f, 1.0f };
        break;
    case (CinematicUtils::Colors::MENU):
        color = { 0.0f, 0.0f, 0.5f, 1.0f };
        break;
    case (CinematicUtils::Colors::FONT):
        color = { 1.0f, 1.0f, 1.0f, 1.0f };
        if (CinematicUtils::lightness(this->background_color) > 0.5f) {
            color = { 0.0f, 0.0f, 0.0f, 1.0f };
        }
        break;
    case (CinematicUtils::Colors::FONT_HIGHLIGHT):
        color = { 0.75f, 0.75f, 0.0f, 1.0f };
        break;
    case (CinematicUtils::Colors::LETTER_BOX):
        color = { 1.0f, 1.0f, 1.0f, 1.0f };
        if (CinematicUtils::lightness(this->background_color) > 0.5f) {
            color = { 0.0f, 0.0f, 0.0f, 1.0f };
        }
        break;
    case (CinematicUtils::Colors::FRAME_MARKER):
        color = { 1.0f, 0.6f, 0.6f, 1.0f };
        break;
    default: break;
    }

    return color;
}


void CinematicUtils::PushMenu(const std::string& left_label, const std::string& middle_label, const std::string& right_label, float viewport_width, float viewport_height) {

    const float menu_height = this->font_size;

    // Push menu background quad
    this->PushQuadPrimitive(glm::vec3(0.0f, viewport_height, 0.0f), glm::vec3(0.0f, viewport_height - menu_height, 0.0f), 
        glm::vec3(viewport_width, viewport_height - menu_height, 0.0f), glm::vec3(viewport_width, viewport_height, 0.0f), this->Color(CinematicUtils::Colors::MENU));

    // Push menu labels
    float textPosZ = -0.1f;
    float vpWhalf = viewport_width / 2.0f;
    float new_font_size = this->font_size;
    float leftLabelWidth = this->font.LineWidth(this->font_size, left_label.c_str());
    float midleftLabelWidth = this->font.LineWidth(this->font_size, middle_label.c_str());
    float rightLabelWidth = this->font.LineWidth(this->font_size, right_label.c_str());
    while (((leftLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf) || ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
        new_font_size -= 0.5f;
        leftLabelWidth = this->font.LineWidth(new_font_size, left_label.c_str());
        midleftLabelWidth = this->font.LineWidth(new_font_size, middle_label.c_str());
        rightLabelWidth = this->font.LineWidth(new_font_size, right_label.c_str());
    }
    float textPosY = viewport_height - (menu_height / 2.0f) + (new_font_size / 2.0f);
    auto current_back_color = this->Color(CinematicUtils::Colors::BACKGROUND);
    this->SetBackgroundColor(this->Color(CinematicUtils::Colors::MENU));
    auto color = this->Color(CinematicUtils::Colors::FONT);
    this->font.DrawString(glm::value_ptr(color), 0.0f, textPosY, textPosZ, new_font_size, false, left_label.c_str(), megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->font.DrawString(glm::value_ptr(color), (viewport_width - midleftLabelWidth) / 2.0f, textPosY, textPosZ, new_font_size, false, middle_label.c_str(), megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->font.DrawString(glm::value_ptr(color), (viewport_width - rightLabelWidth), textPosY, textPosZ, new_font_size, false, right_label.c_str(), megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->SetBackgroundColor(current_back_color);
}


void CinematicUtils::PushHotkeyList(const std::string& text, glm::vec3 position, float width, float height) {


    // Draw help text 
//if (this->showHelpText) {
//    vislib::StringA helpText = "";
//    helpText += "-----[ GLOBAL ]-----\n";
//    helpText += "[Ctrl+a] Apply current settings to selected/new keyframe. \n";
//    helpText += "[Ctrl+d] Delete selected keyframe. \n";
//    helpText += "[Ctrl+s] Save keyframes to file. \n";
//    helpText += "[Ctrl+l] Load keyframes from file. \n";
//    helpText += "[Ctrl+z] Undo keyframe changes. \n";
//    helpText += "[Ctrl+y] Redo keyframe changes. \n";
//    helpText += "-----[ TRACKING SHOT ]----- \n";
//    helpText += "[Ctrl+q] Toggle different manipulators for the selected keyframe. \n";
//    helpText += "[Ctrl+w] Show manipulators inside/outside of model bounding box. \n";
//    helpText += "[Ctrl+u] Reset look-at vector of selected keyframe. \n";
//    helpText += "-----[ CINEMATIC ]----- \n";
//    helpText += "[Ctrl+r] Start/Stop rendering complete animation. \n";
//    helpText += "[Ctrl+Space] Start/Stop animation preview. \n";
//    helpText += "-----[ TIMELINE ]----- \n";
//    helpText += "[Ctrl+Right/Left Arrow] Move selected keyframe on animation time axis. \n";
//    helpText += "[Ctrl+f] Snap all keyframes to animation frames. \n";
//    helpText += "[Ctrl+g] Snap all keyframes to simulation frames. \n";
//    helpText += "[Ctrl+t] Linearize simulation time between two keyframes. \n";
//    //helpText += "[Ctrl+v] Set same velocity between all keyframes (Experimental).\n"; // Calcualation is not correct yet ...
//    helpText += "[Ctrl+p] Reset shifted and scaled time axes. \n";
//    helpText += "[Left Mouse Button] Select keyframe. \n";
//    helpText += "[Middle Mouse Button] Axes scaling in mouse direction. \n";
//    helpText += "[Right Mouse Button] Drag & drop keyframe / pan axes. \n";

//    float htNumOfRows = 24.0f; // Number of rows the help text has

//    float htFontSize  = vpW*0.027f; // max % of viewport width
//    float htStrHeight = this->theFont.LineHeight(htFontSize);
//    float htX         = 5.0f;
//    float htY         = htX + htStrHeight;
//    // Adapt font size if height of help text is greater than viewport height
//    while ((htStrHeight*htNumOfRows + htX + this->theFont.LineHeight(lbFontSize)) >vpH) {
//        htFontSize -= 0.5f;
//        htStrHeight = this->theFont.LineHeight(htFontSize);
//    }

//    float htStrWidth = this->theFont.LineWidth(htFontSize, helpText);
//    htStrHeight      = this->theFont.LineHeight(htFontSize);
//    htY              = htX + htStrHeight*htNumOfRows;
//    // Draw background colored quad
//    glColor4fv(bgColor);
//    glBegin(GL_QUADS);
//        glVertex2f(htX,              htY);
//        glVertex2f(htX,              htY - (htStrHeight*htNumOfRows));
//        glVertex2f(htX + htStrWidth, htY - (htStrHeight*htNumOfRows));
//        glVertex2f(htX + htStrWidth, htY);
//    glEnd();
//    // Draw help text
//    this->theFont.DrawString(fgColor, htX, htY, htFontSize, false, helpText, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
//}



}


void CinematicUtils::PushText(const std::string& text, float x, float y, float z) {

    float text_width = this->font.LineWidth(this->font_size, text.c_str());
    auto color = this->Color(CinematicUtils::Colors::FONT);
    this->font.DrawString(glm::value_ptr(color), x, y, z, this->font_size, false, text.c_str(), megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
}


void CinematicUtils::DrawAll(glm::mat4& mat_mvp, glm::vec2 dim_vp) {

    if (!this->init_once) {
        vislib::sys::Log::DefaultLog.WriteError("Cinematic utilities must be initialized before drawing. [%s, %s, line %d)]\n", __FILE__, __FUNCTION__, __LINE__);
        return;
    }

    this->DrawAllPrimitives(mat_mvp, dim_vp);

    // Font rendering takes matrices from OpenGL stack
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glLoadMatrixf(glm::value_ptr(mat_mvp));
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    this->font.BatchDrawString();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    this->font.ClearBatchDrawCache();
}


float CinematicUtils::GetTextLineHeight(void) {

    return this->font.LineHeight(this->font_size);
}


float CinematicUtils::GetTextLineWidth(const std::string& text_line) {

    return this->font.LineWidth(this->font_size, text_line.c_str());
}


void CinematicUtils::SetTextRotation(float a, float x, float y, float z) {

    this->font.SetRotation(a, x, y, z);
}


const float CinematicUtils::lightness(glm::vec4 background) const {

    return ((glm::max(background[0], glm::max(background[1], background[2])) + glm::min(background[0], glm::min(background[1], background[2]))) / 2.0f);
}
