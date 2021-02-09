/*
 * RenderUtils.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <array>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLOWL_OPENGL_INCLUDE_GLAD
#include <glowl/BufferObject.hpp>

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/view/Camera_2.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/graphics/gl/ShaderSource.h"

namespace megamol::core::view {

// #### Utility vector conversion functions ############################ //

static inline vislib::math::Vector<float, 3> glm_to_vislib_vector(glm::vec3 v) {
    return vislib::math::Vector<float, 3>(v.x, v.y, v.z);
}

static inline glm::vec3 vislib_vector_to_glm(vislib::math::Vector<float, 3> v) {
    return glm::vec3(v.X(), v.Y(), v.Z());
}

static inline vislib::math::Point<float, 3> glm_to_vislib_point(glm::vec3 v) {
    return vislib::math::Point<float, 3>(v.x, v.y, v.z);
}

static inline glm::vec3 vislib_point_to_glm(vislib::math::Point<float, 3> v) {
    return glm::vec3(v.X(), v.Y(), v.Z());
}


// #### Utility quaternion functions ################################### //

static inline glm::quat quaternion_from_vectors(glm::vec3 view_vector, glm::vec3 up_vector) {

    glm::vec3 view = view_vector * glm::vec3(-1.0f, -1.0f, -1.0f); /// why?
    glm::vec3 right = glm::cross(up_vector, view);
    glm::vec3 up = glm::cross(view, right);

    glm::vec3 norm_right = glm::normalize(right);
    glm::vec3 norm_up = glm::normalize(up);
    glm::vec3 norm_view = glm::normalize(view);

    glm::mat3 matrix_basis;
    matrix_basis[0] = norm_right;
    matrix_basis[1] = norm_up;
    matrix_basis[2] = norm_view;

    glm::quat orientation_quat = glm::quat_cast(matrix_basis);

    return orientation_quat; // glm::normalize(orientation_quat);
}


// ##################################################################### //
/*
 * Utility class providing simple primitive rendering (using non legacy opengl).
 */
class RenderUtils {

public:
    bool InitPrimitiveRendering(megamol::core::utility::ShaderSourceFactory& factory);

    bool LoadTextureFromFile(std::wstring filename, GLuint& out_texture_id);

    void PushPointPrimitive(const glm::vec3& pos_center, float size, const glm::vec3& cam_view,
        const glm::vec3& cam_pos, const glm::vec4& color, bool sort = false);

    void PushLinePrimitive(const glm::vec3& pos_start, const glm::vec3& pos_end, float line_width,
        const glm::vec3& cam_view, const glm::vec3& cam_pos, const glm::vec4& color, bool sort = false);

    void PushQuadPrimitive(const glm::vec3& pos_center, float width, float height, const glm::vec3& cam_view,
        const glm::vec3& cam_up, const glm::vec4& color, bool sort = false);

    void PushQuadPrimitive(const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left,
        const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, const glm::vec4& color, bool sort = false);

    /// Default color requires alpha = zero to recognise in shader whether global color for texture is set or not.
    void Push2DColorTexture(GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left,
        const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y = false,
        const glm::vec4& color = glm::vec4(0.0f));

    void Push2DDepthTexture(GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left,
        const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y = false,
        const glm::vec4& color = glm::vec4(0.0f));

    inline void DrawPointPrimitives(glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::POINTS, mat_mvp, dim_vp);
        this->clearQueue(Primitives::POINTS);
    }

    inline void DrawLinePrimitives(glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::LINES, mat_mvp, dim_vp);
        this->clearQueue(Primitives::LINES);
    }

    inline void DrawQuadPrimitives(glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::QUADS, mat_mvp, dim_vp);
        this->clearQueue(Primitives::QUADS);
    }

    inline void DrawTextures(glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::DEPTH_TEXTURE, mat_mvp, dim_vp);
        this->drawPrimitives(RenderUtils::Primitives::COLOR_TEXTURE, mat_mvp, dim_vp);
        this->clearQueue(Primitives::COLOR_TEXTURE);
        this->clearQueue(Primitives::DEPTH_TEXTURE);
    }

    inline void DrawAllPrimitives(glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->DrawPointPrimitives(mat_mvp, dim_vp);
        this->DrawLinePrimitives(mat_mvp, dim_vp);
        this->DrawQuadPrimitives(mat_mvp, dim_vp);
        this->DrawTextures(mat_mvp, dim_vp);
    }

    inline void Smoothing(bool s) {
        this->smooth = s;
    }

protected:
    RenderUtils();

    ~RenderUtils();

private:
    typedef std::vector<float> DataType;

    enum Primitives : size_t { LINES = 0, POINTS = 1, QUADS = 2, COLOR_TEXTURE = 3, DEPTH_TEXTURE = 4, PRIM_COUNT = 5 };

    enum Buffers : GLuint { POSITION = 0, COLOR = 1, TEXTURE_COORD = 2, ATTRIBUTES = 3, BUFF_COUNT = 4 };

    typedef struct _shader_data_ {
        GLuint texture_id;
        DataType position;
        DataType color;
        DataType texture_coord;
        DataType attributes;
    } ShaderDataType;

    typedef vislib::graphics::gl::OpenGLTexture2D TextureType;
    typedef std::vector<std::shared_ptr<TextureType>> TexturesType;
    typedef std::array<ShaderDataType, Primitives::PRIM_COUNT> QueuesType;
    typedef std::array<vislib::graphics::gl::GLSLShader, Primitives::PRIM_COUNT> ShadersType;
    typedef std::array<std::unique_ptr<glowl::BufferObject>, Buffers::BUFF_COUNT> BuffersType;

    // VARIABLES ------------------------------------------------------- //

    bool smooth;
    bool init_once;
    TexturesType textures;
    GLuint vertex_array;
    QueuesType queues;
    ShadersType shaders;
    BuffersType buffers;

    // FUNCTIONS ------------------------------------------------------- //

    void pushQuad(Primitives primitive, GLuint texture_id, const glm::vec3& pos_bottom_left,
        const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right,
        const glm::vec4& color, const glm::vec4& attributes);

    void drawPrimitives(Primitives primitive, glm::mat4& mat_mvp, glm::vec2 dim_vp);

    void sortPrimitiveQueue(Primitives primitive);

    bool createShader(vislib::graphics::gl::GLSLShader& shader, const std::string* const vertex_code,
        const std::string* const fragment_code);

    const std::string getShaderCode(megamol::core::utility::ShaderSourceFactory& factory, std::string snippet_name);

    size_t loadRawFile(std::wstring filename, BYTE** outData);

    void clearQueue(Primitives primitive);

    void pushShaderData(Primitives primitive, GLuint texture_id, const glm::vec3& position, const glm::vec4& color,
        const glm::vec2& texture_coord, const glm::vec4& attributes);

    void pushQueue(DataType& d, float v, UINT cnt = 1);

    void pushQueue(DataType& d, glm::vec2 v, UINT cnt = 1);

    void pushQueue(DataType& d, glm::vec3 v, UINT cnt = 1);

    void pushQueue(DataType& d, glm::vec4 v, UINT cnt = 1);

    glm::vec3 arbitraryPerpendicular(glm::vec3 in);
};

} // end namespace
