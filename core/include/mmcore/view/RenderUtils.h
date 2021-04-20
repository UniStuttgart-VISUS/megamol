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
#include <glowl/Texture2D.hpp>
#include <glowl/GLSLProgram.hpp>

#include "glowl/FramebufferObject.hpp"

#include "mmcore/utility/FileUtils.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/view/Camera_2.h"


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


// #### Utility transformation functions ################################### //

static inline glm::vec3 worldspace_to_screenspace(
    const glm::vec3& vec_world, const glm::mat4& mvp, const glm::vec2& viewport) {

    glm::vec4 world = {vec_world.x, vec_world.y, vec_world.z, 1.0f};
    world = mvp * world;
    world = world / world.w;
    glm::vec3 screen;
    screen.x = (world.x + 1.0f) / 2.0f * viewport.x;
    screen.y = (world.y + 1.0f) / 2.0f * viewport.y; // flipped y-axis: glm::abs(world.y - 1.0f)
    screen.z = -1.0f * (world.z + 1.0f) / 2.0f;
    return screen;
}


static inline glm::vec3 screenspace_to_worldspace(
    const glm::vec3& vec_screen, const glm::mat4& mvp, const glm::vec2& viewport) {

    glm::vec3 screen;
    screen.x = (vec_screen.x * 2.0f / viewport.x) - 1.0f;
    screen.y = (vec_screen.y * 2.0f / viewport.y) - 1.0f;
    screen.z = ((vec_screen.z * 2.0f * -1.0f) - 1.0f);
    glm::vec4 world = {screen.x, screen.y, screen.z, 1.0f};
    glm::mat4 mvp_inverse = glm::inverse(mvp);
    world = mvp_inverse * world;
    world = world / world.w;
    glm::vec3 vec3d = glm::vec3(world.x, world.y, world.z);
    return vec3d;
}


// ##################################################################### //
/*
 * Utility class providing simple primitive rendering (using non legacy opengl).
 */
class RenderUtils {

public:

    // STATIC functions -------------------------------------------------------

    /**
     * Load textures.
     */
    static bool LoadTextureFromFile(std::shared_ptr<glowl::Texture2D>& out_texture_ptr, const std::wstring& filename) {
        return megamol::core::view::RenderUtils::LoadTextureFromFile(
            out_texture_ptr, megamol::core::utility::to_string(filename));
    }

    static bool LoadTextureFromFile(std::shared_ptr<glowl::Texture2D>& out_texture_ptr, const std::string& filename);

    static bool LoadTextureFromData(
        std::shared_ptr<glowl::Texture2D>& out_texture_ptr, int width, int height, float* data);

    /**
     * Create shader.
     */
    static bool CreateShader(std::shared_ptr<glowl::GLSLProgram>& out_shader_ptr, const std::string& vertex_code,
        const std::string& fragment_code);

    // LOCAL functions -------------------------------------------------------

    bool InitPrimitiveRendering(megamol::core::utility::ShaderSourceFactory& factory);

    // Keeps the texture object in render utils for later access via texture id
    bool LoadTextureFromFile(GLuint& out_texture_id, const std::wstring& filename, bool reload = false) {
        return this->LoadTextureFromFile(out_texture_id, megamol::core::utility::to_string(filename), reload);
    }
    bool LoadTextureFromFile(GLuint& out_texture_id, const std::string& filename, bool reload = false);

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
        const glm::vec4& color = glm::vec4(0.0f), bool force_opaque = false);

    void Push2DDepthTexture(GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left,
        const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y = false,
        const glm::vec4& color = glm::vec4(0.0f));

    inline void DrawPointPrimitives(const glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::POINTS, mat_mvp, dim_vp);
        this->clearQueue(Primitives::POINTS);
    }

    inline void DrawLinePrimitives(const glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::LINES, mat_mvp, dim_vp);
        this->clearQueue(Primitives::LINES);
    }

    inline void DrawQuadPrimitives(const glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::QUADS, mat_mvp, dim_vp);
        this->clearQueue(Primitives::QUADS);
    }

    inline void DrawTextures(const glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->drawPrimitives(RenderUtils::Primitives::DEPTH_TEXTURE, mat_mvp, dim_vp);
        this->drawPrimitives(RenderUtils::Primitives::COLOR_TEXTURE, mat_mvp, dim_vp);
        this->clearQueue(Primitives::COLOR_TEXTURE);
        this->clearQueue(Primitives::DEPTH_TEXTURE);
    }

    inline void DrawAllPrimitives(const glm::mat4& mat_mvp, glm::vec2 dim_vp) {
        this->DrawPointPrimitives(mat_mvp, dim_vp);
        this->DrawLinePrimitives(mat_mvp, dim_vp);
        this->DrawQuadPrimitives(mat_mvp, dim_vp);
        this->DrawTextures(mat_mvp, dim_vp);
    }

    inline void Smoothing(bool s) {
        this->smooth = s;
    }

    inline bool isInitialized() {
        return this->init_once;
    }

    unsigned int GetTextureWidth(GLuint texture_id) const;
    unsigned int GetTextureHeight(GLuint texture_id) const;

    inline void DeleteAllTextures(void) {
        this->textures.clear();
    }

    RenderUtils();

    ~RenderUtils();

private:

    enum Primitives : size_t { LINES = 0, POINTS = 1, QUADS = 2, COLOR_TEXTURE = 3, DEPTH_TEXTURE = 4, PRIM_COUNT = 5 };

    enum Buffers : GLuint { POSITION = 0, COLOR = 1, TEXTURE_COORD = 2, ATTRIBUTES = 3, BUFF_COUNT = 4 };

    typedef struct _shader_data_ {
        GLuint texture_id;
        std::vector<float> position;
        std::vector<float> color;
        std::vector<float> texture_coord;
        std::vector<float> attributes;
    } ShaderDataType;

    // VARIABLES ------------------------------------------------------- //

    bool smooth;
    bool init_once;
    GLuint vertex_array;
    std::vector<std::shared_ptr<glowl::Texture2D>> textures;
    std::array<ShaderDataType, Primitives::PRIM_COUNT> queues;
    std::array<std::shared_ptr<glowl::GLSLProgram>, Primitives::PRIM_COUNT> shaders;
    std::array<std::shared_ptr<glowl::BufferObject>, Buffers::BUFF_COUNT> buffers;

    // FUNCTIONS ------------------------------------------------------- //

    bool createShader(std::shared_ptr<glowl::GLSLProgram>& out_shader_ptr,
        megamol::core::utility::ShaderSourceFactory& shader_factory, const std::string& vertex_btf_snipprt,
        const std::string& fragment_btf_snippet);

    void pushQuad(Primitives primitive, GLuint texture_id, const glm::vec3& pos_bottom_left,
        const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right,
        const glm::vec4& color, const glm::vec4& attributes);

    void drawPrimitives(Primitives primitive, const glm::mat4& mat_mvp, glm::vec2 dim_vp);

    void sortPrimitiveQueue(Primitives primitive);

    size_t loadRawFile(std::wstring filename, BYTE** outData);

    void clearQueue(Primitives primitive);

    void pushShaderData(Primitives primitive, GLuint texture_id, const glm::vec3& position, const glm::vec4& color,
        const glm::vec2& texture_coord, const glm::vec4& attributes);

    void pushQueue(std::vector<float>& d, float v, UINT cnt = 1);

    void pushQueue(std::vector<float>& d, glm::vec2 v, UINT cnt = 1);

    void pushQueue(std::vector<float>& d, glm::vec3 v, UINT cnt = 1);

    void pushQueue(std::vector<float>& d, glm::vec4 v, UINT cnt = 1);

    glm::vec3 arbitraryPerpendicular(glm::vec3 in);
};

} // end namespace
