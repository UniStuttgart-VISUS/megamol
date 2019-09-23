/*
 * CinematicUtils.h
*
* Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_CINEMATIC_CINEMATICUTILS_H_INCLUDED
#define MEGAMOL_CINEMATIC_CINEMATICUTILS_H_INCLUDED


#include <array>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glowl/BufferObject.hpp>

#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/ShaderSourceFactory.h"
#include "mmcore/utility/SDFFont.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"

#include "vislib/sys/Log.h"


// GLOBAL CINEMATIC DEFINES
#ifndef CC_MENU_HEIGHT
    #define CC_MENU_HEIGHT (25.0f)
#endif


namespace megamol {
namespace cinematic {

    /*
    * Utility class providing simple primitive rendering for the cinematic plugin (using non legacy opengl).
    */
    class RenderUtils {

    public:

        bool InitPrimitiveRendering(megamol::core::utility::ShaderSourceFactory& factory);

        bool LoadTextureFromFile(std::wstring filename, GLuint& out_texture_id);


        void PushPointPrimitive(const glm::vec3& pos_center, float size, const glm::vec3& cam_pos, const glm::vec4& color);

        void PushLinePrimitive(const glm::vec3& pos_start, const glm::vec3& pos_end, float line_width, const glm::vec3& cam_pos, const glm::vec4& color);

        void PushQuadPrimitive(const glm::vec3& pos_center, float width, float height, const glm::vec3& cam_pos, const glm::vec3& cam_up, const glm::vec4& color);

        void PushQuadPrimitive(const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, const glm::vec4& color);

        void Push2DTexture(GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right, const glm::vec3& pos_bottom_right, bool flip_y = false, const glm::vec4& color = glm::vec4(1.0f, 1.0f, 1.0f, 0.0));

        inline void DrawPointPrimitives(glm::mat4& mat_mvp) {
            this->drawPrimitives(RenderUtils::Primitives::POINTS, mat_mvp);
        }

        inline void DrawLinePrimitives(glm::mat4& mat_mvp) {
            this->drawPrimitives(RenderUtils::Primitives::LINES, mat_mvp);
        }

        inline void DrawQuadPrimitives(glm::mat4& mat_mvp) {

            this->drawPrimitives(RenderUtils::Primitives::QUADS, mat_mvp);
        }

        inline void Draw2DTextures(glm::mat4& mat_mvp) {

            this->drawPrimitives(RenderUtils::Primitives::TEXTURE, mat_mvp);
        }

        inline void DrawAllPrimitives(glm::mat4& mat_mvp) {
            this->DrawPointPrimitives(mat_mvp);
            this->DrawLinePrimitives(mat_mvp);
            this->DrawQuadPrimitives(mat_mvp);
            this->Draw2DTextures(mat_mvp);
        }

        inline void ClearPointPrimitiveQueue(void) {
            this->clearQueue(Primitives::POINTS);
        }

        inline void ClearLinePrimitiveQueue(void) {
            this->clearQueue(Primitives::LINES);
        }

        inline void ClearQuadPrimitiveQueue(void) {
            this->clearQueue(Primitives::QUADS);
        }

        inline void Clear2DTextureQueue(void) {
            this->clearQueue(Primitives::TEXTURE);
        }

        inline void ClearAllQueues(void) {
            this->ClearPointPrimitiveQueue();
            this->ClearLinePrimitiveQueue();
            this->ClearQuadPrimitiveQueue();
            this->Clear2DTextureQueue();
        }

        inline void Smoothing(bool s) {
            this->smooth = s;
        }

    protected: 

        RenderUtils();

        ~RenderUtils();

    private:

        typedef std::vector<float> DataType;

        enum Primitives : size_t {
            LINES = 0,
            POINTS = 1,
            QUADS = 2,
            TEXTURE = 3,
            PRIM_COUNT = 4
        };

        enum Buffers : GLuint {
            POSITION = 0,
            COLOR = 1,
            TEXTURE_COORD = 2,
            ATTRIBUTES = 3,
            BUFF_COUNT = 4
        };

        typedef struct _shader_data_ {
            GLuint texture_id;
            DataType position;
            DataType color;
            DataType texture_coord;
            DataType attributes;
        } ShaderDataType;

        typedef vislib::graphics::gl::OpenGLTexture2D TextureType;
        typedef std::vector< std::shared_ptr<TextureType>> TexturesType;
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

        void pushQuad(Primitives primitive, GLuint texture_id, const glm::vec3& pos_bottom_left, const glm::vec3& pos_upper_left, const glm::vec3& pos_upper_right,
            const glm::vec3& pos_bottom_right, const glm::vec4& color, const glm::vec4& attributes);

        void drawPrimitives(Primitives primitive, glm::mat4& mat_mvp);

        bool createShader(vislib::graphics::gl::GLSLShader& shader, const std::string * const vertex_code, const std::string * const fragment_code);

        const std::string getShaderCode(megamol::core::utility::ShaderSourceFactory& factory, std::string snippet_name);

        size_t loadRawFile(std::wstring filename, BYTE **outData);

        void clearQueue(Primitives primitive);

        void pushShaderData(Primitives primitive, GLuint texture_id, const glm::vec3& position, const glm::vec4& color, const glm::vec2& texture_coord, const glm::vec4& attributes);

        void pushQueue(DataType& d, float v, UINT cnt = 1);

        void pushQueue(DataType& d, glm::vec2 v, UINT cnt = 1);

        void pushQueue(DataType& d, glm::vec3 v, UINT cnt = 1);

        void pushQueue(DataType& d, glm::vec4 v, UINT cnt = 1);

        glm::vec3 arbitraryPerpendicular(glm::vec3 in);

    };


    /*
    * Cinematic utility functionality (colors, menu drawing, ...).
    */
    class CinematicUtils : public RenderUtils {

    public:

        CinematicUtils(void);

        ~CinematicUtils(void);

        enum Colors {
            BACKGROUND,
            FOREGROUND,
            KEYFRAME,
            KEYFRAME_DRAGGED,
            KEYFRAME_SELECTED,
            KEYFRAME_SPLINE,
            KEYFRAME_MARKER,
            MENU,
            FONT,
            FONT_HIGHLIGHT,
            LETTER_BOX,
            FRAME_MARKER
        };

        bool Initialise(megamol::core::CoreInstance* core_instance);

        const glm::vec4 Color(CinematicUtils::Colors color) const;

        void SetBackgroundColor(glm::vec4 bc) {
            this->background_color = bc;
        };

        //void PushMenu(std::string left, std::string middle, std::string right, float viewport_width);

        //void PushHelpText(std::string text, glm::vec3 position, float width, float height);

        //void PushText(std::string text, float font_size, glm::vec3 position);

        //void DrawAll(void);

    private:

        // VARIABLES ------------------------------------------------------- //

        glm::vec4 background_color;
        //megamol::core::utility::SDFFont font;

        // FUNCTIONS ------------------------------------------------------- //

        const float lightness(glm::vec4 background) const;
        
    };

} /* end namespace cinematic */
} /* end namespace megamol */

#endif // MEGAMOL_CINEMATIC_CINEMATICUTILS_H_INCLUDED
