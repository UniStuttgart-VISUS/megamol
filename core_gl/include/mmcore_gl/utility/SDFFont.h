/**
 * MegaMol
 * Copyright (c) 2006, MegaMol Dev Team
 * All rights reserved.
 */
// This implementation is based on "vislib/graphics/OutlinetFont.h"

#pragma once

#include <memory>

#include <glm/glm.hpp>
#include <glowl/glowl.h>

#include "mmcore/CoreInstance.h"
#include "mmcore_gl/utility/RenderUtils.h"

namespace megamol::core::utility {

/**
 * -----------------------------------------------------------------------------------------------------------------
 *
 * Implementation of font rendering using signed distance field (SDF) texture and glyph information stored as bitmap font.
 *
 * -----------------------------------------------------------------------------------------------------------------
 * >>> USAGE example (for megamol modules):
 *
 *     - Declare:            megamol::core::utility::SDFFont sdfFont;
 *
 *     - Ctor:               this->sdfFont(megamol::core::utility::SDFFont::PRESET_EVOLVENTA_SANS, megamol::core::utility::SDFFont::RENDERMODE_FILL);
 *                           OR: this->sdfFont("filename-of-own-font");
 *
 *     - Initialise (once):  this->sdfFont.Initialise(this->GetCoreInstance());
 *                           !!! DO NOT CALL Initialise() in CTOR because CoreInstance is not available there yet (call once e.g. in create()) !!!
 *
 *     - Draw:               this->sdfFont.DrawString(mvm, pm, color, x, y, z, size, false, text, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
 *
 *     - Render Mode:        this->sdfFont.SetRenderMode(megamol::core::utility::SDFFont::RENDERMODE_OUTLINE);
 *
 *     - Rotation:           this->sdfFont.SetRotation(60.0f, vislib::math::Vector<float, 3>(0.0f1.0f0.0f));
 *
 *     - Billboard Mode:     this->sdfFont.SetBillboard(true);
 *                           ! Requires use of DrawString(mvm, pm, ...) version providing separate model view and projection matrices!
 *
 *     - Batch rendering     this->sdffont.SetBatchDrawMode(true);
 *                           - Call this->sdfFont.DrawString() arbitrary times ...
 *                           - Call this->sdfFont.BatchDrawString() as often as needed
 *                           - Call this->ClearBatchDrawCache() to finally clear stored batch
 *
 * -----------------------------------------------------------------------------------------------------------------
 * >>> Predefined FONTS: (free for commercial use)
 *     -> Available: Regular - TODO: Bold,Oblique,Bold-Oblique
 *
 *     - EVOLVENTA_SANS       "Evolventa-SansSerif"      Source: https://evolventa.github.io/
 *     - ROBOTO_SANS          "Roboto-SansSerif"         Source: https://www.fontsquirrel.com/fonts/roboto
 *     - UBUNTU_MONO          "Ubuntu-Mono"              Source: https://www.fontsquirrel.com/fonts/ubuntu-mono
 *     - VOLLKORN_SERIF       "Vollkorn-Serif"           Source: https://www.fontsquirrel.com/fonts/vollkorn
 *
 * -----------------------------------------------------------------------------------------------------------------
 * >>> PATH the fonts are stored:
 *
 *     - <megamol>/share/resources/<fontname>(.fnt/.png)
 *
 * -----------------------------------------------------------------------------------------------------------------
 * >>> SDF font GENERATION using "Hiero":
 *     https://github.com/libgdx/libgdx/wiki/Hiero
 *
 *     Use followings SETTINGS:
 *     - Padding - Top,Right,Bottom,Left:   10
 *     - Padding - X,Y:                    -20
 *     - Bold,Italic:                       false
 *     - Rendering:                         java
 *     - Glyph Cache Page - Width,Height:   1024
 *     - Glyph set:                         ASCII + ™ + €
 *     - Size:                             ~90 (glyphs must fit on !one! page)
 *     - Distance Field - Spread:           10
 *     - Distance Field - Scale:            50 (set in the end, operation is expensive)
 *
 * -----------------------------------------------------------------------------------------------------------------
 *
 * These fonts can render text onto the currently active graphics context
 * in the object space x-y-plane. The class also contains metric
 * functions.
 *
 * The planes are defined as follows:
 *  The positive direction of the x-axis is to the right.
 *  The positive direction of the y-axis is downwards.
 * However, you can change the direction of the y-axis to upwards by
 * setting the 'flipY' flag.
 *
 * There are two types of 'DrawString' methods, which perform a different
 * text alignment. When using the methods using a single point for
 * positioning the text the alignment specifies in which corner of the
 * text string the position point should be:
 *
 *         Left     Center   Right
 *
 * Top     A-----+  +--A--+  +-----A
 *         | str |  | str |  | str |
 *         +-----+  +-----+  +-----+
 *
 * Center  +-----+  +-----+  +-----+
 *         A str |  | sAr |  | str A
 *         +-----+  +-----+  +-----+
 *
 * Bottom  +-----+  +-----+  +-----+
 *         | str |  | str |  | str |
 *         A-----+  +--A--+  +-----A
 *
 * When using the methods which use a rectangle to specify the text
 * position the alignment specifies in which corner of that rectangle the
 * text should be placed. The positions (x, y) specifies the minimum values
 * on both axis (regardless 'flipY' flag) and the size (w, h) should always
 * be positive.
 * -----------------------------------------------------------------------------------------------------------------
 */

class SDFFont {
public:
    // clang-format off

        /** Available predefined fonts. */
        enum PresetFontName : int {
            PRESET_EVOLVENTA_SANS = 0,
            PRESET_ROBOTO_SANS    = 1,
            PRESET_VOLLKORN_SERIF = 2,
            PRESET_UBUNTU_MONO    = 3
        };

        /** Possible render modes for the font. */
        enum RenderMode : int {
            RENDERMODE_NONE    = 0,     // Do not render anything
            RENDERMODE_FILL    = 1,     // Render the filled glyphs */
            RENDERMODE_OUTLINE = 2      // Render the outline 
        };

        /** Possible values for the text alignment. */
        enum Alignment {
            ALIGN_LEFT_TOP      = 0x00,
            ALIGN_CENTER_TOP    = 0x01,
            ALIGN_RIGHT_TOP     = 0x02,
            ALIGN_LEFT_MIDDLE   = 0x10,
            ALIGN_CENTER_MIDDLE = 0x11,
            ALIGN_RIGHT_MIDDLE  = 0x12,
            ALIGN_LEFT_BOTTOM   = 0x20,
            ALIGN_CENTER_BOTTOM = 0x21,
            ALIGN_RIGHT_BOTTOM  = 0x22
        };

        /**
        * Ctor.
        *
        * @param fn          The predefined font name or the font name as string
        * @param render_mode The render to be used
        * @param size        The size of the font in logical units 
        * @param flipY       The vertical flip flag
        */
        SDFFont(PresetFontName fn);
        SDFFont(PresetFontName fn, RenderMode render_mode);
        SDFFont(PresetFontName fn, float size);
        SDFFont(PresetFontName fn, bool flipY);
        SDFFont(PresetFontName fn, RenderMode render_mode, bool flipY);
        SDFFont(PresetFontName fn, float size, bool flipY);
        SDFFont(PresetFontName fn, float size, RenderMode render_mode);
        SDFFont(PresetFontName fn, float size, RenderMode render_mode, bool flipY);

        SDFFont(std::string fn);
        SDFFont(std::string fn, RenderMode render_mode);
        SDFFont(std::string fn, float size);
        SDFFont(std::string fn, bool flipY);
        SDFFont(std::string fn, RenderMode render_mode, bool flipY);
        SDFFont(std::string fn, float size, bool flipY);
        SDFFont(std::string fn, float size, RenderMode render_mode);
        SDFFont(std::string fn, float size, RenderMode render_mode, bool flipY);

        SDFFont(const SDFFont& src);

        /** Dtor. */
        ~SDFFont(void);

        /**
         * Initialises the object. You must not call this method directly.
         * Instead call 'Initialise'. You must call 'Initialise' before the
         * object can be used.
         *
         * @param conf The megamol core isntance. Needed for being able to load files from 'share/shaders' and 'share/resources' folders.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool Initialise(megamol::core::CoreInstance* core_instance_ptr, megamol::frontend_resources::RuntimeConfig const& runtimeConf);

        /**
         * Deinitialises the object.
         */
        void Deinitialise(void);

        /**
         * 
         * Draws text into a specified rectangular area in world space.
         * Performs soft-breaks if necessary.
         *
         * @param mvm   The current [3D]: model view matrix [2D]: const glm::mat4&().
         * @param pm    The current [3D]: projection matrix [2D]: pre-multiplied model view projection matrix.
         * 
         * @param mvp    The current pre-multiplied model view projection matrix.
         *
         * @param col   The color as RGBA.
         * @param x     The left coordinate of the rectangle.
         * @param y     The upper coordinate of the rectangle.
         * @param z     The z coordinate of the position.
         * @param w     The width of the rectangle.
         * @param h     The height of the rectangle.
         * @param size  The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt   The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        // mvm pm col x y size flipy txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const;
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, size, flipY, this->to_string(txt).c_str(), align);
        }
        // mvm pm col x y z size flipy txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float z, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const;
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float z, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, z, size, flipY, this->to_string(txt).c_str(), align);
        }
        // mvm pm col x y w h size flipy txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float w, float h, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const;
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float w, float h, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, w, h, size, flipY, this->to_string(txt).c_str(), align);
        }
        // mvm pm col x y z w h size flipy txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const;
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, z, w, h, size, flipY, this->to_string(txt).c_str(), align);
        }

        // mvp col x y size flipy txt (align)
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float size, bool flipY, const char* txt,  Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(glm::mat4(1.0f), mvp, col, x, y, size, flipY, txt, align);
        }
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvp, col, x, y, size, flipY, this->to_string(txt).c_str(), align);
        }
        // mvp col x y z size flipy txt (align)
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float z, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(glm::mat4(1.0f), mvp, col, x, y, z, size, flipY, txt, align);
        }
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float z, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvp, col, x, y, z, size, flipY, this->to_string(txt).c_str(), align);
        }
        // mvp col x y w h size flipy txt (align)
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float w, float h, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(glm::mat4(1.0f), mvp, col, x, y, w, h, size, flipY, txt, align);
        }
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float w, float h, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvp, col, x, y, w, h, size, flipY, this->to_string(txt).c_str(), align);
        }
        // mvp col x y w h z size flipy txt (align)
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(glm::mat4(1.0f), mvp, col, x, y, z, w, h, size, flipY, txt, align);
        }
        void DrawString(const glm::mat4& mvp, const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvp, col, x, y, z, w, h, size, flipY, this->to_string(txt).c_str(), align);
        }
        
        /*
        // mvm pm col x y txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, const char* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, this->globalSize, this->globalFlipY, txt, align);
        }
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, this->globalSize, this->globalFlipY, this->to_string(txt).c_str(), align);
        }
        // mvm pm col x y size txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float size, const char* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, size, this->globalFlipY, txt, align);
        }
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, float size, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, size, this->globalFlipY, this->to_string(txt).c_str(), align);
        }
        // mvm pm col x y flipy txt (align)
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, bool flipY, const char* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, this->globalSize, flipY, txt, align);
        }
        void DrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], float x, float y, bool flipY, const wchar_t* txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(mvm, pm, col, x, y, this->globalSize, flipY, this->to_string(txt).c_str(), align);
        }
        */

        // mvm pm col x y z txt (align)
        // mvm pm col x y z size txt (align)
        // mvm pm col x y z flipy txt (align)

        // mvm pm col x y w h txt (align)
        // mvm pm col x y w h size txt (align)
        // mvm pm col x y w h flipy txt (align)

        // mvm pm col x y z w h txt (align)
        // mvm pm col x y z w h size txt (align)
        // mvm pm col x y z w h flipy txt (align)

        // Batch Draw Mode ----------------------------------------------------

        /**
         * Enable/Disable batch draw mode.  
         * Determines that all DrawString() calls are cached for later batch draw.
         */
        inline void SetBatchDrawMode(bool state) { 
            this->batchDrawMode = state;
        }

        /**
         * Answer status of batch draw.
         *
         * @return True if batch draw is enabled, false otherwise.
         */
        inline bool GetBatchDrawMode(void) const {
            return this->batchDrawMode;
        }

        inline void SetSmoothMode(bool state) {
            this->smoothMode = state;
        }

        inline bool GetSmoothMode() const {
            return this->smoothMode;
        }

        /**
         * Renders all cached string data at once.
         * Given color is used for all cached DrawString() calls (-> Faster version).
         *
         * @param mvp    The current orthographic projection matrix.
         * 
         * @param pm    The current projection matrix.
         * @param mvm   The current model view matrix.
         * @param col   The color.
         */
        void BatchDrawString(const glm::mat4& mvm, const glm::mat4& pm, const float col[4]) const;
        void BatchDrawString(const glm::mat4& mvp, const float col[4]) const {
            this->BatchDrawString(glm::mat4(1.0f), mvp, col);
        }

        /**
         * Renders all cached string data at once.
         * Color data from individual DrawString() calls are used in additional vbo (-> Slower version).
         *
         * @param mvp    The current orthographic projection matrix.
         * 
         * @param pm    The current projection matrix.
         * @param mvm   The current model view matrix.
         */
        void BatchDrawString(const glm::mat4& mvm, const glm::mat4& pm) const;
        void BatchDrawString(const glm::mat4& mvp) const {
            this->BatchDrawString(glm::mat4(1.0f), mvp);
        }

        /**
         * Clears the batch draw caches.
         */
        inline void ClearBatchDrawCache(void) {
            this->posBatchCache.clear();
            this->texBatchCache.clear();
            this->colBatchCache.clear();
        }

        // Global Font Settings -----------------------------------------------

        /**
         * Answers the height of a single line in logical units. This default
         * implementation returns 'size', since this is the value representing
         * a line height. The line height is always positive.
         *
         * @param size The font size to use.
         *
         * @return The height of a single line in logical units.
         */
        float LineHeight(float size) const {
            return std::max(0.0f, size);
        }

        /**
         * Answers the height of a single line with the default font size in
         * logical units. The line height is always positive.
         *
         * @return The height of a single line in logical units.
         */
        inline float LineHeight(void) const {
            return this->LineHeight(this->globalSize);
        }

        /**
        * Answers the width of the line 'txt' in logical units.
        *
        * @param size The font size to use.
        * @param txt  The text to measure.
        *
        * @return The width in the text in logical units.
        */
        float LineWidth(float size, const char *txt) const;
        float LineWidth(float size, const wchar_t* txt) const {
            return this->LineWidth(size, this->to_string(txt).c_str());
        }

        /**
        * Calculates the height of a text block in number of lines, when
        * drawn with the rectangle-based versions of 'DrawString' with the
        * specified maximum width and font size.
        *
        * @param maxWidth The maximum width.
        * @param size     The font size to use.
        * @param txt      The text to measure.
        *
        * @return The height of the text block in number of lines.
        */
        unsigned int BlockLines(float maxWidth, float size, const char *txt) const;
        unsigned int BlockLines(float maxWidth, float size, const wchar_t* txt) const {
            return this->BlockLines(maxWidth, size, this->to_string(txt).c_str());
        }

        /**
         * Sets the render_mode type of the font globally.
         *
         * @param t The render_mode type for the font
         */
        inline void SetRenderMode(RenderMode rm) {
            this->renderMode = rm;
        }

        /**
         * Answers the globally used render mode of the font.
         *
         * @return The render mode of the font
         */
        inline RenderMode GetRenderMode(void) const {
            return this->renderMode;
        }

        /**
         * Enable/Disable billboard mode.
         */
        inline void SetBillboardMode(bool state) { 
            this->billboardMode = state; 
        }

        /**
         * Answers the globally used status of billboard mode.
         *
         * @return True if billboard mode is enabled, false otherwise.
         */
        inline bool GetBillboardMode(void) const {
            return this->billboardMode;
        }

        /**
        * Set font rotation globally.
        *
        * @param a The rotation angle in degrees.
        * @param v The rotation axis.
        */
        inline void SetRotation(float a, glm::vec3 v) {
            this->rotation = glm::angleAxis(glm::radians(a), v);
        }

        inline void SetRotation(float a, float x, float y, float z) {
            this->SetRotation(a, glm::vec3(x, y, z));
        }

        /**
        * Reset font rotation globally.
        * (Facing in direction of positive z-Axis)
        */
        inline void ResetRotation(void) {
            this->rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        }

        /**
         * Set outline color.
         *
         * @param color     The color of the outline.
         * @param thikness  The thickness of the outline in per cent to the total font width in range [0, 1]
         */
        inline void SetOutline(glm::vec3 color, float thickness = 0.2f) { 
            this->outlineColor = glm::vec3(std::clamp(color.x, 0.0f, 1.0f), std::clamp(color.y, 0.0f, 1.0f), std::clamp(color.z, 0.0f, 1.0f));
            this->outlineThickness = std::clamp(thickness, 0.0f, 1.0f);
        }

        /**
         * Gets the default size of the font. The size is specified in logical
         * units used to measure the text in object space.
         *
         * @return The default size of the font.
         */
        inline float GetSize(void) const {
            return this->globalSize;
        }

        /**
         * Sets the default size of the font. The size is specified in logical
         * units used to measure the text in object space.
         *
         * @param size The new default size of the font.
         *
         * @throw IllegalParamException if size is less than zero.
         */
        void SetSize(float size) {
            this->globalSize = std::max(0.0f, size);
        }

        /**
         * Sets the flag 'flipY'. If the flag is set, the direction of the
         * y-axis is upward, otherwise it is downward.
         *
         * @param flipY The new value for the 'flipY' flag.
         */
        void SetFlipY(bool flipY) {
            this->globalFlipY = flipY;
        }

        /**
         * Answer the flag 'flipY'. If 'flipY' is true, the direction of the
         * y-axis is upward, otherwise the direction is downward.
         *
         * @return The flag 'flipY'
         */
        inline bool IsFlipY(void) const {
            return this->globalFlipY;
        }

    private:

        /**********************************************************************
        * variables
        **********************************************************************/

// Disable dll export warning for not exported classes in ::vislib and ::std 
#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** Indicating if font could be loaded successfully. */
        bool initialised;

        /** The font file name. */
        std::string fontFileName;

        /** The render_mode type used. */
        RenderMode renderMode;

        /** Billboard mode. */
        bool billboardMode;

        /** String batch cache status. */
        bool batchDrawMode;

        /** Smooth mode. TODO 'off' only works with renderMode filled. */
        bool smoothMode;

        /** Quaternion for font rotation. */
        glm::quat rotation;

        /** Optional outline data. */
        glm::vec3 outlineColor;
        float outlineThickness;

        /** The default size of the font */
        float globalSize;

        /** flag whether to flip the y-axis */
        bool globalFlipY;

        // Render data --------------------------------------------------------

        /** The shaders of the font for different color modes. */
        std::shared_ptr<glowl::GLSLProgram> shaderglobcol;
        std::shared_ptr<glowl::GLSLProgram> shadervertcol;

        /** The texture of the font. */
        std::shared_ptr<glowl::Texture2D> texture;

        /** Vertex buffer object attributes. */
        enum VBOAttrib : int {
            POSITION = 0,
            TEXTURE  = 1,
            COLOR    = 2  
        };
        /** Vertex buffer object info. */
        struct SDFVBO {
            GLuint           handle;  // buffer handle
            std::string      name;    // varaible name of attribute in shader
            GLuint           index;   // index of attribute location
            unsigned int     dim;     // dimension of data
        };
        /** Vertex array object. */
        GLuint vaoHandle;
        /** Vertex buffer objects. */
        std::vector<SDFVBO> vbos;

        /** Position, texture and color data cache for batch draw. */
        mutable std::vector<float> posBatchCache;
        mutable std::vector<float> texBatchCache;
        mutable std::vector<float> colBatchCache;

        // Font data ----------------------------------------------------------

        /** The glyph kernings. */
        struct SDFGlyphKerning {
            unsigned int previous;  // The previous character id
            unsigned int current;   // The current character id
            float        xamount;   // How much the x position should be adjusted when drawing this character immediately following the previous one
        };

        /** The SDF glyph info. */
        struct SDFGlyphInfo {
            unsigned int id;             // The character id
            float             texX0;     // The left position of the character image in the texture
            float             texY0;     // The top position of the character image in the texture
            float             texX1;     // The right position of the character image in the texture
            float             texY1;     // The bottom position of the character image in the texture
            float             width;     // The width of the character 
            float             height;    // The height of the character 
            float             xoffset;   // How much the current position should be offset when copying the image from the texture to the screen
            float             yoffset;   // How much the current position should be offset when copying the image from the texture to the screen
            float             xadvance;  // How much the current position should be advanced after drawing the character
            unsigned int      kernCnt;   // Number of kernings in array
            SDFGlyphKerning  *kerns;     // Array of kernings
        };

        /** The glyphs. */
        std::vector<SDFGlyphInfo>    glyphs;
        /** The glyphs sorted by index. */
        std::vector<SDFGlyphInfo*>   glyphIdcs;
        /** The glyph kernings. */
        std::vector<SDFGlyphKerning> glyphKrns;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /**********************************************************************
        * functions
        **********************************************************************/

        bool loadFont(megamol::core::CoreInstance *core_instance_ptr, megamol::frontend_resources::RuntimeConfig const& runtimeConf);

        bool loadFontBuffers();

        bool loadFontInfo(vislib::StringW filename);

        bool loadFontShader(megamol::frontend_resources::RuntimeConfig const& runtimeConf);

        /**
        * Answer the number of lines in the glyph run
        *
        * @param run The glyph run
        * @param deleterun Deletes the glyph run after use
        *
        * @return the number of lines.
        */
        int lineCount(int *run, bool deleterun) const;

        /**
        * Answer the width of the line 'run' starts.
        *
        * @param run     The glyph run
        * @param iterate If 'true' 'run' will be set to point to the first
        *                glyph of the next line. If 'false' the value of
        *                'run' will not be changed
        *
        * @return The width of the line
        */
        float lineWidth(int *&run, bool iterate) const;

        /**
        * Generates the glyph runs for the text 'txt'
        *
        * @param txt      The input text 
        * @param maxWidth The maximum width (normalized logical units)
        *
        * @return The resulting glyph run
        */
        int *buildGlyphRun(const char *txt, float maxWidth) const;

        /**
        * Draw font glyphs.
        *
        * @param pm     The current projection matrix.
        * @param mvm    The current modelview matrix.
        * @param col    The color as RGBA.
        * @param run    The glyph run
        * @param x      The reference x coordinate
        * @param y      The reference y coordinate
        * @param z      The reference z coordinate
        * @param size   The size
        * @param flipY  The flag controlling the direction of the y-axis
        * @param align  The alignment
        */
        void drawGlyphs(const glm::mat4& mvm, const glm::mat4& pm, const float col[4], int* run, float x,
            float y, float z, float size, bool flipY, Alignment align) const;

        /** 
        * Renders buffer data. 
        *
        * @param pm           The current projection matrix.
        * @param mvm          The current modelview matrix.
        * @param glyph_count  The total glyph count to render.
        * @param color_ptr    Pointer to the color array. If col is nullptr, per vertex color is used.
        */
        void render(const glm::mat4& mvm, const glm::mat4& pm, unsigned int glyph_count,
            const float* color_ptr[4]) const;

        /**
        * Translate enum font name into font file name.
        *
        * @param fn The predefined font name.
        */
        inline std::string presetFontNameToString(PresetFontName fn) const ;

        // Convert wchar_t to char
        inline const std::string to_string(const wchar_t* wstr) const {
            return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(wstr);
        }

    // clang-format on
};

} // namespace megamol::core::utility
