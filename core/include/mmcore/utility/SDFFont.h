/*
 * SDFFont.h
 *
 * Copyright (C) 2006 - 2018 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 *
 * This implementation is based on "vislib/graphics/OutlinetFont.h"
 */

#ifndef MEGAMOL_SDFFONT_H_INCLUDED
#define MEGAMOL_SDFFONT_H_INCLUDED

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/utility/AbstractFont.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/utility/ResourceWrapper.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"

#include "vislib/CharTraits.h"
#include "vislib/UTF8Encoder.h"

#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Matrix.h"

#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/File.h"
#include "vislib/Trace.h"

#include <float.h>


namespace megamol {
namespace core {
namespace utility {

    /**
     * -----------------------------------------------------------------------------------------------------------------
     * 
     * Implementation of font rendering using signed distance field texture and glyph information stored as bitmap font.
     * 
     * -----------------------------------------------------------------------------------------------------------------
     * >>> USAGE example (for megamol modules):
     *
     *     - Declare:            megamol::core::utility::SDFFont sdfFont;
     *
     *     - Ctor:               this->sdfFont(megamol::core::utility::SDFFont::FontName::EVOLVENTA_SANS);
     *                       OR: this->sdfFont("filename-of-own-font");
     *
     *     - Initialise (once):  this->sdfFont.Initialise(this->GetCoreInstance());
     *                           !!! DO NOT CALL Initialise() in CTOR because CoreInstance is not available there yet (call once e.g. in create()) !!!
     *
     *     - Draw:               this->sdfFont.DrawString(color, x, y, z, size, false, text, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
     *
     *     - RenderType:         this->sdfFont.SetRenderType(megamol::core::utility::SDFFont::RenderType::RENDERTYPE_OUTLINE);
     *     - Rotation:           this->sdfFont.SetRotation(60.0f, vislib::math::Vector<float, 3>(0.0f1.0f0.0f));
     *     - Billboard:          this->sdfFont.SetBillboard(true);
     *
     *     - BATCH-rendering     ...
     *
     *
     *
     * -----------------------------------------------------------------------------------------------------------------
     * >>> Predefined FONTS: (free for commercial use) 
     *     -> Available: Regular - TODO: Bold,Oblique,Bold-Oblique
     *
     *     - "Evolventa-SansSerif"      Source: https://evolventa.github.io/
     *     - "Roboto-SansSerif"         Source: https://www.fontsquirrel.com/fonts/roboto
     *     - "Ubuntu-Mono"              Source: https://www.fontsquirrel.com/fonts/ubuntu-mono
     *     - "Vollkorn-Serif"           Source: https://www.fontsquirrel.com/fonts/vollkorn
     *
     * -----------------------------------------------------------------------------------------------------------------
     * >>> PATH the fonts are stored: 
     *
     *     - <megamol>/share/resource/<fontname>(.fnt/.png)
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
     */

    class MEGAMOLCORE_API SDFFont : public AbstractFont {
    public:

        /** Available predefined fonts. */
        enum FontName {
            EVOLVENTA_SANS,
            ROBOTO_SANS,
            VOLLKORN_SERIF,
            UBUNTU_MONO
        };

        /** Possible render types for the font. */
        enum RenderType {
            RENDERTYPE_NONE    = 0,     // Do not render anything
            RENDERTYPE_FILL    = 1,     // Render the filled glyphs */
            RENDERTYPE_OUTLINE = 2      // Render the outline 
        };

        /**
        * Ctor.
        *
        * @param fn The predefined font name.
        */
        SDFFont(FontName fn);

        /**
        * Ctor.
        *
        * @param fn     The predefined font name.
        * @param render The render type to be used
        */
        SDFFont(FontName fn, RenderType render);

        /**
        * Ctor.
        *
        * @param fn   The predefined font name.
        * @param size The size of the font in logical units
        */
        SDFFont(FontName fn, float size);

        /**
        * Ctor.
        *
        * @param fn    The predefined font name.
        * @param flipY The vertical flip flag
        */
        SDFFont(FontName fn, bool flipY);

        /**
        * Ctor.
        *
        * @param fn     The predefined font name.
        * @param render The render type to be used
        * @param flipY  The vertical flip flag
        */
        SDFFont(FontName fn, RenderType render, bool flipY);

        /**
        * Ctor.
        *
        * @param fn    The predefined font name.
        * @param size  The size of the font in logical units
        * @param flipY The vertical flip flag
        */
        SDFFont(FontName fn, float size, bool flipY);

        /**
        * Ctor.
        *
        * @param fn     The predefined font name.
        * @param size   The size of the font in logical units
        * @param render The render type to be used
        */
        SDFFont(FontName fn, float size, RenderType render);

        /**
        * Ctor.
        *
        * @param fn     The predefined font name.
        * @param size   The size of the font in logical units
        * @param render The render type to be used
        * @param flipY  The vertical flip flag
        */
        SDFFont(FontName fn, float size, RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param fn The file name of the bitmap font.
         */
        SDFFont(vislib::StringA fn);

        /**
         * Ctor.
         *
         * @param fn     The file name of the bitmap font.
         * @param render The render type to be used
         */
        SDFFont(vislib::StringA fn, RenderType render);

        /**
         * Ctor.
         *
         * @param fn   The file name of the bitmap font.
         * @param size The size of the font in logical units
         */
        SDFFont(vislib::StringA fn, float size);

        /**
         * Ctor.
         *   
         * @param fn    The file name of the bitmap font.
         * @param flipY The vertical flip flag
         */
        SDFFont(vislib::StringA fn, bool flipY);

        /**
         * Ctor.
         *
         * @param fn     The file name of the bitmap font.
         * @param render The render type to be used
         * @param flipY  The vertical flip flag
         */
        SDFFont(vislib::StringA fn, RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param fn    The file name of the bitmap font.
         * @param size  The size of the font in logical units
         * @param flipY The vertical flip flag
         */
        SDFFont(vislib::StringA fn, float size, bool flipY);

        /**
         * Ctor.
         *
         * @param fn     The file name of the bitmap font.
         * @param size   The size of the font in logical units
         * @param render The render type to be used
         */
        SDFFont(vislib::StringA fn, float size, RenderType render);

        /**
         * Ctor.
         *
         * @param fn     The file name of the bitmap font.
         * @param size   The size of the font in logical units
         * @param render The render type to be used
         * @param flipY  The vertical flip flag
         */
        SDFFont(vislib::StringA fn, float size, RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         */
        SDFFont(const SDFFont& src);

        /**
         * Ctor.
         *
         * @param src    The source object to clone from
         * @param render The render type to be used
         */
        SDFFont(const SDFFont& src, RenderType render);

        /**
         * Ctor.
         *
         * @param src  The source object to clone from
         * @param size The size of the font in logical units
         */
        SDFFont(const SDFFont& src, float size);

        /**
         * Ctor.
         *
         * @param src   The source object to clone from
         * @param flipY The vertical flip flag
         */
        SDFFont(const SDFFont& src, bool flipY);

        /**
         * Ctor.
         *
         * @param src    The source object to clone from
         * @param render The render type to be used
         * @param flipY  The vertical flip flag
         */
        SDFFont(const SDFFont& src, RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param src   The source object to clone from
         * @param size  The size of the font in logical units
         * @param flipY The vertical flip flag
         */
        SDFFont(const SDFFont& src, float size, bool flipY);

        /**
         * Ctor.
         *
         * @param src    The source object to clone from
         * @param size   The size of the font in logical units
         * @param render The render type to be used
         */
        SDFFont(const SDFFont& src, float size, RenderType render);

        /**
         * Ctor.
         *
         * @param src    The source object to clone from
         * @param size   The size of the font in logical units
         * @param render The render type to be used
         * @param flipY  The vertical flip flag
         */
        SDFFont(const SDFFont& src, float size, RenderType render, bool flipY);

        /** Dtor. */
        virtual ~SDFFont(void);

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param c     The color as RGBA.
         * @param x     The left coordinate of the rectangle.
         * @param y     The upper coordinate of the rectangle.
         * @param w     The width of the rectangle.
         * @param h     The height of the rectangle.
         * @param size  The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt   The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        virtual void DrawString(const float col[4], float x, float y, float w, float h, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(const float col[4], float x, float y, float w, float h, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
        * Draws a text into a specified rectangular area, and performs
        * soft-breaks if necessary.
        *
        * @param c     The color as RGBA.
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
        virtual void DrawString(const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(const float col[4], float x, float y, float z, float w, float h, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
         * Draws a text at the specified position.
         *
         * @param c     The color as RGBA.
         * @param x     The x coordinate of the position.
         * @param y     The y coordinate of the position.
         * @param size  The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt   The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        virtual void DrawString(const float col[4], float x, float y, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(const float col[4], float x, float y, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
        * Draws a text at the specified position.
        *  
        * @param c     The color as RGBA.
        * @param x     The x coordinate of the position.
        * @param y     The y coordinate of the position.
        * @param z     The z coordinate of the position.
        * @param size  The size to use.
        * @param flipY The flag controlling the direction of the y-axis.
        * @param txt   The zero-terminated string to draw.
        * @param align The alignment of the text.
        */
        virtual void DrawString(const float col[4], float x, float y, float z, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(const float col[4], float x, float y, float z, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
        * Answers the width of the line 'txt' in logical units.
        *
        * @param size The font size to use.
        * @param txt  The text to measure.
        *
        * @return The width in the text in logical units.
        */
        virtual float LineWidth(float size, const char *txt) const;
        virtual float LineWidth(float size, const wchar_t *txt) const;

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
        virtual unsigned int BlockLines(float maxWidth, float size, const char *txt) const;
        virtual unsigned int BlockLines(float maxWidth, float size, const wchar_t *txt) const;

        // --- Global font settings -------------------------------------------

        /**
         * Answers the globally used render type of the font.
         *
         * @return The render type of the font
         */
        inline RenderType GetRenderType(void) const {
            return this->renderType;
        }

        /**
         * Sets the render type of the font globally.
         *
         * @param t The render type for the font
         */
        inline void SetRenderType(RenderType t) {
            this->renderType = t;           
        }

        /**
         * Enable/Disable billboard mode.
         */
        inline void SetBillboardMode(bool state) { 
            this->billboard = state; 
        }

        /**
        * Answers the globally used status of billboard mode.
        *
        * @return True if billboard mode is enabled, false otherwise.
        */
        inline bool GetBillboardMode(void) const {
            return this->billboard;
        }

        /**
        * Set font rotation globally.
        *
        * @param a The rotation angle in degrees.
        * @param v The rotation axis.
        */
        inline void SetRotation(float a, vislib::math::Vector<float, 3> v) {
            this->rotation.Set((a * (float)M_PI / 180.0f), v);
        }

        inline void SetRotation(float a, float x, float y, float z) {
            this->SetRotation(a, vislib::math::Vector<float, 3>(x, y, z));
        }

        /**
        * Reset font rotation globally.
        * (Facing in direction of positive z-Axis)
        */
        inline void ResetRotation(void) {
            this->rotation.Set(0.0f, vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
        }

        /**
        * Get the globally used rotation.
        *
        * @param a The returned angle in degrees.
        * @param v The returned rotation axis.
        */
        inline void GetRotation(float &a, vislib::math::Vector<float, 3> &v) {
            this->rotation.AngleAndAxis(a, v);
            a = (a / (float)M_PI * 180.0f);
        }

        /**
         * Enable/Disable batch draw mode.  
         * Determines that all DrawString() calls are cached for later batch draw.
         */
        inline void SetBatchDrawMode(bool state) { 
            this->useBatchDraw = state;
        }

        /**
         * Answer status of batch draw.
         * 
         * @return True if batch draw is enabled, false otherwise.
         */
        inline bool GetBatchDrawMode(void) const { 
            return this->useBatchDraw;
        }

        /**
         * Clears the batch draw caches.
         */
        inline void ClearBatchDrawCache(void) {
            this->posBatchCache.clear();
            this->texBatchCache.clear();
            this->colBatchCache.clear();
        }

        /**
         * Renders all cached string data at once.
         * Given color is used for all cached DrawString() calls (-> Faster version).
         * 
         * @param col The color.
         */
        void BatchDrawString(const float col[4]) const;

        /**
         * Renders all cached string data at once.
         * Color data from individual DrawString() calls are used in additional vbo (-> Slower version).
         */
         void BatchDrawString() const;

    protected:

        /**
         * Initialises the object. You must not call this method directly.
         * Instead call 'Initialise'. You must call 'Initialise' before the
         * object can be used.
         *
         * @param conf The megamol core isntance. Needed for being able to load files from 'share/shaders' and 'share/resources' folders.
         *
         * @return 'true' on success, 'false' on failure.
         */
        virtual bool initialise(megamol::core::CoreInstance *core);

        /**
         * Deinitialises the object. You must not call this method directly.
         * Instead call 'Deinitialise'. Derived classes must call
         * 'Deinitialise' in EACH dtor.
         */
        virtual void deinitialise(void);

    private:

        /**********************************************************************
        * variables
        **********************************************************************/

// Disable dll export warning for not exported classes in ::vislib and ::std 
#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        /** The font file name. */
        vislib::StringA fontFileName;

        /** The render type used. */
        RenderType renderType;

        /** Billboard mode. */
        bool billboard;

        /** Quaternion for rotation. */
        vislib::math::Quaternion<float> rotation;

        /** Inidcating if font could be loaded successfully. */
        bool initialised;

        /** The shader of the font. */
        vislib::graphics::gl::GLSLShader shader;
        vislib::graphics::gl::GLSLShader shadervertcol;

        /** The texture of the font. */
        vislib::graphics::gl::OpenGLTexture2D texture;

        /** Vertex buffer object attributes. */
        enum VBOAttrib {
            POSITION = 0,
            TEXTURE  = 1,
            COLOR    = 2  
        };

        /** Vertex buffer object info. */
        struct SDFVBO {
            GLuint           handle;  // buffer handle
            vislib::StringA  name;    // varaible name of attribute in shader
            GLuint           index;   // index of attribute location
            unsigned int     dim;     // dimension of data
        };
        /** Vertex array object. */
        GLuint vaoHandle;
        /** Vertex buffer objects. */
        std::vector<SDFVBO> vbos;

        /** String batch cache status. */
        bool useBatchDraw;

        /** Position, texture and color data cache for batch draw. */
        mutable std::vector<float> posBatchCache;
        mutable std::vector<float> texBatchCache;
        mutable std::vector<float> colBatchCache;


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

        // Regular font -------------------------------------------------------
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

        /** Loading font. */
        bool loadFont(megamol::core::CoreInstance *core);

        /** Load buffers. */
        bool loadFontBuffers();

        /** Load font info from file. */
        bool loadFontInfo(vislib::StringA filename);

        /** Load texture from file. */
        bool loadFontTexture(vislib::StringA filename);

        /** Load shaders from files. */
        bool loadFontShader(megamol::core::CoreInstance *core);

        /** Load file into outData buffer and return size. */
        size_t loadFile(vislib::StringA filename, BYTE **outData);

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
        int *buildGlyphRun(const char *txt,  float maxWidth) const;
        int *buildGlyphRun(const wchar_t *txt, float maxWidth) const;

        /**
        * Generates the glyph runs for the text 'txt'
        *
        * @param txtutf8  The input text in utf8 encoding
        * @param maxWidth The maximum width (normalized logical units)
        *
        * @return The resulting glyph run
        */
        int *buildUpGlyphRun(const char *txtutf8, float maxWidth) const;

        /**
        * Draw font glyphs.
        *
        * @param col   The color as RGBA.
        * @param run   The glyph run
        * @param x     The reference x coordinate
        * @param y     The reference y coordinate
        * @param z     The reference z coordinate
        * @param size  The size
        * @param flipY The flag controlling the direction of the y-axis
        * @param align The alignment
        */
        void drawGlyphs(const float col[4], int *run, float x, float y, float z, float size, bool flipY, Alignment align) const;

        /** 
        * Renders buffer data. 
        * 
        * @param gc   The total glyph count to render.
        * @param col  Pointer to the color array. If col is nullptr, per vertex color is used.
        */
        void render(unsigned int gc, const float *col[4]) const;

        /**
        * Translate enum font name into font file name.
        *
        * @param fn The predefined font name.
        */
        vislib::StringA translateFontName(FontName fn);

    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_SDFFONT_H_INCLUDED */

