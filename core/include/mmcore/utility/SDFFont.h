/*
 * SDFFont.h
 *
 * Copyright (C) 2006 - 2018 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_SDFFONT_H_INCLUDED
#define MEGAMOL_SDFFONT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/utility/AbstractFont.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"

#include <vector>


namespace megamol {
    namespace core {
        namespace utility {

    /**
     * -----------------------------------------------------------------------------------------------------------------
     * Implementation of font rendering using signed distance field texture and glyph information stored as bitmap font.
     * -----------------------------------------------------------------------------------------------------------------
     *
     * USAGE EXAMPLE:
     *
     *     DECLARE:     megamol::core::utility::SDFFont sdfFont;
     *     CTOR:        this->sdfFont(megamol::core::utility::SDFFont::BitmapFont::VERDANA)
     *     INITIALISE:  this->sdfFont.Initialise(this->GetCoreInstance())
     *     RENDERTYPE:  this->sdfFont.SetRenderType(megamol::core::utility::SDFFont::RenderType::RENDERTYPE_OUTLINE)
     *     DRAW:        this->sdfFont.DrawString(x, y, sdfWidth, 1.0f, fontSize, true, sdfString, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP)
     *
     */
    class MEGAMOLCORE_API SDFFont : public AbstractFont {
    public:

        /** Available predefined open source bitmap fonts. */
        enum BitmapFont {
            EVOLVENTA,
            VERDANA
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
         * @param ofi The outline font info of the font
         */
        SDFFont(const BitmapFont bmf);

        /**
         * Ctor.
         *
         * @param ofi    The outline font info of the font
         * @param render The render type to be used
         */
        SDFFont(const BitmapFont bmf, RenderType render);

        /**
         * Ctor.
         *
         * @param ofi  The outline font info of the font
         * @param size The size of the font in logical units
         */
        SDFFont(const BitmapFont bmf, float size);

        /**
         * Ctor.
         *
         * @param ofi   The outline font info of the font
         * @param flipY The vertical flip flag
         */
        SDFFont(const BitmapFont bmf, bool flipY);

        /**
         * Ctor.
         *
         * @param ofi    The outline font info of the font
         * @param render The render type to be used
         * @param flipY  The vertical flip flag
         */
        SDFFont(const BitmapFont bmf, RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param ofi   The outline font info of the font
         * @param size  The size of the font in logical units
         * @param flipY The vertical flip flag
         */
        SDFFont(const BitmapFont bmf, float size, bool flipY);

        /**
         * Ctor.
         *
         * @param ofi    The outline font info of the font
         * @param size   The size of the font in logical units
         * @param render The render type to be used
         */
        SDFFont(const BitmapFont bmf, float size, RenderType render);

        /**
         * Ctor.
         *
         * @param ofi    The outline font info of the font
         * @param size   The size of the font in logical units
         * @param render The render type to be used
         * @param flipY  The vertical flip flag
         */
        SDFFont(const BitmapFont bmf, float size, RenderType render, bool flipY);

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
         * @param x     The left coordinate of the rectangle.
         * @param y     The upper coordinate of the rectangle.
         * @param w     The width of the rectangle.
         * @param h     The height of the rectangle.
         * @param size  The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt   The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        virtual void DrawString(float x, float y, float w, float h, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(float x, float y, float w, float h, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
         * Draws a text at the specified position.
         *
         * @param x     The x coordinate of the position.
         * @param y     The y coordinate of the position.
         * @param size  The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt   The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        virtual void DrawString(float x, float y, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(float x, float y, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
        * Draws a text at the specified position.
        *
        * @param x     The x coordinate of the position.
        * @param y     The y coordinate of the position.
        * @param z     The z coordinate of the position.
        * @param size  The size to use.
        * @param flipY The flag controlling the direction of the y-axis.
        * @param txt   The zero-terminated string to draw.
        * @param align The alignment of the text.
        */
        virtual void DrawString(float x, float y, float z, float size, bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP) const;
        virtual void DrawString(float x, float y, float z, float size, bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

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

        /**
         * Answers the render type of the font
         *
         * @return The render type of the font
         */
        inline RenderType GetRenderType(void) const {
            return this->renderType;
        }

        /**
         * Sets the render type of the font
         *
         * @param t The render type for the font
         */
        inline void SetRenderType(RenderType t) {
            this->renderType = t;           
        }

    protected:

        /**
         * Initialises the object. You must not call this method directly.
         * Instead call 'Initialise'. You must call 'Initialise' before the
         * object can be used.
         *
         * @param conf The megamol core isntance, neeeded for being able to load shader/resource files.
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

        /** The sdf font. */
        BitmapFont font;

        /** The render type used. */
        RenderType renderType;

        /** Inidcating if font could be loaded successfully. */
        bool initialised;

        /** The shader of the font. */
        vislib::graphics::gl::GLSLShader shader;

        /** The texture of the font. */
        vislib::graphics::gl::OpenGLTexture2D texture;

        /** Vertex buffer object attributes. */
        enum VBOAttrib {
            POSITION = 0,
            TEXTURE = 1
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
        std::vector<SDFGlyphInfo> glyphs;
        /** The glyphs sorted by index. */
        SDFGlyphInfo **glyphIdx;
        /** Numbner of indices in index array. */
        unsigned int   idxCnt;
        /** The glyph kernings. */
        std::vector<SDFGlyphKerning> kernings;

        // Bold font ----------------------------------------------------------
        /** The glyphs. */
        //std::vector<SDFGlyphInfo> glyphsBold;
        /** The glyphs sorted by index. */
        //SDFGlyphInfo **glyphIdxBold;
        /** Numbner of indices in index array. */
        //unsigned int   idxCntBold;
        /** The glyph kernings. */
        //std::vector<SDFGlyphKerning> kerningsBold;

        // Oblique font -------------------------------------------------------
        /** The glyphs. */
        //std::vector<SDFGlyphInfo> glyphsOblique;
        /** The glyphs sorted by index. */
        //SDFGlyphInfo **glyphIdxOblique;
        /** Numbner of indices in index array. */
        //unsigned int   idxCntOblique;
        /** The glyph kernings. */
        //std::vector<SDFGlyphKerning> kerningsOblique;

        // Bold and Oblique font ----------------------------------------------
        /** The glyphs. */
        //std::vector<SDFGlyphInfo> glyphsBoldOblique;
        /** The glyphs sorted by index. */
        //SDFGlyphInfo **glyphIdxBoldOblique;
        /** Numbner of indices in index array. */
        //unsigned int   idxCntBoldOblique;
        /** The glyph kernings. */
        //std::vector<SDFGlyphKerning> kerningsBoldOblique;


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
        bool loadFontShader(megamol::core::CoreInstance *core, vislib::StringA vert, vislib::StringA frag);

        /** Load file into outData buffer and return size. */
        SIZE_T loadFile(vislib::StringA filename, void **outData);

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
        * @param run   The glyph run
        * @param x     The reference x coordinate
        * @param y     The reference y coordinate
        * @param z     The reference z coordinate
        * @param size  The size
        * @param flipY The flag controlling the direction of the y-axis
        * @param align The alignment
        */
        void draw(int *run, float x, float y, float z, float size, bool flipY, Alignment align) const;

    };

        } /* end namespace utility */
    } /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_SDFFONT_H_INCLUDED */

