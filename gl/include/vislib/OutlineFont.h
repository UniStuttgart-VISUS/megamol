/*
 * OutlineFont.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OUTLINEFONT_H_INCLUDED
#define VISLIB_OUTLINEFONT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractFont.h"
#include "vislib/forceinline.h"


namespace vislib {
namespace graphics {
namespace gl {

#ifndef VISLIB_OUTLINEGLYPHINFO_STRUCT
#define VISLIB_OUTLINEGLYPHINFO_STRUCT

    typedef struct _outlineglyphinfo_t {
        const float width; /* The glyph width */
        const unsigned short loopCount; /* The number of outline loops */
        const unsigned short *loopLength; /* The length of each outline loop */
        const float *points; /* The points for each outline loop */
        const unsigned short triCount; /* The number of vertices of the
                                        triangles used to fill the glyph */
        const unsigned short *tris; /* Index buffer of the triangles used */
    } OutlineGlyphInfo;

#endif /* VISLIB_OUTLINEGLYPHINFO_STRUCT */

#ifndef VISLIB_OUTLINEFONTINFO_STRUCT
#define VISLIB_OUTLINEFONTINFO_STRUCT

    /**
     * Utility structure storing the data of font glyph outlines
     */
    typedef struct _outlinefontinfo_t {
        const char *name; /* The name of the font */
        const float baseline; /* The distance of the base line from top */
        const float charHeight; /* The main height of a character */
        const float charTop; /* The top line above the main height */
        const float charBottom; /* The bottom line below the base line */
        const unsigned short glyphCount; /* The number of glyphs */
        const OutlineGlyphInfo *glyph; /* The glyph data, sorted acending
                                          according to 'utf8char' */
        const short *glyphIndex; /* The glyph search index */
    } OutlineFontInfo;

#endif /* VISLIB_OUTLINEFONTINFO_STRUCT */


    /**
     * Implementation of simple font using outline glyph information to render
     * the font
     */
    class OutlineFont : public AbstractFont {
    public:

        /** Possible render types for the font */
        enum RenderType {
            RENDERTYPE_NONE, /* Do not render anything */
            RENDERTYPE_OUTLINE, /* Render the outline with GL_LINE_LOOP */
            RENDERTYPE_FILL, /* Render the filled glyphs with GL_TRIANGLES */
            RENDERTYPE_FILL_AND_OUTLINE /* Render the filled glyphs with
                                           GL_TRIANGLES and the outline with
                                           GL_LINE_LOOP afterwards */
        };

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         */
        OutlineFont(const OutlineFontInfo& ofi);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param render The render type to be used
         */
        OutlineFont(const OutlineFontInfo& ofi, RenderType render);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param size The size of the font in logical units
         */
        OutlineFont(const OutlineFontInfo& ofi, float size);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFontInfo& ofi, bool flipY);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param render The render type to be used
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFontInfo& ofi, RenderType render,
            bool flipY);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param size The size of the font in logical units
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFontInfo& ofi, float size, bool flipY);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param size The size of the font in logical units
         * @param render The render type to be used
         */
        OutlineFont(const OutlineFontInfo& ofi, float size,
            RenderType render);

        /**
         * Ctor.
         *
         * @param ofi The outline font info of the font
         * @param size The size of the font in logical units
         * @param render The render type to be used
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFontInfo& ofi, float size,
            RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         */
        OutlineFont(const OutlineFont& src);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param render The render type to be used
         */
        OutlineFont(const OutlineFont& src, RenderType render);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param size The size of the font in logical units
         */
        OutlineFont(const OutlineFont& src, float size);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFont& src, bool flipY);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param render The render type to be used
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFont& src, RenderType render, bool flipY);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param size The size of the font in logical units
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFont& src, float size, bool flipY);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param size The size of the font in logical units
         * @param render The render type to be used
         */
        OutlineFont(const OutlineFont& src, float size, RenderType render);

        /**
         * Ctor.
         *
         * @param src The source object to clone from
         * @param size The size of the font in logical units
         * @param render The render type to be used
         * @param flipY The vertical flip flag
         */
        OutlineFont(const OutlineFont& src, float size, RenderType render,
            bool flipY);

        /** Dtor. */
        virtual ~OutlineFont(void);

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and font size.
         *
         * @param maxWidth The maximum width.
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        virtual unsigned int BlockLines(float maxWidth, float size,
            const char *txt) const;

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and font size.
         *
         * @param maxWidth The maximum width.
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        virtual unsigned int BlockLines(float maxWidth, float size,
            const wchar_t *txt) const;

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param size The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        virtual void DrawString(float x, float y, float size, bool flipY,
            const char *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param size The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        virtual void DrawString(float x, float y, float w, float h, float size,
            bool flipY, const char *txt, Alignment align = ALIGN_LEFT_TOP)
            const;

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param size The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        virtual void DrawString(float x, float y, float size, bool flipY,
            const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const;

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param size The size to use.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        virtual void DrawString(float x, float y, float w, float h, float size,
            bool flipY, const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP)
            const;

        /**
         * Answers the render type of the font
         *
         * @return The render type of the font
         */
        inline RenderType GetRenderType(void) const {
            return this->renderType;
        }

        /**
         * Answers the width of the line 'txt' in logical units.
         *
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        virtual float LineWidth(float size, const char *txt) const;

        /**
         * Answers the width of the line 'txt' in logical units.
         *
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        virtual float LineWidth(float size, const wchar_t *txt) const;

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
         * @return 'true' on success, 'false' on failure.
         */
        virtual bool initialise(void);

        /**
         * Deinitialises the object. You must not call this method directly.
         * Instead call 'Deinitialise'. Derived classes must call
         * 'Deinitialise' in EACH dtor.
         */
        virtual void deinitialise(void);

    private:

        /**
         * Generates the glyph runs for the text 'txt'
         *
         * @param txt The input text
         * @param maxWidth The maximum width (normalized logical units)
         *
         * @return The resulting glyph run
         */
        VISLIB_FORCEINLINE int *buildGlyphRun(const char *txt,
            float maxWidth) const;

        /**
         * Generates the glyph runs for the text 'txt'
         *
         * @param txt The input text
         * @param maxWidth The maximum width (normalized logical units)
         *
         * @return The resulting glyph run
         */
        VISLIB_FORCEINLINE int *buildGlyphRun(const wchar_t *txt,
            float maxWidth) const;

        /**
         * Generates the glyph runs for the text 'txt'
         *
         * @param txtutf8 The input text in utf8 encoding
         * @param maxWidth The maximum width (normalized logical units)
         *
         * @return The resulting glyph run
         */
        int *buildUpGlyphRun(const char *txtutf8, float maxWidth) const;

        /**
         * Draws an filled glyph run
         *
         * @param run The glyph run
         * @param x The reference x coordinate
         * @param y The reference y coordinate
         * @param size The size
         * @param flipY The flag controlling the direction of the y-axis
         * @param align The alignment
         */
        void drawFilled(int *run, float x, float y, float size, bool flipY,
            Alignment align) const;

        /**
         * Draws an outline glyph run
         *
         * @param run The glyph run
         * @param x The reference x coordinate
         * @param y The reference y coordinate
         * @param size The size
         * @param flipY The flag controlling the direction of the y-axis
         * @param align The alignment
         */
        void drawOutline(int *run, float x, float y, float size, bool flipY,
            Alignment align) const;

        /**
         * Answer the number of lines in the glyph run
         *
         * @param run The glyph run
         * @param deleterun Deletes the glyph run after use
         *
         * @return the number of lines.
         */
        VISLIB_FORCEINLINE int lineCount(int *run, bool deleterun) const;

        /**
         * Answer the width of the line 'run' starts.
         *
         * @param run The glyph run
         * @param iterate If 'true' 'run' will be set to point to the first
         *                glyph of the next line. If 'false' the value of
         *                'run' will not be changed
         *
         * @return The width of the line
         */
        VISLIB_FORCEINLINE float lineWidth(int *&run, bool iterate) const;

        /** The font outline info data */
        const OutlineFontInfo& data;

        /** The render type used */
        RenderType renderType;

    };

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_OUTLINEFONT_H_INCLUDED */

