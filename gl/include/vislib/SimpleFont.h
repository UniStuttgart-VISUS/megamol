/*
 * SimpleFont.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEFONT_H_INCLUDED
#define VISLIB_SIMPLEFONT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractFont.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Implementation of AbstractFont using a simple open gl texture.
     *
     * @see vislib::graphics::AbstractFont
     */
    class SimpleFont : public AbstractFont {
    public:

        /** Ctor. */
        SimpleFont(void);

        /** Dtor. */
        virtual ~SimpleFont(void);

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
         * Helper struct for a line of layouted text.
         */
        typedef struct _textline_t {

            /** Pointer to the text of this line */
            const char *text;

            /** The length of this line in characters */
            unsigned int length;

            /** The width of this line in texture space */
            float width;

        } TextLine;

        /**
         * Draws the text lines.
         *
         * @param x The x coordinate (see 'halign')
         * @param y The y coordinate (minimum; top)
         * @param lines The text lines
         * @param lineCnt The number of text lines
         * @param size The font size
         * @param flipY Flag controlling the direction of the y-axis
         * @param halign The horizontal alignment
         */
        void drawText(float x, float y, TextLine *lines, unsigned int lineCnt,
            float size, bool flipY, Alignment halign) const;

        /**
         * Enters text mode
         */
        void enterTextMode(void) const;

        /**
         * Layouts the text in separated lines.
         *
         * @param text The text to layout.
         * @param maxWidth The maximum width of the text (in texture space).
         * @param outLineCnt Receives the number of lines returned.
         *
         * @return An array of 'TextLine' structs. The caller must free the
         *         returned memory with 'delete[]' after it has been used.
         */
        TextLine* layoutText(const char *text, float maxWidth,
            unsigned int &outLineCnt) const;

        /**
         * Leaves text mode
         */
        void leaveTextMode(void) const;

        /** The open gl texture object id */
        unsigned int texId;

        /** flag if textureing was enabled when entering text mode */
        mutable bool texEnabled;

        /** flag if blending was enabled when entering text mode */
        mutable bool blendEnabled;

        /** values of the blend function when entering text mode */
        mutable int blendS, blendD;

        /** The old bound texture when entering text mode */
        mutable int oldTex;

        /** The old texture mode when entering text mode */
        mutable int oldMode;

        /** flag if depth test was enabled when entering text mode */
        mutable bool depthEnabled;

    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEFONT_H_INCLUDED */
