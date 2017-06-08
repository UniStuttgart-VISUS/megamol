/*
 * AbstractFont.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTFONT_H_INCLUDED
#define VISLIB_ABSTRACTFONT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/String.h"

namespace vislib {
namespace graphics {


    /**
     * Abstract base class for graphics fonts.
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
     */
    class AbstractFont {
    public:

        /** Possible values for the text alignment */
        enum Alignment {
            ALIGN_LEFT_TOP = 0x00,
            ALIGN_CENTER_TOP = 0x01,
            ALIGN_RIGHT_TOP = 0x02,
            ALIGN_LEFT_MIDDLE = 0x10,
            ALIGN_CENTER_MIDDLE = 0x11,
            ALIGN_RIGHT_MIDDLE = 0x12,
            ALIGN_LEFT_BOTTOM = 0x20,
            ALIGN_CENTER_BOTTOM = 0x21,
            ALIGN_RIGHT_BOTTOM = 0x22
        };

        /** Dtor. */
        virtual ~AbstractFont(void);

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and the default font size.
         *
         * @param maxWidth The maximum width.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        inline unsigned int BlockLines(float maxWidth, const char *txt) const {
            return this->BlockLines(maxWidth, this->size, txt);
        }

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and the default font size.
         *
         * @param maxWidth The maximum width.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        inline unsigned int BlockLines(float maxWidth,
                const vislib::StringA& txt) const {
            return this->BlockLines(maxWidth, this->size, txt.PeekBuffer());
        }

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and the default font size.
         *
         * @param maxWidth The maximum width.
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        inline unsigned int BlockLines(float maxWidth, float size,
                const vislib::StringA& txt) const {
            return this->BlockLines(maxWidth, size, txt.PeekBuffer());
        }

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and the default font size.
         *
         * @param maxWidth The maximum width.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        inline unsigned int BlockLines(float maxWidth, const wchar_t *txt)
                const {
            return this->BlockLines(maxWidth, this->size, txt);
        }

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and the default font size.
         *
         * @param maxWidth The maximum width.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        inline unsigned int BlockLines(float maxWidth,
                const vislib::StringW& txt) const {
            return this->BlockLines(maxWidth, this->size, txt.PeekBuffer());
        }

        /**
         * Calculates the height of a text block in number of lines, when
         * drawn with the rectangle-based versions of 'DrawString' with the
         * specified maximum width and the default font size.
         *
         * @param maxWidth The maximum width.
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The height of the text block in number of lines.
         */
        inline unsigned int BlockLines(float maxWidth, float size,
                const vislib::StringW& txt) const {
            return this->BlockLines(maxWidth, size, txt.PeekBuffer());
        }

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
            const char *txt) const = 0;

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
            const wchar_t *txt) const = 0;

        /**
         * Deinitialises the object if required. Derived classes must call
         * 'Deinitialise' in EACH dtor.
         */
        void Deinitialise(void);

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, const char *txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, this->size, this->flipY, txt, align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, const vislib::StringA& txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, this->size, this->flipY, txt.PeekBuffer(),
                align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h,
                const char *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, this->size, this->flipY, txt, align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h,
                const vislib::StringA& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, w, h, this->size, this->flipY,
                txt.PeekBuffer(), align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, float size,
                const vislib::StringA &txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, size, this->flipY, txt.PeekBuffer(), align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, float size, const char *txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, size, this->flipY, txt, align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, float size,
                const vislib::StringA& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, w, h, size, this->flipY, txt.PeekBuffer(),
                align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, float size,
                const char *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, size, this->flipY, txt, align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, bool flipY, const char *txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, this->size, flipY, txt, align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, bool flipY,
                const vislib::StringA& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, this->size, flipY, txt.PeekBuffer(), align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, bool flipY,
                const char *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, this->size, flipY, txt, align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, bool flipY,
                const vislib::StringA& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, w, h, this->size, flipY, txt, align);
        }

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
        inline void DrawString(float x, float y, float size, bool flipY,
                const vislib::StringA &txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, size, flipY, txt.PeekBuffer(), align);
        }

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
            const char *txt, Alignment align = ALIGN_LEFT_TOP) const = 0;

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
        inline void DrawString(float x, float y, float w, float h, float size,
                bool flipY, const vislib::StringA& txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, size, flipY, txt.PeekBuffer(), align);
        }

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
            const = 0;

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, const wchar_t *txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, this->size, this->flipY, txt, align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, const vislib::StringW& txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, this->size, this->flipY, txt.PeekBuffer(),
                align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h,
                const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, this->size, this->flipY, txt, align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h,
                const vislib::StringW& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, w, h, this->size, this->flipY,
                txt.PeekBuffer(), align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, float size,
                const vislib::StringW &txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, size, this->flipY, txt.PeekBuffer(), align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, float size,
                const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, size, this->flipY, txt, align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, float size,
                const vislib::StringW& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, w, h, size, this->flipY, txt.PeekBuffer(),
                align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param size The size to use.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, float size,
                const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, size, this->flipY, txt, align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, bool flipY,
                const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, this->size, flipY, txt, align);
        }

        /**
         * Draws a text at the specified position.
         *
         * @param x The x coordinate of the position.
         * @param y The y coordinate of the position.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text.
         */
        inline void DrawString(float x, float y, bool flipY,
                const vislib::StringW& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, this->size, flipY, txt.PeekBuffer(), align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, bool flipY,
                const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, this->size, flipY, txt, align);
        }

        /**
         * Draws a text into a specified rectangular area, and performs
         * soft-breaks if necessary.
         *
         * @param x The left coordinate of the rectangle.
         * @param y The upper coordinate of the rectangle.
         * @param w The width of the rectangle.
         * @param h The height of the rectangle.
         * @param flipY The flag controlling the direction of the y-axis.
         * @param txt The zero-terminated string to draw.
         * @param align The alignment of the text inside the area.
         */
        inline void DrawString(float x, float y, float w, float h, bool flipY,
                const vislib::StringW& txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, w, h, this->size, flipY, txt, align);
        }

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
        inline void DrawString(float x, float y, float size, bool flipY,
                const vislib::StringW &txt, Alignment align = ALIGN_LEFT_TOP)
                const {
            this->DrawString(x, y, size, flipY, txt.PeekBuffer(), align);
        }

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
            const wchar_t *txt, Alignment align = ALIGN_LEFT_TOP) const = 0;

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
        inline void DrawString(float x, float y, float w, float h, float size,
                bool flipY, const vislib::StringW& txt,
                Alignment align = ALIGN_LEFT_TOP) const {
            this->DrawString(x, y, w, h, size, flipY, txt.PeekBuffer(), align);
        }

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
            const = 0;


        /**
         * Gets the default size of the font. The size is specified in logical
         * units used to measure the text in object space.
         *
         * @return The default size of the font.
         */
        inline float GetSize(void) const {
            return this->size;
        }

        /**
         * Initialises the object if required. You must call 'Initialise'
         * before the object can be used.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool Initialise(void);

        /**
         * Answer the flag 'flipY'. If 'flipY' is true, the direction of the
         * y-axis is upward, otherwise the direction is downward.
         *
         * @return The flag 'flipY'
         */
        inline bool IsFlipY(void) const {
            return this->flipY;
        }

        /**
         * Answers whether the font object is initialised. If the method
         * returns 'false' the font object is not initialised. You must call
         * 'Initialise' (and the method must return 'true') in order to
         * initialise the font object. You must not use an uninitialised
         * object.
         */          
        inline bool IsInitialised(void) const {
            return this->initialised;
        }

        /**
         * Answers the height of a single line with the default font size in
         * logical units. The line height is always positive.
         *
         * @return The height of a single line in logical units.
         */
        inline float LineHeight(void) const {
            return this->LineHeight(this->size);
        }

        /**
         * Answers the height of a single line in logical units. This default
         * implementation returns 'size', since this is the value representing
         * a line height. The line height is always positive.
         *
         * @param size The font size to use.
         *
         * @return The height of a single line in logical units.
         */
        virtual float LineHeight(float size) const;

        /**
         * Answers the width of the line 'txt' with the default font size in
         * logical units. The line width is always positive.
         *
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        inline float LineWidth(const char *txt) const {
            return this->LineWidth(this->size, txt);
        }

        /**
         * Answers the width of the line 'txt' with the default font size in
         * logical units. The line width is always positive.
         *
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        inline float LineWidth(const vislib::StringA& txt) const {
            return this->LineWidth(this->size, txt.PeekBuffer());
        }

        /**
         * Answers the width of the line 'txt' in logical units. The line
         * width is always positive.
         *
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        virtual float LineWidth(float size, const char *txt) const = 0;

        /**
         * Answers the width of the line 'txt' in logical units. The line
         * width is always positive.
         *
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        inline float LineWidth(float size, const vislib::StringA& txt) const {
            return this->LineWidth(size, txt.PeekBuffer());
        }

        /**
         * Answers the width of the line 'txt' with the default font size in
         * logical units. The line width is always positive.
         *
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        inline float LineWidth(const wchar_t *txt) const {
            return this->LineWidth(this->size, txt);
        }

        /**
         * Answers the width of the line 'txt' with the default font size in
         * logical units. The line width is always positive.
         *
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        inline float LineWidth(const vislib::StringW& txt) const {
            return this->LineWidth(this->size, txt.PeekBuffer());
        }

        /**
         * Answers the width of the line 'txt' in logical units. The line
         * width is always positive.
         *
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        virtual float LineWidth(float size, const wchar_t *txt) const = 0;

        /**
         * Answers the width of the line 'txt' in logical units. The line
         * width is always positive.
         *
         * @param size The font size to use.
         * @param txt The text to measure.
         *
         * @return The width in the text in logical units.
         */
        inline float LineWidth(float size, const vislib::StringW& txt) const {
            return this->LineWidth(size, txt.PeekBuffer());
        }

        /**
         * Sets the flag 'flipY'. If the flag is set, the direction of the
         * y-axis is upward, otherwise it is downward.
         *
         * @param flipY The new value for the 'flipY' flag.
         */
        virtual void SetFlipY(bool flipY);

        /**
         * Sets the default size of the font. The size is specified in logical
         * units used to measure the text in object space.
         *
         * @param size The new default size of the font.
         *
         * @throw IllegalParamException if size is less than zero.
         */
        virtual void SetSize(float size);

    protected:

        /** Ctor. */
        AbstractFont(void);

        /**
         * Initialises the object. You must not call this method directly.
         * Instead call 'Initialise'. You must call 'Initialise' before the
         * object can be used.
         *
         * @return 'true' on success, 'false' on failure.
         */
        virtual bool initialise(void) = 0;

        /**
         * Deinitialises the object. You must not call this method directly.
         * Instead call 'Deinitialise'. Derived classes must call
         * 'Deinitialise' in EACH dtor.
         */
        virtual void deinitialise(void) = 0;

    private:

        /** flag whether or not this font is initialized. */
        bool initialised;

        /** The default size of the font */
        float size;

        /** flag whether to flip the y-axis */
        bool flipY;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTFONT_H_INCLUDED */
