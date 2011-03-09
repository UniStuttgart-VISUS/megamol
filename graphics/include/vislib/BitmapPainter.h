/*
 * BitmapPainter.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BITMAPPAINTER_H_INCLUDED
#define VISLIB_BITMAPPAINTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Array.h"
#include "vislib/BitmapImage.h"
#include "vislib/forceinline.h"
#include "vislib/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/Point.h"
#include "vislib/ShallowPoint.h"
#include "vislib/types.h"
#include <climits>


namespace vislib {
namespace graphics {


    /**
     * Utility class to draw into a bitmap image object
     */
    class BitmapPainter {
    public:

        /**
         * Ctor.
         *
         * @param img Pointer to the image to be used by the painter. The
         *            image objects memory will not be released by the painter
         *            object if the image pointer is changed or the painter
         *            object is destroied. The caller is responsible to
         *            handling the memory.
         */
        BitmapPainter(BitmapImage *img = NULL);

        /** Dtor. */
        ~BitmapPainter(void);

        /**
         * Clears the image with the set colour
         */
        void Clear(void);

        /**
         * Draws a straight line between two points. Both points will be set!
         *
         * @param x1 The x coordinate of the starting point
         * @param y1 The y coordinate of the starting point
         * @param x2 The x coordinate of the ending point
         * @param y2 The y coordinate of the ending point
         */
        void DrawLine(int x1, int y1, int x2, int y2);

        /**
         * Draws a straight line between two points. Both points will be set!
         *
         * @param x1 The x coordinate of the starting point
         * @param y1 The y coordinate of the starting point
         * @param x2 The x coordinate of the ending point
         * @param y2 The y coordinate of the ending point
         */
        template<class Tp> inline void DrawLine(const Tp& x1, const Tp& y1,
                const Tp& x2, const Tp& y2) {
            this->DrawLine(static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2), static_cast<int>(y2));
        }

        /**
         * Fills a pixels inside the specified polygon
         *
         * @param point A pointer to 'count' point objects forming the polygon
         * @param count The number of point objects forming the polygon
         */
        template<class Sp>
        void FillPolygon(const math::AbstractPoint<int, 2, Sp> *points,
            SIZE_T count);

        /**
         * Fills a pixels inside the specified polygon
         *
         * @param point A pointer to 'count' point objects forming the polygon
         * @param count The number of point objects forming the polygon
         */
        inline void FillPolygon(const math::Point<int, 2> *points,
            SIZE_T count) {
            this->FillPolygon(
                reinterpret_cast<const math::AbstractPoint<int, 2, int[2]>*>(
                    points), count);
        }

        /**
         * Fills a pixels inside the specified polygon
         *
         * @param point A pointer to 'count' point objects forming the polygon
         * @param count The number of point objects forming the polygon
         */
        inline void FillPolygon(const math::ShallowPoint<int, 2> *points,
            SIZE_T count) {
            this->FillPolygon(
                reinterpret_cast<const math::AbstractPoint<int, 2, int*>*>(
                    points), count);
        }

        /**
         * Fills a pixels inside the specified polygon
         *
         * @param point A pointer to 'count' point objects forming the polygon
         * @param count The number of point objects forming the polygon
         */
        template<class Tp>
        inline void FillPolygon(const Tp *points, SIZE_T count) {
            math::Point<int, 2> *pts = new math::Point<int, 2>[count];
            for (SIZE_T i = 0; i < count; i++) {
                pts[i] = points[i];
            }
            this->FillPolygon(pts, count);
            delete[] pts;
        }

        /**
         * Fills a pixels inside the specified polygon
         *
         * @param point A pointer to 'count' point objects forming the polygon
         * @param count The number of point objects forming the polygon
         */
        template<class Tp>
        inline void FillPolygon(const Array<Tp>& points) {
            this->FillPolygon(points.PeekElements(), points.Count());
        }

        /**
         * Accesses the pointer to the image to be used by the painter. Use
         * this to set the pointer to an image object to be used before
         * calling any drawing method. The image objects memory will not be
         * released by the painter object if the image pointer is changed or
         * the painter object is destroied. The caller is responsible to
         * handling the memory.
         *
         * @return Reference to the image object pointer of the painter.
         */
        inline BitmapImage* &Image(void) {
            return this->img;
        }

        /**
         * Gets the pointer to the image object that is used by the painter.
         *
         * @return The image object pointer that is used by the painter.
         */
        inline const BitmapImage *Image(void) const {
            return this->img;
        }

        /**
         * Sets the colour to be used. The colour value will be applied to all
         * colour channels.
         *
         * @param c The colour value
         */
        template<class Tp>
        inline void SetColour(Tp c) {
            this->col.SetCount(1);
            this->setColourEntry(0, UINT_MAX, BitmapImage::CHANNEL_UNDEF, c);
            this->clearColourCache();
        }

        /**
         * Sets the colour to be used for RGB colour channels. Other colour
         * channels will not be changed
         *
         * @param r The value for the red colour channels
         * @param g The value for the green colour channels
         * @param b The value for the blue colour channels
         */
        template<class Tp1, class Tp2, class Tp3>
        inline void SetColour(Tp1 r, Tp2 g, Tp3 b) {
            this->col.SetCount(3);
            this->setColourEntry(0, UINT_MAX, BitmapImage::CHANNEL_RED, r);
            this->setColourEntry(1, UINT_MAX, BitmapImage::CHANNEL_GREEN, g);
            this->setColourEntry(2, UINT_MAX, BitmapImage::CHANNEL_BLUE, b);
            this->clearColourCache();
        }

        /**
         * Sets the specified pixel to the current colour. Pixels outside the
         * image will be silently ignored.
         *
         * @param x The x coordinate
         * @param y The y coordinate
         */
        inline void SetPixel(int x, int y) {
            this->preDraw();
            if ((x >= 0) && (y >= 0)
                    && (x < static_cast<int>(this->img->Width()))
                    && (y < static_cast<int>(this->img->Height()))) {
                this->setPixel(this->img->PeekDataAs<unsigned char>()
                    + (x + y * this->img->Width())
                    * this->img->BytesPerPixel());
            }
        }

        /**
         * Sets the specified pixel to the current colour. Pixels outside the
         * image will be silently ignored.
         *
         * @param x The x coordinate
         * @param y The y coordinate
         */
        inline void SetPixel(unsigned int x, unsigned int y) {
            this->preDraw();
            if ((x < this->img->Width()) && (y < this->img->Height())) {
                this->setPixel(this->img->PeekDataAs<unsigned char>()
                    + (x + y * this->img->Width())
                    * this->img->BytesPerPixel());
            }
        }

        /**
         * Sets the specified pixel to the current colour. Pixels outside the
         * image will be silently ignored.
         *
         * @param x The x coordinate
         * @param y The y coordinate
         */
        template<class Tp>
        inline void SetPixel(const Tp& x, const Tp& y) {
            this->SetPixel(static_cast<int>(x), static_cast<int>(y));
        }

    private:

        /** Helper struct specifying a colour value for one channel */
        typedef struct _channelcolour_t {

            /** The channel index or UINT_MAX */
            unsigned int idx;

            /** The channel label or CHANNEL_UNDEF */
            BitmapImage::ChannelLabel label;

            /** The colour type */
            BitmapImage::ChannelType type;

            /** The colour value */
            union _value_t {

                /** The colour value as byte */
                unsigned char asByte;

                /** The colour value as word */
                unsigned short asWord;

                /** The colour value as float */
                float asFloat;

            } value;

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True iff this and rhs are equal
             */
            bool operator==(const struct _channelcolour_t& rhs) const {
                return (this->idx == rhs.idx)
                    && (this->label == rhs.label)
                    && (this->type == rhs.type)
                    && (this->value.asFloat == rhs.value.asFloat);
                        // seems evil, but is ok.
            }

        } ChannelColour;

        /**
         * helper struct for housekeeping and sorting in 'FillPolygon'
         */
        typedef struct _tableEdge_t {

            /**
             * Compare functions used to sort the active edge table
             *
             * @param one The left hand side operand
             * @param other The right hand side operand
             *
             * @return The compare result
             */
            static int AETComparator(const struct _tableEdge_t &one,
                    const struct _tableEdge_t &other) {
                return vislib::math::Compare(one.x, other.x);
            }

            /**
             * The y value on which to remove this edge from the active edge
             * table
             */
            int end;

            /**
             * The interpolated x value of this edge for the current y value
             */
            float x;

            /** The inverted slope to interpolate the x values */
            float invSlope;

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True if this and rhs are equal
             */
            bool operator ==(const struct _tableEdge_t &rhs) const {
                return (this->end == rhs.end) && (this->x == rhs.x)
                    && (this->invSlope == rhs.invSlope);
            }

        } tableEdge;

        /**
         * Sets one colour entry
         *
         * @param i The zero-based index of the colour entry to be set
         * @param idx The colour channel index
         * @param label The colour channel label
         * @param v The colour value
         */
        VISLIB_FORCEINLINE void setColourEntry(SIZE_T i, unsigned int idx,
                BitmapImage::ChannelLabel label, unsigned char v) {
            this->col[i].idx = idx;
            this->col[i].label = label;
            this->col[i].type = BitmapImage::CHANNELTYPE_BYTE;
            this->col[i].value.asByte = v;
        }

        /**
         * Sets one colour entry
         *
         * @param i The zero-based index of the colour entry to be set
         * @param idx The colour channel index
         * @param label The colour channel label
         * @param v The colour value
         */
        VISLIB_FORCEINLINE void setColourEntry(SIZE_T i, unsigned int idx,
                BitmapImage::ChannelLabel label, unsigned short v) {
            this->col[i].idx = idx;
            this->col[i].label = label;
            this->col[i].type = BitmapImage::CHANNELTYPE_WORD;
            this->col[i].value.asWord = v;
        }

        /**
         * Sets one colour entry
         *
         * @param i The zero-based index of the colour entry to be set
         * @param idx The colour channel index
         * @param label The colour channel label
         * @param v The colour value
         */
        VISLIB_FORCEINLINE void setColourEntry(SIZE_T i, unsigned int idx,
                BitmapImage::ChannelLabel label, float v) {
            this->col[i].idx = idx;
            this->col[i].label = label;
            this->col[i].type = BitmapImage::CHANNELTYPE_FLOAT;
            this->col[i].value.asFloat = v;
        }

        /**
         * Clears all cached colour information
         */
        VISLIB_FORCEINLINE void clearColourCache(void) {
            this->colSize = 0;
            ARY_SAFE_DELETE(this->colBits);
            ARY_SAFE_DELETE(this->colMask);
        }

        /**
         * Performs pre draw checks and throws exceptions on failures
         */
#ifdef _WIN32
        inline
#endif /* _WIN32 */
        void preDraw(void);

        /**
         * Tries to set one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param idx The index of the colour channel
         * @param label The label of the colour channel
         *
         * @return True if the value was set
         */
        template<class Tp>
        inline bool setColourCacheValue(Tp* dst, unsigned int idx,
            BitmapImage::ChannelLabel label);

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(unsigned char* dst,
                const unsigned char& src) {
            *dst = src;
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(unsigned char* dst,
                const unsigned short& src) {
            unsigned short w = src / 256;
            if (w > 255) w = 255;
            *dst = static_cast<unsigned char>(w);
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(unsigned char* dst,
                const float& src) {
            float f = src * 255.0f;
            if (f < 0.0f) f = 0.0f;
            if (f > 255.0f) f = 255.0f;
            *dst = static_cast<unsigned char>(f);
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(unsigned short* dst,
                const unsigned char& src) {
            *dst = static_cast<unsigned short>(src) * 256
                + static_cast<unsigned short>(src);
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(unsigned short* dst,
                const unsigned short& src) {
            *dst = src;
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(unsigned short* dst,
                const float& src) {
            float f = src * 65535.0f;
            if (f < 0.0f) f = 0.0f;
            if (f > 65535.0f) f = 65535.0f;
            *dst = static_cast<unsigned short>(f);
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(float* dst,
                const unsigned char& src) {
            *dst = static_cast<float>(src) / 255.0f;
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(float* dst,
                const unsigned short& src) {
            *dst = static_cast<float>(src) / 65535.0f;
        }

        /**
         * Sets one value of the colour cache
         *
         * @param dst Pointer to the memory where the value should be set
         * @param src The value to be set
         */
        inline void setColourCacheValue(float* dst, const float& src) {
            *dst = src;
        }

        /**
         * Sets the specified pixel to the current colour
         *
         * @param dst Pointer to the pixel
         */
#ifdef _WIN32
        VISLIB_FORCEINLINE
#endif /*_WIN32 */
        void setPixel(unsigned char *dst);

        /** The bitmap image used by the codec */
        BitmapImage *img;

        /** The colour */
        vislib::Array<ChannelColour> col;

        /** The size of the colour cache */
        unsigned int colSize;

        /** The colour cache bit values */
        unsigned char *colBits;

        /** The colour cache bit mask */
        unsigned char *colMask;

    };


    /*
     * BitmapPainter::FillPolygon
     */
    template<class Sp>
    void BitmapPainter::FillPolygon(
            const math::AbstractPoint<int, 2, Sp> *points, SIZE_T count) {
        math::Point<int, 2> p1;
        math::Point<int, 2> p2;
        math::Point<int, 2> p;
        int bottom = INT_MAX, top = -INT_MAX, y, x;
        //UINT64 pixelCount = 0;

        // find polygon y bounds
        for (SIZE_T i = 0; i < count; i++) {
            if (points[i].Y() < bottom) {
                bottom = points[i].Y();
            }
            if (points[i].Y() > top) {
                top = points[i].Y();
            }
        }

        // build a edge table
        Array<Array<tableEdge> > edgeTable;
        for (y = bottom; y <= top; y++) {
            Array<tableEdge> edgeLine;
            edgeLine.SetCapacityIncrement(4);
            for (SIZE_T i = 0; i < count; i++) {
                p = points[i];
                p2 = points[(i + 1) % count];
                if (p2.Y() < p.Y()) {
                    p1 = p2;
                    p2 = p;
                } else {
                    p1 = p;
                }
                // all go up; also, horizontal edges are useless
                if (p1.Y() == y && p2.Y() != y) {
                    tableEdge e;
                    e.x = static_cast<float>(p1.X());
                    e.end = p2.Y();
                    e.invSlope = static_cast<float>(p2.X() - p1.X())
                        / static_cast<float>(p2.Y() - p1.Y());
                    edgeLine.Add(e);
                }
            }
            edgeTable.Add(edgeLine);
        }

        // active edge table algorithm
        Array<tableEdge> activeEdges;
        y = 0;
        while (y < static_cast<int>(edgeTable.Count())) {
            for (x = 0; x < static_cast<int>(activeEdges.Count()); x++) {
                if (activeEdges[x].end == y + bottom) {
                    activeEdges.RemoveAt(x);
                    x = -1;
                }
            }
            for (x = 0; x < static_cast<int>(edgeTable[y].Count()); x++) {
                activeEdges.Add(edgeTable[y][x]);
            }
            activeEdges.Sort(tableEdge::AETComparator);
            for (x = 0; x < static_cast<int>(activeEdges.Count()); x += 2) {
                //pixelCount += static_cast<int>(activeEdges[x + 1].x) 
                //    - static_cast<int>(activeEdges[x].x) + 1;
                for (int i = static_cast<int>(activeEdges[x].x);
                        i <= activeEdges[x + 1].x; i++) {
                    this->SetPixel(i, y + bottom);
                }
            }

            y++;
            for (x = 0; x < static_cast<int>(activeEdges.Count()); x++) {
                activeEdges[x].x += activeEdges[x].invSlope;
            }
        }

        //return pixelCount;

    }


} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BITMAPPAINTER_H_INCLUDED */
