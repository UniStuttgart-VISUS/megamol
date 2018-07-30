/*
 * LoopBuffer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ArxelBuffer.h"
#include "vislib/Array.h"
#include "vislib/math/Point.h"
#include "vislib/math/Rectangle.h"


namespace megamol {
namespace demos {


    /**
     * Class holding the iso-line loops extracted from a single arxel slice
     */
    class LoopBuffer {
    public:

        /**
         * Class holding all information about a single loop
         */
        class Loop {
        public:

            /** Ctor */
            Loop(void);

            /**
             * Copy ctor
             *
             * @param src The object to clone
             */
            Loop(const Loop& src);

            /** Dtor */
            ~Loop(void);

            /**
             * Adds an edge-vertex pair
             *
             * @param vertex The new vertex to be added to the end of the loop
             * @param edge The value of the edge starting at 'vertex'
             */
            void AddVertex(const vislib::math::Point<int, 2>& vertex,
                const ArxelBuffer::ArxelType& edge);

            /**
             * Answer the area of the loop as number of arxels inside the loop
             * excluding any inner loops.
             *
             * @return The area of the loop
             */
            inline UINT64 Area(void) const {
                return this->area;
            }

            /**
             * Answer the bounding box of the loop
             *
             * @return The bounding box of the loop
             */
            inline const vislib::math::Rectangle<int>& BoundingBox(void) const {
                return this->bbox;
            }

            /** Deletes all vertex data */
            void ClearVertices(void);

            /**
             * Answer whether this loop contains the point p
             *
             * @return whether this loop contains p
             */
            bool Contains(const vislib::math::Point<int, 2>& p) const;

            /**
             * Answer the loop enclosing this loop
             *
             * @return The enclosing loop or NULL
             */
            inline const Loop * EnclosingLoop(void) const {
                return this->enclosingLoop;
            }

            /**
             * Answer the length of the loop in number of edges
             *
             * @return The length of the loop
             */
            inline SIZE_T Length(void) const {
                return this->vertices.Count();
            }

            /**
             * Sets the area of the loop as number of arxels inside the loop
             * excluding any inner loops.
             *
             * @param area The new area value
             */
            inline void SetArea(UINT64 area) {
                this->area = area;
            }

            /**
             * Sets 'loop' as loop enclosing this loop
             *
             * @param loop The loop enclosing this loop
             */
            inline void SetEnclosingLoop(const Loop *loop) {
                this->enclosingLoop = loop;
            }

            /**
             * Sets the white Arxels to count.
             *
             * @param count the number of white Arxels
             */
            inline void SetWhiteArxels(const UINT64 count) {
                this->whiteArxels = count;
            }

            /**
             * Answer the idx-th vertex of the loop
             *
             * @param idx The zero-based index of the vertex to return
             *
             * @return The requested vertex
             */
            inline const vislib::math::Point<int, 2>& Vertex(SIZE_T idx) const {
                return this->vertices[idx];
            }

            /**
             * Answer the vertices of the loop
             *
             * @return The vertices of the loop
             */
            inline const vislib::Array<vislib::math::Point<int, 2> >& Vertices(void) const {
                return this->vertices;
            }

            /**
             * Answer the number of isolated, non-black Arxels inside.
             *
             * @return the number of isolated, non-black Arxels
             */
            inline const UINT64 WhiteArxels(void) const {
                return this->whiteArxels;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True if this and rhs are equal
             */
            bool operator==(const Loop& rhs) const;

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return A reference to this
             */
            Loop& operator=(const Loop& rhs);

        private:

            /**
             * number of arxels inside the loop excluding any inner loops.
             */
            UINT64 area;

            /** The bounding box of the loop */
            vislib::math::Rectangle<int> bbox;

            /**
             * The values of the edges of the loop.
             * edgeValue[0] is the value of the edge connecting vertices[0]
             * and vertices[1]
             */
            vislib::Array<ArxelBuffer::ArxelType> edgeVals;

            /** The enclosing loop */
            const Loop *enclosingLoop;

            /** The sorted loop vertices */
            vislib::Array<vislib::math::Point<int, 2> > vertices;

            /** The ignored, isolated, non-black Arxels inside this loop */
            UINT64 whiteArxels;
        };

        /** Ctor */
        LoopBuffer(void);

        /** Dtor */
        ~LoopBuffer(void);

        /**
         * Answer the number of isolated black Arxels in the loop. This can be negative
         * when a non-black arxel could not be matched to any one loop!
         *
         * @return the number of isolated black Arxels
         */
        inline INT64 BlackArxels(void) const {
            return this->blackArxels;
        }

        /**
         * Answer the bounds of the LoopBuffer
         *
         * @return the bounds of the LoopBuffer
         */
        inline const vislib::math::Dimension<INT64, 2>& Bounds(void) const {
            return this->bounds;
        }

        /**
         * Access the bounds of the LoopBuffer
         *
         * @return the bounds of the LoopBuffer
         */
        inline vislib::math::Dimension<INT64, 2>& Bounds(void) {
            return this->bounds;
        }

        /** Clears all data */
        void Clear(void);

        /**
         * Gets the list of loops
         *
         * @return The list of loops
         */
        inline vislib::Array<Loop>& Loops(void) {
            return this->loops;
        }

        /**
         * Gets the list of loops
         *
         * @return The list of loops
         */
        inline const vislib::Array<Loop>& Loops(void) const {
            return this->loops;
        }

        /**
         * Creates a new loop object stored in this object
         *
         * @return The new loop object
         */
        Loop& NewLoop(void);

        /**
         * The loop returned by 'NewLoop' is now complete, whatever that means
         */
        void NewLoopComplete(void);

        /**
         * Set the number of isolated black Arxels in the loop.
         *
         * @param count the number of isolated black Arxels
         */
        inline void SetBlackArxels(const INT64 count) {
            this->blackArxels = count;
        }

    private:

        /** the isolated black pixels */
        INT64 blackArxels;

        /** the dimension of this LoopBuffer */
        vislib::math::Dimension<INT64, 2> bounds;

        /** The loops */
        vislib::Array<Loop> loops;

    };

} /* end namespace demos */
} /* end namespace megamol */

