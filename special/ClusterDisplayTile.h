/*
 * ClusterDisplayTile.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERDISPLAYTILE_H_INCLUDED
#define MEGAMOLCORE_CLUSTERDISPLAYTILE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace megamol {
namespace core {
namespace special {


    /**
     * Utility class managing a display tile
     */
    class ClusterDisplayTile {
    public:

        /**
         * Ctor.
         */
        ClusterDisplayTile(void);

        /**
         * Copy ctor.
         *
         * @param src The object to clone from.
         */
        ClusterDisplayTile(const ClusterDisplayTile& src);

        /**
         * Dtor.
         */
        ~ClusterDisplayTile(void);

        /**
         * Answer the height of the tile.
         *
         * @return The height of the tile
         */
        inline float Height(void) const {
            return this->h;
        }

        /**
         * Answer the plane identifier of the tile.
         *
         * @return The plane identifier of the tile
         */
        inline unsigned int Plane(void) const {
            return this->plane;
        }

        /**
         * Sets the height of the tile.
         *
         * @param h The new height for the tile
         */
        inline void SetHeight(float h) {
            this->h = h;
        }

        /**
         * Sets the plane identifier of the tile.
         *
         * @param p The new plane identifier for the tile
         */
        inline void SetPlane(unsigned int p) {
            this->plane = p;
        }

        /**
         * Sets the width of the tile.
         *
         * @param w The new width for the tile
         */
        inline void SetWidth(float w) {
            this->w = w;
        }

        /**
         * Sets the left coordinate of the tile.
         *
         * @param x The new left coordinate for the tile
         */
        inline void SetX(float x) {
            this->x = x;
        }

        /**
         * Sets the bottom coordinate of the tile.
         *
         * @param y The new bottom coordinate for the tile
         */
        inline void SetY(float y) {
            this->y = y;
        }

        /**
         * Answer the width of the tile.
         *
         * @return The width of the tile
         */
        inline float Width(void) const {
            return this->w;
        }

        /**
         * Answer the left coordinate of the tile.
         *
         * @return The left coordinate of the tile
         */
        inline float X(void) const {
            return this->x;
        }

        /**
         * Answer the bottom coordinate of the tile.
         *
         * @return The bottom coordinate of the tile
         */
        inline float Y(void) const {
            return this->y;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand
         *
         * @return A refnerence to 'this'
         */
        ClusterDisplayTile& operator=(const ClusterDisplayTile& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'rhs' and 'this' are equal.
         */
        bool operator==(const ClusterDisplayTile& rhs) const;

    private:

        /** The height of the tile */
        float h;

        /** The plane id */
        unsigned int plane;

        /** The width of the tile */
        float w;

        /** The left coordinate of the tile */
        float x;

        /** The bottom coordinate of the tile */
        float y;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERDISPLAYTILE_H_INCLUDED */
