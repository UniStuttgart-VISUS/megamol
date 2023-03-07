/*
 * QuartzParticleGridDataCall.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/math/Cuboid.h"


namespace megamol::demos_gl {

/**
 * Call transporting gridded quartz crystal particle data
 */
class ParticleGridDataCall : public megamol::core::AbstractGetData3DCall {
public:
    /**
     * Nested class holding the data of one particle list
     */
    class List {
    public:
        /** Ctor */
        List() : cnt(0), data(NULL), type(0) {
            // intentionally empty
        }

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        List(const List& src) : cnt(src.cnt), data(src.data), type(src.type) {
            // intentionally empty
        }

        /** Dtor */
        ~List() {
            this->cnt = 0;
            this->data = NULL; // DO NOT DELETE
        }

        /**
         * Answer the number of particles in this list
         *
         * @return The number of particles in this list
         */
        inline unsigned int Count() const {
            return this->cnt;
        }

        /**
         * Answer the data pointer of this list. The data is interleaved
         * layouted using eight floats per particle (x y z r qx qy qz qw).
         *
         * @return The pointer to the particle data
         */
        inline const float* Data() const {
            return this->data;
        }

        /**
         * Answer the crystalite type index of the particles
         *
         * @return The crystalite type index of the particles
         */
        inline unsigned int Type() const {
            return this->type;
        }

        /**
         * Sets the data of this list
         *
         * @param cnt The number of particles
         * @param data The particle data
         * @param type The crytslite type index of the particles
         */
        inline void Set(unsigned int cnt, float* data, unsigned int type) {
            this->cnt = cnt;
            this->data = data;
            this->type = type;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        List& operator=(const List& rhs) {
            this->cnt = rhs.cnt;
            this->data = rhs.data;
            this->type = rhs.type;
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        bool operator==(const List& rhs) {
            return (this->cnt == rhs.cnt) && (this->data == rhs.data) && (this->type == rhs.type);
        }

    private:
        /** The number of particles */
        unsigned int cnt;

        /** The particle data (x y z r qx qy qz qw) */
        float* data;

        /** The crystalite type (index) */
        unsigned int type;
    };

    /**
     * Nested class holding the data of one grid cell
     */
    class Cell {
    public:
        /** Ctor */
        Cell()
                : bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
                , cbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
                , cnt(0)
                , lists(NULL) {
            // intentionally empty
        }

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        Cell(const Cell& src) : bbox(src.bbox), cbox(src.cbox), cnt(src.cnt), lists(src.lists) {
            // intentionally empty
        }

        /** Dtor */
        ~Cell() {
            this->cnt = 0;
            this->lists = NULL; // DO NOT DELETE
        }

        /**
         * Answer the bounding box of the cell
         *
         * @return The bounding box of the cell
         */
        inline const vislib::math::Cuboid<float>& BoundingBox() const {
            return this->bbox;
        }

        /**
         * Answer the clipping box of the cell
         *
         * @return The clipping box of the cell
         */
        inline const vislib::math::Cuboid<float>& ClipBox() const {
            return this->cbox;
        }

        /**
         * Answer the number of lists in this cell
         *
         * @return The number of lists in this cell
         */
        inline unsigned int Count() const {
            return this->cnt;
        }

        /**
         * Answer the lists in this cell
         *
         * @return The lists in this cell
         */
        inline const List* Lists() const {
            return this->lists;
        }

        /**
         * Sets the content of this cell.
         *
         * @param bbox The bounding box
         * @param cbox The clipping box
         * @param cnt The number of lists
         * @param lists The lists
         */
        inline void Set(const vislib::math::Cuboid<float>& bbox, const vislib::math::Cuboid<float>& cbox,
            unsigned int cnt, List* lists) {
            this->bbox = bbox;
            this->cbox = cbox;
            this->cnt = cnt;
            this->lists = lists;
        }

        /**
         * Sets the content of this cell.
         *
         * @param bbox The bounding box
         */
        inline void SetBBox(const vislib::math::Cuboid<float>& bbox) {
            this->bbox = bbox;
        }

        /**
         * Sets the clipping box relative to the bounding box
         *
         * @param border The border for clipping
         */
        inline void SetCBoxRelative(float border) {
            this->cbox = this->bbox;
            this->cbox.Grow(border);
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        Cell& operator=(const Cell& rhs) {
            this->bbox = rhs.bbox;
            this->cbox = rhs.cbox;
            this->cnt = rhs.cnt;
            this->lists = rhs.lists;
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return true if this and rhs are equal
         */
        bool operator==(const Cell& rhs) {
            return (this->bbox == rhs.bbox) && (this->cbox == rhs.cbox) && (this->cnt == rhs.cnt) &&
                   (this->lists == rhs.lists);
        }

    private:
        /** The bounding box of the cell */
        vislib::math::Cuboid<float> bbox;

        /** The clipping box of the cell */
        vislib::math::Cuboid<float> cbox;

        /** The number of lists */
        unsigned int cnt;

        /** The particle lists */
        List* lists;
    };

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /** Index of the 'GetExtent' function */
    static const unsigned int CallForGetExtent;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "QuartzParticleGridDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call transporting gridded quartz crystal particle data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** Ctor */
    ParticleGridDataCall();

    /** Dtor. */
    ~ParticleGridDataCall() override;

    /**
     * Answer the cells
     *
     * @return The cells
     */
    inline const Cell* Cells() const {
        return this->cells;
    }

    /**
     * Sets the data for the call
     *
     * @param sx The number of grid cells in x direction
     * @param sy The number of grid cells in y direction
     * @param sz The number of grid cells in z direction
     * @param cells Pointer to the grid cells
     */
    inline void Set(unsigned int sx, unsigned int sy, unsigned int sz, Cell* cells) {
        this->sx = sx;
        this->sy = sy;
        this->sz = sz;
        this->cells = cells;
    }

    /**
     * Answer the number of grid cells
     *
     * @return The number of grid cells
     */
    inline unsigned int Size() const {
        return this->sx * this->sy * this->sz;
    }

    /**
     * Answer the number of grid cells in x direction
     *
     * @return The number of grid cells in x direction
     */
    inline unsigned int SizeX() const {
        return this->sx;
    }

    /**
     * Answer the number of grid cells in y direction
     *
     * @return The number of grid cells in y direction
     */
    inline unsigned int SizeY() const {
        return this->sy;
    }

    /**
     * Answer the number of grid cells in z direction
     *
     * @return The number of grid cells in z direction
     */
    inline unsigned int SizeZ() const {
        return this->sz;
    }

private:
    /** The linearized pointer to the grid (x + y * sx + z * sx * sy) */
    Cell* cells;

    /** The size of the grid */
    unsigned int sx, sy, sz;
};

} // namespace megamol::demos_gl
