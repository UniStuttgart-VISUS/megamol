/*
 * ParticleGridDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLEGRIDDATACALL_H_INCLUDED
#define MEGAMOLCORE_PARTICLEGRIDDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {


/**
 * Call for gridded multi-stream particle data.
 */
class ParticleGridDataCall : public core::AbstractGetData3DCall {
public:

    /**
     * Class holding all information about one particle type
     *
     * This class currenty can only hold data for spheres and should
     * be extended to be able to handle data for arbitrary glyphs.
     * This also applies to interpolation of data.
     */
    class ParticleType {
    public:

        /** type alias for the vertex data type */
        typedef core::moldyn::MultiParticleDataCall::Particles::VertexDataType VertexDataType;

        /** type alias for the vertex data type */
        typedef core::moldyn::MultiParticleDataCall::Particles::ColourDataType ColourDataType;

        /**
         * Ctor
         */
        ParticleType(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        ParticleType(const ParticleType &src);

        /**
         * Dtor
         */
        ~ParticleType(void);

        /**
         * Answer the colour data type
         *
         * @return The colour data type
         */
        inline ColourDataType GetColourDataType(void) const {
            return this->colDataType;
        }

        /**
         * Answer the global colour
         *
         * @return The global colour as a pointer to four unsigned bytes
         *         storing the RGBA colour components
         */
        inline const unsigned char *GetGlobalColour(void) const {
            return this->col;
        }

        /**
         * Answer the global radius
         *
         * @return The global radius
         */
        inline float GetGlobalRadius(void) const {
            return this->radius;
        }

        /**
         * Answer the maximum colour index value to be mapped
         *
         * @return The maximum colour index value to be mapped
         */
        inline float GetMaxColourIndexValue(void) const {
            return this->maxColI;
        }

        /**
         * Answer the minimum colour index value to be mapped
         *
         * @return The minimum colour index value to be mapped
         */
        inline float GetMinColourIndexValue(void) const {
            return this->minColI;
        }

        /**
         * Answer the vertex data type
         *
         * @return The vertex data type
         */
        inline VertexDataType GetVertexDataType(void) const {
            return this->vertDataType;
        }

        /**
         * Sets the colour data type
         *
         * @param t The type of the colour data
         */
        void SetColourDataType(ColourDataType t) {
            this->colDataType = t;
        }

        /**
         * Sets the colour map index values
         *
         * @param minVal The minimum colour index value to be mapped
         * @param maxVal The maximum colour index value to be mapped
         */
        inline void SetColourMapIndexValues(float minVal, float maxVal) {
            this->maxColI = maxVal;
            this->minColI = minVal;
        }

        /**
         * Sets the global colour data
         *
         * @param r The red colour component
         * @param g The green colour component
         * @param b The blue colour component
         * @param a The opacity alpha
         */
        inline void SetGlobalColour(unsigned int r, unsigned int g,
            unsigned int b, unsigned int a = 255) {
            this->col[0] = r;
            this->col[1] = g;
            this->col[2] = b;
            this->col[3] = a;
        }

        /**
         * Sets the global colour data
         *
         * @param col Pointer to the RGBA byte colour array
         */
        inline void SetGlobalColour(const unsigned char *col) {
            this->col[0] = col[0];
            this->col[1] = col[0];
            this->col[2] = col[0];
            this->col[3] = col[0];
        }

        /**
         * Sets the global radius
         *
         * @param r The global radius
         */
        inline void SetGlobalRadius(float r) {
            this->radius = r;
        }

        /**
         * Sets the vertex data type
         *
         * @param t The type of the vertex data
         */
        inline void SetVertexDataType(VertexDataType t) {
            this->vertDataType = t;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        ParticleType &operator=(const ParticleType &rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const ParticleType &rhs) const;

    private:

        /** The global colour */
        unsigned char col[4];

        /** The colour data type */
        ColourDataType colDataType;

        /** The maximum colour index value to be mapped */
        float maxColI;

        /** The minimum colour index value to be mapped */
        float minColI;

        /** The global radius */
        float radius;

        /** The vertex data type */
        VertexDataType vertDataType;

    };

    /**
     * Class holding a list of particles of one type withing one grid cell
     */
    class Particles {
    public:

        /**
         * Ctor
         */
        Particles(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        Particles(const Particles &rhs);

        /**
         * Dtor
         */
        ~Particles(void);

        /**
         * Answer the colour data pointer
         *
         * @return The colour data pointer
         */
        inline const void *GetColourData(void) const {
            return this->colPtr;
        }

        /**
         * Answer the colour data stride
         *
         * @return The colour data stride
         */
        inline unsigned int GetColourDataStride(void) const {
            return this->colStride;
        }

        /**
         * Answer the number of stored objects
         *
         * @return The number of stored objects
         */
        inline SIZE_T GetCount(void) const {
            return this->count;
        }

        /**
         * Gets the maximum radius of the particles inside
         *
         * @return The maximum radius of the particles inside
         */
        inline float GetMaxRadius(void) const {
            return this->maxRad;
        }

        /**
         * Answer the vertex data pointer
         *
         * @return The vertex data pointer
         */
        inline const void *GetVertexData(void) const {
            return this->vertPtr;
        }

        /**
         * Answer the vertex data stride
         *
         * @return The vertex data stride
         */
        inline unsigned int GetVertexDataStride(void) const {
            return this->vertStride;
        }

        /**
         * Sets the colour data
         *
         * @param p The pointer to the colour data (must not be NULL if t
         *          is not 'COLDATA_NONE'
         * @param s The stride of the colour data
         */
        void SetColourData(const void *p, unsigned int s = 0) {
            this->colPtr = p;
            this->colStride = s;
        }

        /**
         * Sets the number of objects stored and resets all data pointers!
         *
         * @param cnt The number of stored objects
         */
        void SetCount(SIZE_T cnt) {
            this->colPtr = NULL;  // DO NOT DELETE
            this->vertPtr = NULL; // DO NOT DELETE

            this->count = cnt;
        }

        /**
         * Sets the maximum radius of the particles inside
         *
         * @param r The new maximum radius of the particles inside
         */
        inline void SetMaxRadius(float r) {
            this->maxRad = r;
        }

        /**
         * Sets the vertex data
         *
         * @param p The pointer to the vertex data (must not be NULL if t
         *          is not 'VERTDATA_NONE'
         * @param s The stride of the vertex data
         */
        void SetVertexData(const void *p, unsigned int s = 0) {
            this->vertPtr = p;
            this->vertStride = s;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        Particles &operator=(const Particles &rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const Particles &rhs) const;

    private:

        /** The colour data pointer */
        const void *colPtr;

        /** The colour data stride */
        unsigned int colStride;

        /** The number of particles in this list */
        SIZE_T count;

        /** The maximum radius of the particles inside */
        float maxRad;

        /** The vertex data pointer */
        const void *vertPtr;

        /** The vertex data stride */
        unsigned int vertStride;

    };

    /**
     * Class holding all particles of one grid cell
     */
    class GridCell {
    public:

        /**
         * Ctor
         */
        GridCell(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        GridCell(const GridCell &src);

        /**
         * Dtor
         */
        ~GridCell(void);

        /**
         * Accesses the list of particle lists
         *
         * @return A pointer to the list or particle lists
         */
        inline Particles *AccessParticleLists(void) {
            return this->particles;
        }

        /**
         * Accesses the list of particle lists
         *
         * @return A pointer to the list or particle lists
         */
        inline const Particles *AccessParticleLists(void) const {
            return this->particles;
        }

        /**
         * Allocates the list of particle lists
         *
         * @param cnt The number or particle lists to be allocated in the
         *            list of this cell
         */
        inline void AllocateParticleLists(unsigned int cnt) {
            delete[] this->particles;
            this->particles = new Particles[cnt];
        }

        /**
         * Answer the bounding box of this cell
         *
         * @return The bounding box of this cell
         */
        inline const vislib::math::Cuboid<float> &GetBoundingBox(void) const {
            return this->bbox;
        }

        /**
         * Sets the bounding box for this cell
         *
         * @param bbox The new bounding box for this cell
         */
        inline void SetBoundingBox(const vislib::math::Cuboid<float> &bbox) {
            this->bbox = bbox;
        }

        /**
         * Assignment operator.
         *
         * WARNING! This does not copy the particle lists!
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        GridCell &operator=(const GridCell &rhs);

        /**
         * Test for equality
         *
         * WARNING! This does not compare the particle lists!
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const GridCell &rhs) const;

    private:

        /** the particles of this cell */
        Particles *particles;

        /** The bouding box of this cell */
        vislib::math::Cuboid<float> bbox;

    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char *ClassName(void) {
        return "ParticleGridDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char *Description(void) {
        return "Call to get gridded multi-stream particle sphere data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char *FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** Ctor. */
    ParticleGridDataCall(void);

    /** Dtor. */
    virtual ~ParticleGridDataCall(void);

    /**
     * Calculates the index of a cell specified by its coordinates
     *
     * @param x The cells x coordinate
     * @param y The cells y coordinate
     * @param z The cells z coordinate
     *
     * @return The index of the specified cell
     */
    VISLIB_FORCEINLINE unsigned int CellIndex(unsigned int x,
        unsigned int y, unsigned int z) const {
        return x + (y + z * this->cntCellsY) * this->cntCellsX;
    }

    ///**
    // * Accesses the cells of the grid
    // *
    // * @return A pointer to the cells of the grid
    // */
    //inline GridCell * Cells(void) {
    //    return this->cells;
    //}

    /**
     * Accesses the cells of the grid
     *
     * @return A pointer to the cells of the grid
     */
    inline const GridCell *Cells(void) const {
        return this->cells;
    }

    /**
     * Answer the number of cells in x direction
     *
     * @return The number of cells in x direction
     */
    inline unsigned int CellsXCount(void) const {
        return this->cntCellsX;
    }

    /**
     * Answer the number of cells in y direction
     *
     * @return The number of cells in y direction
     */
    inline unsigned int CellsYCount(void) const {
        return this->cntCellsY;
    }

    /**
     * Answer the number of cells in z direction
     *
     * @return The number of cells in z direction
     */
    inline unsigned int CellsZCount(void) const {
        return this->cntCellsZ;
    }

    /**
     * Answers the overall number of cells
     *
     * @return The overall number of cells
     */
    inline unsigned int CellsCount(void) const {
        return this->cntCells;
    }

    /**
     * Sets references to grid data
     *
     * @param sizeX the size of the grid data in x direction
     * @param sizeY the size of the grid data in y direction
     * @param sizeZ the size of the grid data in z direction
     * @param grid Pointer to the grid data. The call object will not
     *             take ownership of this memory.
     */
    inline void SetGridDataRef(unsigned int sizeX, unsigned int sizeY,
        unsigned int sizeZ, const GridCell *grid) {
        this->cntCellsX = sizeX;
        this->cntCellsY = sizeY;
        this->cntCellsZ = sizeZ;
        this->cntCells = sizeX * sizeY * sizeZ;
        if (this->ownCellMem) {
            delete[] this->cells;
        }
        this->cells = grid;
        this->ownCellMem = false;
    }

    /**
     * Sets reference to type data
     *
     * @param cnt The number of types
     * @param types Pointer to the type data. The call object will not
     *              take ownership of this memory.
     */
    inline void SetTypeDataRef(unsigned int cnt,
        const ParticleType *types) {
        this->cntTypes = cnt;
        if (this->ownTypeMem) {
            delete[] this->types;
        }
        this->types = types;
        this->ownTypeMem = false;
    }

    ///**
    // * Accesses the list of particle types
    // *
    // * @return A pointer to the list of particle types
    // */
    //inline ParticleType * Types(void) {
    //    return this->types;
    //}

    /**
     * Accesses the list of particle types
     *
     * @return A pointer to the list of particle types
     */
    inline const ParticleType *Types(void) const {
        return this->types;
    }

    /**
     * Answer the number of particle types
     *
     * @return The number of particle types
     */
    inline unsigned int TypesCount(void) const {
        return this->cntTypes;
    }

    /**
     * Assignment operator.
     * Makes a deep copy of all members. While for data these are only
     * pointers, the pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    ParticleGridDataCall &operator=(const ParticleGridDataCall &rhs);

private:

    /** The numbers of cells in the grid */
    unsigned int cntCellsX, cntCellsY, cntCellsZ, cntCells;

    /** Pointer to the array of grid cells */
    const GridCell *cells;

    /** Flag if the memory of 'cells' is owned by this call */
    bool ownCellMem;

    /** The number of particle types */
    unsigned int cntTypes;

    /** Pointer to the array of particle types */
    const ParticleType *types;

    /** Flag if the memory of 'types' is owned by this call */
    bool ownTypeMem;

};


/** Description class typedef */
typedef core::factories::CallAutoDescription<ParticleGridDataCall>
ParticleGridDataCallDescription;

}
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARTICLEGRIDDATACALL_H_INCLUDED */
