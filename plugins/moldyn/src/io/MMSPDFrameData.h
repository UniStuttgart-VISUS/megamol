/*
 * MMSPDFrameData.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "vislib/Array.h"
//#include "vislib/Cuboid.h"
#include "vislib/RawStorage.h"
#include "vislib/memutils.h"
#include "vislib/types.h"


namespace megamol::moldyn::io {


/**
 * MMSPD file frame data
 */
class MMSPDFrameData {
public:
    /**
     * Class holding all particles of one type
     *
     * NOTE: NO Double supported! Doubles will be converted to floats!
     */
    class Particles {
    public:
        /**
         * Ctor
         */
        Particles();

        /**
         * Dtor
         */
        ~Particles();

        /**
         * Allocates the field map overwriting any previously stored field map
         *
         * @param size The size of the new field map in entries
         */
        inline void AllocateFieldMap(SIZE_T size) {
            delete[] this->fieldMap;
            this->fieldMap = new unsigned int[size];
            ZeroMemory(this->fieldMap, sizeof(unsigned int) * size);
        }

        /**
         * Gets the number of particles
         *
         * @return The number of particles
         */
        inline UINT64 Count() const {
            return this->count;
        }

        /**
         * Accesses the particle data
         *
         * @return The particle data
         */
        inline vislib::RawStorage& Data() {
            return this->data;
        }

        /**
         * Accesses the field map of the particle data
         *
         * @return The field map of the particle data
         */
        inline unsigned int* FieldMap() {
            return this->fieldMap;
        }

        /**
         * Answer the field map of the particle data
         *
         * @return The field map of the particle data
         */
        inline const unsigned int* const FieldMap() const {
            return this->fieldMap;
        }

        /**
         * Gets the particle data
         *
         * @return The particle data
         */
        inline const vislib::RawStorage& GetData() const {
            return this->data;
        }

        /**
         * Sets the number of particles
         *
         * @param cnt The number of particles
         */
        inline void SetCount(UINT64 cnt) {
            this->count = cnt;
        }

        /**
         * Test for equality
         *
         * @param rhs The right-hand side operand
         *
         * @return True if 'this' and 'rhs' are equal
         */
        bool operator==(const Particles& rhs);

    private:
        /** The number of particles */
        UINT64 count;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
        /**
         * The particle data
         *
         * This is the raw data following the variable fields defined by
         * the particle type. (never storing the type id field!)
         *
         * NOTE: NO Double supported! Doubles will be converted to floats!
         */
        vislib::RawStorage data;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

        /**
         * Mapping field indices of the header to storage positions.
         *
         * This is used to reorder the fields to group fields in memory
         * for optimal use for visualizaion.
         */
        unsigned int* fieldMap;
    };

    /**
     * Ctor
     *
     * @param owner The owning data module
     */
    MMSPDFrameData();

    /**
     * Dtor
     */
    virtual ~MMSPDFrameData();

    /**
     * Accesses the particle data
     *
     * @return The particle data
     */
    inline vislib::Array<Particles>& Data() {
        return this->data;
    }

    /**
     * Gets the particle data
     *
     * @return The particle data
     */
    inline const vislib::Array<Particles>& GetData() const {
        return this->data;
    }

    /**
     * Accesses the particle index reconstruction data
     *
     * @return The particle index reconstruction data
     */
    inline vislib::RawStorage& IndexReconstructionData() {
        return this->idxRec;
    }

    /**
     * Gets the particle index reconstruction data
     *
     * @return The particle index reconstruction data
     */
    inline const vislib::RawStorage& GetIndexReconstructionData() const {
        return this->idxRec;
    }

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /**
     * The particle data
     *
     * For each particle type (array index) the list of particles is stored
     * with their raw data corresponding to the field of the particle type
     * definition (optionally including UINT64 index; but never including
     * the type).
     */
    vislib::Array<Particles> data;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /**
     * The particle index reconstruction data
     *
     * If loaded, this memory contains the blocks from the different arrays
     * of 'data' in the order they were stored in the data file. The first
     * integer defines the type index and the second integer defines the
     * number of particles.
     * All integers are RLE-Bit-Encoded.
     * (worst case 2 byte per particle)
     */
    vislib::RawStorage idxRec;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};


} // namespace megamol::moldyn::io
