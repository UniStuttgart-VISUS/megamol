/*
 * QuartzParticleDataCall.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/IllegalStateException.h"
#include "vislib/OutOfRangeException.h"


namespace megamol::demos_gl {

/**
 * Call transporting quartz crystal particle data
 */
class ParticleDataCall : public megamol::core::AbstractGetData3DCall {
public:
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
        return "QuartzParticleDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call transporting quartz crystal particle data";
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
    ParticleDataCall();

    /** Dtor. */
    ~ParticleDataCall() override;

    /**
     * Gets the number of groups
     *
     * @param idx The zero-based group index
     *
     * @return the number of groups
     */
    inline unsigned int GetGroupCount() const {
        return this->grpCnt;
    }

    /**
     * Gets the crystal type indices of the groups
     *
     * @param idx The zero-based group index
     *
     * @return the crystal type indices of the groups
     */
    inline unsigned int GetCrystalType(unsigned int idx) const {
        if (this->types == NULL)
            throw vislib::IllegalStateException("No data", __FILE__, __LINE__);
        if (idx >= this->grpCnt)
            throw vislib::OutOfRangeException(idx, 0, this->grpCnt - 1, __FILE__, __LINE__);
        return this->types[idx];
    }

    /**
     * Gets the particle count per group
     *
     * @param idx The zero-based group index
     *
     * @return the particle count per group
     */
    inline unsigned int GetParticleCount(unsigned int idx) const {
        if (this->cnt == NULL)
            throw vislib::IllegalStateException("No data", __FILE__, __LINE__);
        if (idx >= this->grpCnt)
            throw vislib::OutOfRangeException(idx, 0, this->grpCnt - 1, __FILE__, __LINE__);
        return this->cnt[idx];
    }

    /**
     * Gets the particle data (x y z r qx qy qz qw)
     *
     * @param idx The zero-based group index
     *
     * @return the particle data (x y z r qx qy qz qw)
     */
    inline const float* GetParticleData(unsigned int idx) const {
        if (this->part == NULL)
            throw vislib::IllegalStateException("No data", __FILE__, __LINE__);
        if (idx >= this->grpCnt)
            throw vislib::OutOfRangeException(idx, 0, this->grpCnt - 1, __FILE__, __LINE__);
        return this->part[idx];
    }

    /**
     * Sets teh particle data
     *
     * @param grpCnt The number of groups
     * @param types The crystal type indices of the groups
     * @param cnts The particle count per group
     * @param parts The particle data (x y z r qx qy qz qw)
     */
    inline void SetParticleData(
        unsigned int grpCnt, const unsigned int* types, const unsigned int* cnts, const float* const* parts) {
        this->grpCnt = grpCnt;
        this->types = types;
        this->cnt = cnts;
        this->part = parts;
    }

private:
    /** Number of groups */
    unsigned int grpCnt;

    /** The crystal type indices of the groups */
    const unsigned int* types;

    /** The particle count per group */
    const unsigned int* cnt;

    /** The particle data (x y z r qx qy qz qw) */
    const float* const* part;
};

} // namespace megamol::demos_gl
