/*
 * ParticleListMergeModule.h
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARTICLELISTMERGEMODULE_H_INCLUDED
#define MEGAMOLCORE_PARTICLELISTMERGEMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "datatools/AbstractParticleManipulator.h"
#include "datatools_gl/TransferFunctionQuery.h"
#include "mmcore/Module.h"
#include "vislib/math/Cuboid.h"


namespace megamol::datatools_gl::misc {


/**
 * In-Between management module to change time codes of a data set
 */
class ParticleListMergeModule : public datatools::AbstractParticleManipulator {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ParticleListMergeModule";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module to merge all lists from the MultiParticleDataCall into a single list";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    ParticleListMergeModule(void);

    /** Dtor. */
    virtual ~ParticleListMergeModule(void);

protected:
    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    /**
     * Copies the incoming data 'inDat' into the object's fields
     *
     * @param inDat The incoming data
     */
    void setData(geocalls::MultiParticleDataCall& inDat);

    /** The transfer function query */
    TransferFunctionQuery tfq;

    /** The hash id of the data stored */
    size_t dataHash;

    /** The frame id of the data stored */
    unsigned int frameId;

    /** The single list of particles */
    geocalls::MultiParticleDataCall::Particles parts;

    /** The stored particle data */
    vislib::RawStorage data;
};
} // namespace megamol::datatools_gl::misc

#endif /* MEGAMOLCORE_PARTICLELISTMERGEMODULE_H_INCLUDED */
