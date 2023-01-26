/*
 * MultiParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/AbstractParticleDataCall.h"
#include "geometry_calls/SimpleSphericalParticles.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::geocalls {

template class AbstractParticleDataCall<SimpleSphericalParticles>;


/**
 * Call for multi-stream particle data.
 */
class MultiParticleDataCall : public AbstractParticleDataCall<SimpleSphericalParticles> {
public:
    /** typedef for legacy name */
    typedef SimpleSphericalParticles Particles;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "MultiParticleDataCall";
    }

    /** Ctor. */
    MultiParticleDataCall(void);

    /** Dtor. */
    ~MultiParticleDataCall(void) override;

    /**
     * Assignment operator.
     * Makes a deep copy of all members. While for data these are only
     * pointers, the pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    MultiParticleDataCall& operator=(const MultiParticleDataCall& rhs);
};


/** Description class typedef */
typedef core::factories::CallAutoDescription<MultiParticleDataCall> MultiParticleDataCallDescription;


} /* end namespace megamol::geocalls */
