/*
 * MultiParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "mmcore/moldyn/SimpleSphericalParticles.h"

namespace megamol {
namespace core {
namespace moldyn {

MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<SimpleSphericalParticles>;


/**
 * Call for multi-stream particle data.
 */
class MEGAMOLCORE_API MultiParticleDataCall : public AbstractParticleDataCall<SimpleSphericalParticles> {
public:
    /** typedef for legacy name */
    typedef SimpleSphericalParticles Particles;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "MultiParticleDataCall"; }

    /** Ctor. */
    MultiParticleDataCall(void);

    /** Dtor. */
    virtual ~MultiParticleDataCall(void);

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
typedef factories::CallAutoDescription<MultiParticleDataCall> MultiParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED */
