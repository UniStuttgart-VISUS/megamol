/*
 * OSCBFix.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OSCBFIX_H_INCLUDED
#define MEGAMOLCORE_OSCBFIX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace core::moldyn {

/** forward declaration of supported call */
class MultiParticleDataCall;

} // namespace core::moldyn

namespace quartz {


/**
 * Module loading a quartz crystal definition file
 */
class OSCBFix : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSCBFix";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module fixing the object space clip box, by accuratly calculating it's extents";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    OSCBFix();

    /** Dtor */
    ~OSCBFix() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(core::Call& c);

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getExtent(core::Call& c);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    void calcOSCB(class geocalls::MultiParticleDataCall& data);

    /** The data callee slot */
    core::CalleeSlot dataOutSlot;

    /** The data caller slot */
    core::CallerSlot dataInSlot;

    /** The data hash */
    SIZE_T datahash;

    /** The frame number */
    unsigned int frameNum;

    /** The new and improved object space clipping box */
    vislib::math::Cuboid<float> oscb;
};

} /* end namespace quartz */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_OSCBFIX_H_INCLUDED */
