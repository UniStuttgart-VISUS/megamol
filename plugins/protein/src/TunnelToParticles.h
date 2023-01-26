/*
 * TunnelToParticles.h
 * Copyright (C) 2006-2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMPROTEINPLUGIN_TUNNELTOPARTICLES_H_INCLUDED
#define MMPROTEINPLUGIN_TUNNELTOPARTICLES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace protein {

class TunnelToParticles : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "TunnelToParticles";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module for writing tunnel-vertex-information into a MultiParticleDataCall";
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
    TunnelToParticles(void);

    /** Dtor. */
    ~TunnelToParticles(void) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'release'.
     */
    void release(void) override;

    /**
     * Call for get data.
     */
    bool getData(megamol::core::Call& call);

    /**
     * Call for get extent.
     */
    bool getExtent(megamol::core::Call& call);

private:
    /** Slot for the particle data output */
    core::CalleeSlot dataOutSlot;

    /** Slot for the tunnel data input */
    core::CallerSlot tunnelInSlot;
};

} /* end namespace protein */
} /* end namespace megamol */

#endif
