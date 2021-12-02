/*
 * TunnelToBFactor.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMPROTEINPLUGIN_TUNNELTOBFACTOR_H_INCLUDED
#define MMPROTEINPLUGIN_TUNNELTOBFACTOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/TunnelResidueDataCall.h"

namespace megamol {
namespace protein {

class TunnelToBFactor : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "TunnelToBFactor";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module for writing tunnel-information to the B-factor of a MolecularDataCall";
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
    TunnelToBFactor(void);

    /** Dtor. */
    virtual ~TunnelToBFactor(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * Call for get data.
     */
    bool getData(megamol::core::Call& call);

    /**
     * Call for get extent.
     */
    bool getExtent(megamol::core::Call& call);

private:
    /**
     * Applies the B-Factor changes to the outgoing call
     */
    void applyBFactor(protein_calls::MolecularDataCall* outCall, protein_calls::MolecularDataCall* inCall,
        protein_calls::TunnelResidueDataCall* tunnelCall);

    /** Slot for the MolecularDataCall output */
    core::CalleeSlot dataOutSlot;

    /** Slot for the molecule data input */
    core::CallerSlot molInSlot;

    /** Slot for the tunnel input */
    core::CallerSlot tunnelInSlot;

    /** Storage for the new bFactors */
    std::vector<float> bFactors;
};

} /* end namespace protein */
} /* end namespace megamol */

#endif
