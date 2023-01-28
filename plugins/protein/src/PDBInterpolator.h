#ifndef MMPROTEINPLUGIN_PDBINTERPOLATOR_H_INCLUDED
#define MMPROTEINPLUGIN_PDBINTERPOLATOR_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol::protein {

class PDBInterpolator : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "PDBInterpolator";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Interpolates the pdb data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    PDBInterpolator();
    ~PDBInterpolator() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'release'.
     */
    void release() override;

    /**
     * Call for get data.
     */
    bool getData(megamol::core::Call& call);

    /**
     * Call for get extent.
     */
    bool getExtent(megamol::core::Call& call);

private:
    /** data caller slot */
    megamol::core::CallerSlot getDataSlot;
    megamol::core::CalleeSlot dataOutSlot;
};

} // namespace megamol::protein

#endif // MMPROTEINPLUGIN_PDBINTERPOLATOR_H_INCLUDED
