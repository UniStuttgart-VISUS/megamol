//
// MolecularNeighborhood.h
//
// Copyright (C) 2016 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_MOLECULARNEIGHBORHOOD_H_INCLUDED
#define MMPROTEINPLUGIN_MOLECULARNEIGHBORHOOD_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"
#include <vector>

namespace megamol {
namespace protein {

class MolecularNeighborhood : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MolecularNeighborhood";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Computes the local neighborhood for each atom of the ingoing call";
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
    MolecularNeighborhood(void);

    /** Dtor. */
    ~MolecularNeighborhood(void) override;

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
    /**
     * Searches the neighboring atoms for each atom in the given call.
     *
     * @param call The call providing the atom data.
     * @param radius The search radius around each atom.
     */
    void findNeighborhoods(megamol::protein_calls::MolecularDataCall& call, float radius);

    /** data caller slot */
    megamol::core::CallerSlot getDataSlot;

    /** slot for outgoing data */
    megamol::core::CalleeSlot dataOutSlot;

    /** the search radius parameter for the neighborhood search */
    megamol::core::param::ParamSlot neighRadiusParam;

    /** The hash of the last data set rendered */
    SIZE_T lastDataHash;

    /** The last data set hash that was sent to the render */
    SIZE_T lastHashSent;

    /** Vector containing the neighborhood for each atom as array of atom indices */
    std::vector<vislib::Array<unsigned int>> neighborhood;

    /** Vector containing the sizes of the neighborhoods */
    std::vector<unsigned int> neighborhoodSizes;

    /** Pointers to the raw neighborhood data (only relevant to be sent to the outgoing call */
    std::vector<const unsigned int*> dataPointers;
};
} /* namespace protein */
} /* namespace megamol */

#endif // MMPROTEINPLUGIN_MOLECULARNEIGHBORHOOD_H_INCLUDED
