/*
 * ProteinAligner.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Author: Karsten Schatz
 * All rights reserved.
 */
#ifndef MMPROTEINPLUGIN_PROTEINALIGNER_H_INCLUDED
#define MMPROTEINPLUGIN_PROTEINALIGNER_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "protein_calls/MolecularDataCall.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Point.h"

namespace megamol {
namespace protein {

class ProteinAligner : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ProteinAligner"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Aligns two given proteins by performing a RMSD calculation. The input protein is aligned against a "
               "second reference protein";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ProteinAligner(void);

    /** Dtor. */
    virtual ~ProteinAligner(void);

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
    bool getData(core::Call& call);

    /**
     * Call for get extent.
     */
    bool getExtents(core::Call& call);

private:

    /**
     * Aligns the atom positions of the input protein onto the reference protein
     * The result will be written to the alignedPositions vector
     * 
     * @param input Call containing the input protein
     * @param ref Call containing the reference protein
     */
    bool alignPositions(const protein_calls::MolecularDataCall& input, const protein_calls::MolecularDataCall& ref);

    /**
     * Retrieves all positions of c alpha atoms from an input call and writes them into the given array
     *
     * @param input The input protein call
     * @param cAlphaPositions Will contain the retrieved c alpha positions
     */
    void getCAlphaPosList(const protein_calls::MolecularDataCall& input, std::vector<float>& cAlphaPositions);

    /** Output slot for the moved and rotated protein */
    core::CalleeSlot dataOutSlot;

    /** Input protein slot */
    core::CallerSlot inputProteinSlot;

    /** Input slot for the reference protein that is not altered */
    core::CallerSlot referenceProteinSlot;

    /** Slot that enables or disables the alignment of this module */
    core::param::ParamSlot isActiveSlot;

    /** vector containing the aligned atom position data */
    std::vector<float> alignedPositions;

    /** the new bounding box of the data */
    vislib::math::Cuboid<float> boundingBox;
};

} // namespace protein
} // namespace megamol

#endif /* MMPROTEINPLUGIN_PROTEINALIGNER_H_INCLUDED */
