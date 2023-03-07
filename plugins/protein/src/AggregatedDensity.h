/*
 * BSpline.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol::protein {


/**
 * Class generation buck ball informations
 */
class AggregatedDensity : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "AggregatedDensity";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Calculates the density from molecular data";
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
    AggregatedDensity();

    /** Dtor */
    ~AggregatedDensity() override;

protected:
    bool aggregate();
    bool aggregate_frame(float* pos, float* vel, unsigned int n_atoms);

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /**

     * @return 'true' on success, 'false' on failure.
     */
    bool getDensityCallback(megamol::core::Call& caller);

    /**
     * @return 'true' on success, 'false' on failure.
     */
    bool getZvelocityCallback(megamol::core::Call& caller);


    /**
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(megamol::core::Call& caller);

    /** The slot for requesting data */
    megamol::core::CalleeSlot getDensitySlot;

    /** The slot for requesting data */
    megamol::core::CalleeSlot getZvelocitySlot;


    /** MolecularDataCall caller slot */
    megamol::core::CallerSlot molDataCallerSlot;

    /** The distance volume resolution */
    unsigned int volRes;

    /** The distance volume */
    float* vol;

    std::vector<std::string> xtcfilenames;
    std::string pdbfilename;
    float origin_x;
    float origin_y;
    float origin_z;
    float box_x;
    float box_y;
    float box_z;
    float res;
    float* density;
    float* velocity;
    unsigned int xbins;
    unsigned int ybins;
    unsigned int zbins;
    bool is_aggregated;
    unsigned int framecounter;
};


} // namespace megamol::protein
