/*
 * MSMSCavityFinder.h
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMPROTEINPLUGIN_MSMSCAVITYFINDER_H_INCLUDED
#define MMPROTEINPLUGIN_MSMSCAVITYFINDER_H_INCLUDED
#pragma once

#include "protein_calls/ProteinHelpers.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/TunnelResidueDataCall.h"


namespace megamol::protein_gl {

class MSMSCavityFinder : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MSMSCavityFinder";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module that finds cavities based on the input of two MSMS meshes (using the 3V method).";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    MSMSCavityFinder();

    /** Dtor. */
    ~MSMSCavityFinder() override;

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
    /** Slot for the inner mesh input. */
    core::CallerSlot innerMeshInSlot;
    /** Slot for the outer mesh input. */
    core::CallerSlot outerMeshInSlot;

    /** Slot for the ouput of the cut mesh */
    core::CalleeSlot cutMeshOutSlot;

    /** Param slots */
    core::param::ParamSlot distanceParam;
    core::param::ParamSlot areaParam;

    /** data hash */
    SIZE_T dataHash;

    // variables for detecting new data
    int lastFrame;
    SIZE_T lastHashInner;
    SIZE_T lastHashOuter;
    std::vector<size_t> vertexIndex;
    std::vector<float> distanceToMesh;
    megamol::geocalls_gl::CallTriMeshDataGL::Mesh cavityMesh;
    vislib::Array<unsigned int> triaIndices;
    vislib::Array<megamol::geocalls_gl::CallTriMeshDataGL::Mesh> cavitySubmeshes;
};

} // namespace megamol::protein_gl

#endif
