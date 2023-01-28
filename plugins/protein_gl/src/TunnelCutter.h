/*
 * TunnelCutter.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMPROTEINPLUGIN_TUNNELCUTTER_H_INCLUDED
#define MMPROTEINPLUGIN_TUNNELCUTTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

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

class TunnelCutter : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TunnelCutter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module that is able to cut a mesh. This module then only puts out a certain part of the mesh";
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
    TunnelCutter();

    /** Dtor. */
    ~TunnelCutter() override;

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
    /**
     * Cuts away unnecessary parts from the mesh and writes the result into the meshVector.
     * This method tries to cut the mesh equally so that the amount of geometry around the binding site stays the same
     * in all directions
     *
     * @param meshCall The call containing the input mesh.
     * @param cavityMeshCall The call containing the input cavity mesh.
     * @param tunnelCall The call containing the tunnel data the cut region is based on.
     * @param molCall The call containing the molecular data.
     * @param bsCall The call containing the binding site data.
     * @return True on success, false otherwise.
     */
    bool cutMeshEqually(geocalls_gl::CallTriMeshDataGL* meshCall, geocalls_gl::CallTriMeshDataGL* cavityMeshCall,
        protein_calls::TunnelResidueDataCall* tunnelCall, protein_calls::MolecularDataCall* molCall,
        protein_calls::BindingSiteCall* bsCall);

    /** The lastly received data hash */
    SIZE_T lastDataHash;

    /** The lastly received cavity data hash */
    SIZE_T lastCavityDataHash;

    /** The offset to the lastly received hash */
    SIZE_T hashOffset;

    /** Size of the grown region */
    core::param::ParamSlot growSizeParam;

    /** Activation slot for the cutting */
    core::param::ParamSlot isActiveParam;

    /** Parameter slot for the selected tunnel */
    core::param::ParamSlot tunnelIdParam;

    /** Slot for the mesh input. */
    core::CallerSlot meshInSlot;

    /** Slot for the cavity mesh input */
    core::CallerSlot cavityMeshInSlot;

    /** Slot for the tunnel input */
    core::CallerSlot tunnelInSlot;

    /** Slot for the input of the molecular data */
    core::CallerSlot moleculeInSlot;

    /** Slot for the input of the binding site */
    core::CallerSlot bindingSiteInSlot;

    /** Slot for the ouput of the cut mesh */
    core::CalleeSlot cutMeshOutSlot;

    /** Vector containing the modified mesh data */
    std::vector<geocalls_gl::CallTriMeshDataGL::Mesh> meshVector;

    /** Vector containing the information for each vertex whether to keep it or not */
    std::vector<bool> vertexKeepFlags;

    /** Container for the kept vertices */
    std::vector<float> vertices;

    /** Container for the kept vertex normals */
    std::vector<float> normals;

    /** Container for the kept colors */
    std::vector<unsigned char> colors;

    /** Container for the kept atom index vertex attributes */
    std::vector<unsigned int> attributes;

    /** Container for the kept vertex level attributes */
    std::vector<unsigned int> levelAttributes;

    /** Container for the kept binding site distance attributes */
    std::vector<unsigned int> bindingDistanceAttributes;

    /** Container for the kept faces */
    std::vector<unsigned int> faces;

    /** Dirty flag */
    bool dirt;
};

} // namespace megamol::protein_gl


#endif
