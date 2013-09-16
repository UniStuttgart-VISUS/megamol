//
// SurfaceMapper.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 16, 2013
// Author     : scharnkn
//

#ifdef WITH_CUDA

#ifndef MMPROTEINPLUGIN_SURFACEMAPPER_H_INCLUDED
#define MMPROTEINPLUGIN_SURFACEMAPPER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"

namespace megamol {
namespace protein {

/*
 * This class establishes a mapping relation between a source surface (given by
 * a triangle mesh) and a target shape (given by a level set in a volume
 * texture). The mapping is done using a 'deformable model' approach.
 * Optionally, diffusion is applied to the given volume texture to improve
 * convergence and capture range. The output are mapped positions for all
 * vertices in the source mesh.
 */
class SurfaceMapper : public core::Module {

public:

    /** CTor */
    SurfaceMapper(void);

    /** DTor */
    virtual ~SurfaceMapper(void);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "SurfaceMapper";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Establishes a mapping relation between two surfaces using a \
'deformable model' approach.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Call callback to get the mapped vertex positions
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVtxData(core::Call& call);

    /**
     * Call callback to get the vbo extent
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVtxExtent(core::Call& call);

    /**
     * Call callback to get the GVF volume texture
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVolData(core::Call& call);

    /**
     * Call callback to get the GVF volume texture extent
     *
     * @param call The calling call
     * @return True on success
     */
    bool getVolExtent(core::Call& call);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

private:

    /* Data caller/callee slots */

    /// Caller slot for incoming vertex data (representing the source mesh)
    core::CallerSlot vtxInputSlot;

    /// Callee slot for outgoing vertex data (representing the surface mapping)
    core::CalleeSlot vtxOutputSlot;

    /// Caller slot for incoming volume data (representing the target shape)
    core::CallerSlot volInputSlot;

    /// Callee slot to output the gradient vector flow (mainly for debugging)
    core::CalleeSlot volOutputSlot;

};

} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_SURFACEMAPPER_H_INCLUDED

#endif // WITH_CUDA
