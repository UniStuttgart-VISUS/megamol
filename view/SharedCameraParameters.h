//
// SharedCameraParameters.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MEGAMOLCORE_SHAREDCAMERAPARAMETERS_H_INCLUDED
#define MEGAMOLCORE_SHAREDCAMERAPARAMETERS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CalleeSlot.h"
#include "vislib/CameraParamsStore.h"
#include "Module.h"

namespace megamol {
namespace core {
namespace view {

class SharedCameraParameters : public core::Module,
    public vislib::graphics::CameraParamsStore {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "SharedCameraParameters";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Module that syncs camera parameters amongst several view modules.";
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
    SharedCameraParameters(void);

    /** Dtor. */
    virtual ~SharedCameraParameters(void);

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return true if the initialisation was successful, false otherwise.
     */
    virtual bool create (void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release (void);

   /**
    * Call callback to store the current shared camera parameters in the calling
    * call
    *
    * @param c The calling call
    * @return True on success
    */
    bool getCamParams(megamol::core::Call& call);

    /**
     * Call callback to obtain the current shared camera parameters from the
     * calling call
     *
     * @param c The calling call
     * @return True on success
     */
    bool setCamParams(megamol::core::Call& call);

private:

    /// Callee slot for camera parameters
    core::CalleeSlot camParamsSlot;

    /// Flag whether the module can provide valid camera parameters
    bool valid;

};

} // end namespace view
} // end namespace core
} // end namespace megamol

#endif // MEGAMOLCORE_SHAREDCAMERAPARAMETERS_H_INCLUDED
