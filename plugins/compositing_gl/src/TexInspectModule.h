/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmstd_gl/special/TextureInspector.h"

namespace megamol::compositing_gl {

/**
 * Class implementing the texture inspector
 */
class TexInspectModule : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TextureInspector";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "A module to inspect the current CallTexture2D texture for debugging purposes.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Ctor
     */
    TexInspectModule();

    /**
     * Dtor
     */
    ~TexInspectModule();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Sets the extents (animation and bounding box) into the call object
     *
     * @param call The incoming call
     *
     * @return 'true' on success
     */
    bool getMetaDataCallback(core::Call& caller);

    /**
     * Renders the scene
     *
     * @param call The incoming call
     *
     * @return 'true' on success
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

private:
    mmstd_gl::special::TextureInspector tex_inspector_;

    /** Slot for optionally querying input textures, i.e. rhs connection */
    megamol::core::CallerSlot get_data_slot_;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot output_tex_slot_;
};

} // namespace megamol::compositing_gl
