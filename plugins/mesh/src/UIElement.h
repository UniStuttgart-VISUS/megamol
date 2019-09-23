/*
 * UIElement.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef UI_ELEMENT_H_INCLUDED
#define UI_ELEMENT_H_INCLUDED

#include "mesh/MeshCalls.h"
#include "mesh/mesh.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace mesh {

class MESH_API UIElement : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "UIElement"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "UI element for 3d in viewport manipulation."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    UIElement();
    ~UIElement();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    bool getMetaDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

private:
    /** The gltf file name */
    //core::param::ParamSlot m_interaction_axis;

    /** The gltf file name */
    //core::param::ParamSlot m_interaction_origin;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_getData_slot;

    /** The slot for chaining */
    megamol::core::CallerSlot m_UIElement_callerSlot;
};

} // namespace mesh
} // namespace megamol


#endif // !UI_ELEMENT_H_INCLUDED
