/*
 * OSPRayGlyphGeometry.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "OSPRay_plugin/AbstractOSPRayStructure.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"

namespace megamol {
namespace probe {
    
class OSPRayGlyphGeometry : public ospray::AbstractOSPRayStructure {
    
    public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "OSPRayGlyphGeometry"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Creator for OSPRayGlyphGeometry."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Dtor. */
    virtual ~OSPRayGlyphGeometry(void);

    /** Ctor. */
    OSPRayGlyphGeometry(void);

protected:

    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call& call);
    virtual bool getExtends(core::Call& call);

    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot _get_mesh_slot;
    core::CallerSlot _get_texture_slot;

private:

    size_t _img_data_cached_hash;

};

} // namespace probe
} // namespace megamol