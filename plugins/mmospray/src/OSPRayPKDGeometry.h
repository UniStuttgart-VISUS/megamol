/*
 * OSPRayPKDGeometry.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::ospray {

class OSPRayPKDGeometry : public megamol::core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayPKDGeometry";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Creator for OSPRay PKD geometries.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Dtor. */
    ~OSPRayPKDGeometry() override;

    /** Ctor. */
    OSPRayPKDGeometry();

protected:
    bool create() override;
    void release() override;

    bool getDataCallback(core::Call& call);
    bool getExtendsCallback(core::Call& call);
    bool getDirtyCallback(core::Call& call);

    bool InterfaceIsDirty();

    bool InterfaceIsDirtyNoReset() const;

    core::CalleeSlot deployStructureSlot;

    /** The call for data */
    core::CallerSlot getDataSlot;
    SIZE_T datahash;
    int time;

private:
    megamol::core::param::ParamSlot colorTypeSlot;

    long long int ispcLimit = 1ULL << 30;
};

} // namespace megamol::ospray
