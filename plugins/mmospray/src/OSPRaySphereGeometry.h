/*
 * OSPRaySphereGeometry.h
 * Copyright (C) 20021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRaySphereGeometry : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRaySphereGeometry";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Creator for OSPRay sphere geometries.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor. */
    virtual ~OSPRaySphereGeometry(void);

    /** Ctor. */
    OSPRaySphereGeometry(void);

protected:
    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call& call);
    virtual bool getExtends(core::Call& call);


    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;

        // filtered data
    std::vector<std::vector<std::array<float, 3>>> _enabled_vertices;
    std::vector<std::vector<std::array<float, 4>>> _enabled_colors;
    std::vector < std::vector<std::array<float, 3>>> _selected_vertices;

};

} // namespace ospray
} // namespace megamol
