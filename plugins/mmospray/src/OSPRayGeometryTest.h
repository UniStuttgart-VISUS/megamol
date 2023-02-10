/*
 * OSPRayGeometryTest.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol::ospray {

class OSPRayGeometryTest : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "OSPRayGeometryTest";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Creator for OSPRay no overhead sphere geometries.";
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
    ~OSPRayGeometryTest() override;

    /** Ctor. */
    OSPRayGeometryTest();

protected:
    /**
     * color transfer helper
     * @param array with gray scales
     * @param transferfunction table/texture
     * @param transferfunction table/texture size
     * @param target array (rgba)
     */
    //void colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray);

    bool create() override;
    void release() override;

    bool readData(core::Call& call) override;
    bool getExtends(core::Call& call) override;


    bool InterfaceIsDirty();
};

} // namespace megamol::ospray
