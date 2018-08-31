/*
 * OSPRayAOVSphereGeometry.h
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "OSPRay_plugin/AbstractOSPRayStructure.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace ospray {

class OSPRayAOVSphereGeometry : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "OSPRayAOVSphereGeometry"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Creator for OSPRay no overhead sphere geometries with volume-based ao approximation.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Dtor. */
    virtual ~OSPRayAOVSphereGeometry(void);

    /** Ctor. */
    OSPRayAOVSphereGeometry(void);

protected:
    /**
     * color transfer helper
     * @param array with gray scales
     * @param transferfunction table/texture
     * @param transferfunction table/texture size
     * @param target array (rgba)
     */
    // void colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize,
    // std::vector<float> &rgbaArray);

    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call& call);
    virtual bool getExtends(core::Call& call);


    bool InterfaceIsDirty();

    core::param::ParamSlot particleList;

    core::param::ParamSlot samplingRateSlot;

    core::param::ParamSlot aoThresholdSlot;

    core::param::ParamSlot aoRayOffsetFactorSlot;

    /** The call for data */
    core::CallerSlot getDataSlot;

    core::CallerSlot getVolSlot;

    size_t volDatahash;

    unsigned int volFrameID;

private:
    // color transfer data
    unsigned int tex_size;

    std::pair<float, float> valuerange;

    std::vector<float> gridorigin;

    std::vector<float> gridspacing;

    std::vector<int> dimensions;
};

} // namespace ospray
} // namespace megamol