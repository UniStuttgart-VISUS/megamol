/*
 * OSPRayAOVSphereGeometry.h
 * Copyright (C) 2009-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/Module.h"

namespace megamol {
namespace ospray {

class OSPRayAOVSphereGeometry : public core::Module {

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

    bool getDataCallback(core::Call& call);
    bool getExtendsCallback(core::Call& call);
    bool getDirtyCallback(core::Call& call);

    bool InterfaceIsDirty();

    bool InterfaceIsDirtyNoReset() const;

    core::param::ParamSlot samplingRateSlot;

    core::param::ParamSlot aoThresholdSlot;

    core::param::ParamSlot aoRayOffsetFactorSlot;

    /** The call for data */
    core::CallerSlot getDataSlot;

    core::CallerSlot getVolSlot;

    core::CalleeSlot deployStructureSlot;

    size_t volDatahash;

    unsigned int volFrameID;

private:
    // color transfer data
    unsigned int tex_size;

    size_t datahash;

    size_t time;

    std::array<float,2> valuerange;

    std::array<float,3> gridorigin;

    std::array<float,3> gridspacing;

    std::array<int,3> dimensions;

    long long int ispcLimit = 1ULL << 30;
};

} // namespace ospray
} // namespace megamol
