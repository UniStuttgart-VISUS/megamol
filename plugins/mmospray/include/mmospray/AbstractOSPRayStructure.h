/*
 * AbstractOSPRayStructure.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "CallOSPRayTransformation.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmospray/CallOSPRayStructure.h"
#include "mmstd/flags/FlagCalls.h"


namespace megamol {
namespace ospray {

class AbstractOSPRayStructure : public megamol::core::Module {
protected:
    /** Dtor. */
    virtual ~AbstractOSPRayStructure();

    /** Ctor. */
    AbstractOSPRayStructure();

    virtual bool create() {
        return true;
    }
    virtual void release() {
        this->Release();
    }
    virtual bool readData(megamol::core::Call& call) {
        return true;
    }
    virtual bool getExtends(megamol::core::Call& call) {
        return true;
    }
    bool getExtendsCallback(megamol::core::Call& call);
    bool getStructureCallback(core::Call& call);
    void processMaterial();
    void processTransformation();
    void processClippingPlane();

    /** The callee for Structure */
    core::CalleeSlot deployStructureSlot;

    /** The call for additional structures */
    core::CallerSlot getStructureSlot;

    /** The call for materials */
    core::CallerSlot getMaterialSlot;

    /** The call for transformation */
    core::CallerSlot getTransformationSlot;

    core::CallerSlot getClipplaneSlot;

    core::CallerSlot writeFlagsSlot;
    core::CallerSlot readFlagsSlot;

    SIZE_T datahash;
    float time;
    size_t frameID;

    OSPRayStructureContainer structureContainer;
    OSPRayExtendContainer extendContainer;
};


} // namespace ospray
} // namespace megamol
