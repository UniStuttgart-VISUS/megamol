/*
* AbstractOSPRayStructure.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "OSPRay_plugin/OSPRay_plugin.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"




namespace megamol {
namespace ospray {

class OSPRAY_PLUGIN_API AbstractOSPRayStructure : public megamol::core::Module {
protected:
    /** Dtor. */
    virtual ~AbstractOSPRayStructure(void);

    /** Ctor. */
    AbstractOSPRayStructure(void);

    virtual bool create() { return true; }
    virtual void release() { this->Release(); }
    virtual bool readData(megamol::core::Call &call) { return true; }
    virtual bool getExtends(megamol::core::Call &call) { return true; }
    bool getExtendsCallback(megamol::core::Call &call);
    bool getStructureCallback(core::Call& call);

    /** The callee for Structure */
    core::CalleeSlot deployStructureSlot;

    /** The call for additional structures */
    core::CallerSlot getStructureSlot;

    /** The call for materials */
    core::CallerSlot getMaterialSlot;

    SIZE_T datahash;
    float time;

    OSPRayStructureContainer structureContainer;
    OSPRayExtendContainer extendContainer;

};


} // namespace ospray
} // namespace megamol