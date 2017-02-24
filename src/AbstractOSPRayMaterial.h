/*
* AbstractOSPRayMaterial.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "CallOSPRayMaterial.h"
#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"



namespace megamol {
namespace ospray {

class AbstractOSPRayMaterial : public core::Module {
protected:

    /** Ctor. */
    AbstractOSPRayMaterial(void);

    /** Dtor. */
    virtual ~AbstractOSPRayMaterial (void);

    virtual bool create();
    virtual void release();
    bool getMaterialCallback(core::Call& call);
    virtual bool InterfaceIsDirty();
    virtual void readParams();

    OSPRayMaterialContainer materialContainer;

private:
    core::CallerSlot getMaterialSlot;

};


} // namespace ospray
} // namespace megamol