/*
 * AbstractOSPRayMaterial.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/CallOSPRayMaterial.h"


namespace megamol::ospray {

class AbstractOSPRayMaterial : public core::Module {
protected:
    /** Ctor. */
    AbstractOSPRayMaterial();

    /** Dtor. */
    ~AbstractOSPRayMaterial() override;

    bool create() override;
    void release() override;
    bool getMaterialCallback(core::Call& call);
    virtual bool InterfaceIsDirty() {
        return true;
    };
    virtual void readParams(){};

    OSPRayMaterialContainer materialContainer;

private:
    core::CalleeSlot deployMaterialSlot;
};


} // namespace megamol::ospray
