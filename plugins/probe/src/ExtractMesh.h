/*
 * ExtractMesh.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace probe {

class ExtractMesh : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ExtractMesh"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Extracts a mesh from point data."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ExtractMesh(void);

    /** Dtor. */
    virtual ~ExtractMesh(void);

protected:


    virtual bool create();
    virtual void release();


    core::CallerSlot _getDataCall;
    core::CalleeSlot _deployMeshCall;
    core::param::ParamSlot _algorithmSlot;
    core::param::ParamSlot _xSlot;
    core::param::ParamSlot _ySlot;
    core::param::ParamSlot _zSlot;
    core::param::ParamSlot _xyzSlot;
    core::param::ParamSlot _formatSlot;
    core::param::ParamSlot _alphaSlot;



private:

    bool InterfaceIsDirty();
    //virtual void readParams();
    void calculateAlphaShape();

    bool getData(core::Call& call);

    bool getMetaData(core::Call& call); 

};

} // namespace probe
} // namespace megamol
