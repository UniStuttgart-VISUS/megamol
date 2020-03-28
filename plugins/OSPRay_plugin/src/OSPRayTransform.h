/*
* OSPRayTransform.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once
#include "OSPRayTransform.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/Module.h"
#include "CallOSPRayTransformation.h"


namespace megamol {
namespace ospray {

class OSPRayTransform : public core::Module {
public:
    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayTransform";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Configuration module for an OSPRay transformation";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    OSPRayTransform(void);

    /** Dtor. */
    virtual ~OSPRayTransform(void);

    virtual bool create() {return true;}
    virtual void release() {}
    bool getTransformationCallback(core::Call& call);


private:
    core::CalleeSlot _deployTransformationSlot;
    OSPRayTransformationContainer _transformationContainer;

    core::param::ParamSlot _pos;
    core::param::ParamSlot _rot;
    core::param::ParamSlot _scale;

    virtual bool InterfaceIsDirty();
    virtual void readParams();

};


} // namespace ospray
} // namespace megamol


