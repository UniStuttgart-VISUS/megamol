/*
* CallOSPRayAPIObject.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "OSPRay_plugin/OSPRay_plugin.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRAY_PLUGIN_API CallOSPRayAPIObject : public core::AbstractGetData3DCall {
public:

    /**
    * Answer the name of the objects of this description.
    *
    * @return The name of the objects of this description.
    */
    static const char *ClassName(void) {
        return "CallOSPRayAPIObject";
    }

    /**
    * Gets a human readable description of the module.
    *
    * @return A human readable description of the module.
    */
    static const char *Description(void) {
        return "Call for an OSPRay API object";
    }

    /**
    * Answer the number of functions used for this call.
    *
    * @return The number of functions used for this call.
    */
    static unsigned int FunctionCount(void) {
        return 2;
    }

    /**
    * Answer the name of the function used for this call.
    *
    * @param idx The index of the function to return it's name.
    *
    * @return The name of the requested function.
    */
    static const char * FunctionName(unsigned int idx) {
        switch (idx) {
        case 0: return "GetData";
        case 1: return "GetExtent";
        default: return NULL;
        }
    }

    /**
    * Gets the data defined time stamp
    *
    * @return The data defined time stamp
    */
    inline float GetTimeStamp(void) const {
        return timeStamp;
    }

    /**
    * Sets the data defined time stamp
    *
    * @param timeStamp The new time stamp value
    */
    void SetTimeStamp(float timeStamp) {
        this->timeStamp = timeStamp;
    }

    /** Ctor. */
    CallOSPRayAPIObject();

    /** Dtor. */
    virtual ~CallOSPRayAPIObject(void);

    void setAPIObject(void* api_obj);
    void* getAPIObject();

    void setStructureType(structureTypeEnum strtype);
    structureTypeEnum getStructureType();

private:
    void* api_obj;
    float timeStamp;
    structureTypeEnum type;

};
typedef core::factories::CallAutoDescription<CallOSPRayAPIObject> CallOSPRayAPIObjectDescription;

} // namespace ospray
} // namespace megamol
