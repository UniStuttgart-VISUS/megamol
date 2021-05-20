/*
* CallOSPRayTransformation.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/Call.h"
#include <array>

namespace megamol {
namespace ospray {

struct OSPRayTransformationContainer {

    std::array<float,3> pos;
    std::array<std::array<float,3>,3> MX;

    bool isValid = false;

};


class CallOSPRayTransformation : public megamol::core::Call {
public:

    /**
    * Answer the name of the objects of this description.
    *
    * @return The name of the objects of this description.
    */
    static const char *ClassName(void) {
        return "CallOSPRayTransformation";
    }

    /**
    * Gets a human readable description of the module.
    *
    * @return A human readable description of the module.
    */
    static const char *Description(void) {
        return "Call for an OSPRay transformation";
    }

    /**
    * Answer the number of functions used for this call.
    *
    * @return The number of functions used for this call.
    */
    static unsigned int FunctionCount(void) {
        return 1;
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
        case 0: return "GetTransformation";
        default: return NULL;
        }
    }

    /** Ctor. */
    CallOSPRayTransformation(void);

    /** Dtor. */
    virtual ~CallOSPRayTransformation(void);

    void setTransformationContainer(std::shared_ptr<OSPRayTransformationContainer> tc);
    std::shared_ptr<OSPRayTransformationContainer> getTransformationParameter();

    /**
    * Assignment operator
    *
    * @param rhs The right hand side operand
    *
    * @return A reference to this
    */
    CallOSPRayTransformation& operator=(const CallOSPRayTransformation& rhs);

    bool InterfaceIsDirty();
    void setDirty();

private:
    std::shared_ptr<OSPRayTransformationContainer> _transformationContainer;
    bool _isDirty;

};
typedef core::factories::CallAutoDescription<CallOSPRayTransformation> CallOSPRayTransformationDescription;


} // namespace ospray
} // namespace megamol