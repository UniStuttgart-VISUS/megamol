/*
* CallOSPRayLight.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/Call.h"
#include <functional>
#include <vector>
#include "vislib/String.h"

namespace megamol {
namespace ospray {



enum lightenum {
    NONE,
    DISTANTLIGHT,
    POINTLIGHT,
    SPOTLIGHT,
    QUADLIGHT,
    AMBIENTLIGHT,
    HDRILIGHT
};

class OSPRayLightContainer {
private:
public:
    // General light parameters
    std::vector<float> lightColor;
    float lightIntensity;
    lightenum lightType;
    // Distant light parameters
    std::vector<float> dl_direction;
    float dl_angularDiameter;
    bool dl_eye_direction;
    // point light paramenters
    std::vector<float> pl_position;
    float pl_radius;
    // spot light parameters
    std::vector<float> sl_position;
    std::vector<float> sl_direction;
    float sl_openingAngle;
    float sl_penumbraAngle;
    float sl_radius;
    // quad light parameters
    std::vector<float> ql_position;
    std::vector<float> ql_edgeOne;
    std::vector<float> ql_edgeTwo;
    // hdri light parameters
    std::vector<float> hdri_up;
    std::vector<float> hdri_direction;
    vislib::TString hdri_evnfile;



    OSPRayLightContainer::OSPRayLightContainer();
    OSPRayLightContainer::~OSPRayLightContainer();
    void OSPRayLightContainer::release();

};

typedef std::function<void(std::shared_ptr<ospray::OSPRayLightContainer>, unsigned int)> LightDelegate;

class CallOSPRayLight: public core::Call {
public:

    /**
    * Answer the name of the objects of this description.
    *
    * @return The name of the objects of this description.
    */
    static const char *ClassName(void) {
        return "CallOSPRayLight";
    }

    /**
    * Gets a human readable description of the module.
    *
    * @return A human readable description of the module.
    */
    static const char *Description(void) {
        return "Call for an OSPRay light array";
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
        case 0: return "GetLight";
        default: return NULL;
        }
    }

    /** Ctor. */
    CallOSPRayLight(void);

    /** Dtor. */
    virtual ~CallOSPRayLight(void);

    /**
    * Assignment operator
    *
    * @param rhs The right hand side operand
    *
    * @return A reference to this
    */
    CallOSPRayLight& operator=(const CallOSPRayLight& rhs);

    /**
    * Sets an add function. The add function is used by the module
    * and sets the lights inside the renderer module.
    *
    * @param std::function<void(std::shared_ptr<OSPRayLightContainer>)> deleg
    *
    * @return 
    */
    void SetDelegate(LightDelegate deleg);

    /**
    * Returns the before setted delegate.
    *
    * @return std::function<void(std::shared_ptr<OSPRayLightContainer>)> addFunction
    */
    LightDelegate CallOSPRayLight::GetDelegate();

    /**
    * Sets the call id
    *
    * @param unsigned int id
    */
    void SetID(unsigned int id);

    /**
    * Returns the call id
    *
    * @return unsigned int id
    */
    unsigned int GetID();

    unsigned int ID;
    LightDelegate addFunction;

};
typedef core::factories::CallAutoDescription<CallOSPRayLight> CallOSPRayLightDescription;

} // namespace ospray
} // namespace megamol