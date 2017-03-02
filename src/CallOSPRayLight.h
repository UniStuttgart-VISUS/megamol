/*
* CallOSPRayLight.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/Call.h"
#include <vector>
#include <map>
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
    bool isValid;
    bool dataChanged;


    OSPRayLightContainer::OSPRayLightContainer();
    OSPRayLightContainer::~OSPRayLightContainer();
};


class CallOSPRayLight;
typedef std::map<CallOSPRayLight*, OSPRayLightContainer> OSPRayLightMap;



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
        case 0: return "GetCall";
        default: return NULL;
        }
    }

    /** Ctor. */
    CallOSPRayLight();

    /** Dtor. */
    virtual ~CallOSPRayLight(void);

    void setLightMap(OSPRayLightMap *lm);

    void addLight(OSPRayLightContainer &lc);

    void fillLightMap();


    /**
    * Assignment operator
    *
    * @param rhs The right hand side operand
    *
    * @return A reference to this
    */
     CallOSPRayLight& operator=(const CallOSPRayLight& rhs);

private:
    OSPRayLightMap *lightMap;

};
typedef core::factories::CallAutoDescription<CallOSPRayLight> CallOSPRayLightDescription;

} // namespace ospray
} // namespace megamol