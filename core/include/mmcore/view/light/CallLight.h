/*
 * CallLight.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <map>
#include <vector>
#include "mmcore/Call.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/String.h"

namespace megamol {
namespace core {
namespace view {
namespace light {

enum MEGAMOLCORE_API lightenum { NONE, DISTANTLIGHT, POINTLIGHT, SPOTLIGHT, QUADLIGHT, AMBIENTLIGHT, HDRILIGHT };

class MEGAMOLCORE_API LightContainer {
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

    LightContainer();
    ~LightContainer();
};

class CallLight;
typedef std::map<CallLight*, LightContainer> LightMap;

class MEGAMOLCORE_API CallLight : public core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallLight"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call for an light array"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 1; }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetCall";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallLight();

    /** Dtor. */
    virtual ~CallLight(void);

    void setLightMap(LightMap* lm);

    void addLight(LightContainer& lc);

    void fillLightMap();

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallLight& operator=(const CallLight& rhs);

private:
    LightMap* lightMap;
};
typedef core::factories::CallAutoDescription<CallLight> CallLightDescription;

} // namespace light
} // namespace view
} // namespace core
} // namespace megamol
