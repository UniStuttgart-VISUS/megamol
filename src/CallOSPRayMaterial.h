/*
* CallOSPRayMaterial.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/Call.h"
#include <vector>

namespace megamol {
namespace ospray {

//OSPMaterial material;
enum materialTypeEnum {
    OBJMATERIAL,
    LUMINOUS,
    GLASS,
    MATTE,
    METAL,
    METALLICPAINT,
    PLASTIC,
    THINGLASS,
    VELVET
};

class OSPRayMaterialContainer {
public:
    materialTypeEnum materialType;
    // OBJMaterial/ScivisMaterial
    std::vector<float> Kd;
    std::vector<float> Ks;
    float Ns;
    float d;
    std::vector<float> Tf;
    // LUMINOUS
    std::vector<float> lumColor;
    float lumIntensity;
    float lumTransparency;
    // VELVET
    std::vector<float> velvetReflectance;
    float velvetBackScattering;
    std::vector<float> velvetHorizonScatteringColor;
    float velvetHorizonScatteringFallOff;
    // MATTE
    std::vector<float> matteReflectance;
    // METAL
    std::vector<float> metalReflectance;
    std::vector<float> metalEta;
    std::vector<float> metalK;
    float metalRoughness;
    // METALLICPAINT
    std::vector<float> metallicShadeColor;
    std::vector<float> metallicGlitterColor;
    float metallicGlitterSpread;
    float metallicEta;
    // GLASS
    float glassEtaInside;
    float glassEtaOutside;
    std::vector<float> glassAttenuationColorInside;
    std::vector<float> glassAttenuationColorOutside;
    float glassAttenuationDistance;
    //THINGLASS
    std::vector<float> thinglassTransmission;
    float thinglassEta;
    float thinglassThickness;
    // PLASTIC
    std::vector<float> plasticPigmentColor;
    float plasticEta;
    float plasticRoughness;
    float plasticThickness;

    bool isValid;

    OSPRayMaterialContainer();
    ~OSPRayMaterialContainer();

};


class CallOSPRayMaterial : public megamol::core::Call {
public:

    /**
    * Answer the name of the objects of this description.
    *
    * @return The name of the objects of this description.
    */
    static const char *ClassName(void) {
        return "CallOSPRayMaterial";
    }

    /**
    * Gets a human readable description of the module.
    *
    * @return A human readable description of the module.
    */
    static const char *Description(void) {
        return "Call for an OSPRay material";
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
        case 0: return "GetMaterialCall";
        default: return NULL;
        }
    }

    /** Ctor. */
    CallOSPRayMaterial(void);

    /** Dtor. */
    virtual ~CallOSPRayMaterial(void);

    void setMaterialContainer(std::shared_ptr<OSPRayMaterialContainer> mc);
    std::shared_ptr<OSPRayMaterialContainer> getMaterialParameter();

    /**
    * Assignment operator
    *
    * @param rhs The right hand side operand
    *
    * @return A reference to this
    */
    CallOSPRayMaterial& operator=(const CallOSPRayMaterial& rhs);

    bool InterfaceIsDirty();
    void setDirty();

private:
    std::shared_ptr<OSPRayMaterialContainer> materialContainer;
    bool isDirty;

};
typedef core::factories::CallAutoDescription<CallOSPRayMaterial> CallOSPRayMaterialDescription;


} // namespace ospray
} // namespace megamol