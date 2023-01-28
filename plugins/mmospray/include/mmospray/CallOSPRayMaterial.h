/*
 * CallOSPRayMaterial.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <array>
#include <variant>


namespace megamol::ospray {

//OSPMaterial material;
enum materialTypeEnum { OBJMATERIAL, LUMINOUS, GLASS, MATTE, METAL, METALLICPAINT, PLASTIC, THINGLASS, VELVET };


// OBJMaterial/ScivisMaterial
struct objMaterial {
    std::array<float, 3> Kd;
    std::array<float, 3> Ks;
    float Ns = 0.0f;
    float d = 0.0f;
    std::array<float, 3> Tf;
};

// LUMINOUS
struct luminousMaterial {
    std::array<float, 3> lumColor;
    float lumIntensity = 0.0f;
    float lumTransparency = 0.0f;
};

// VELVET
struct velvetMaterial {
    std::array<float, 3> velvetReflectance;
    float velvetBackScattering = 0.0f;
    std::array<float, 3> velvetHorizonScatteringColor;
    float velvetHorizonScatteringFallOff = 0.0f;
};

// MATTE
struct matteMaterial {
    std::array<float, 3> matteReflectance;
};

// METAL
struct metalMaterial {
    std::array<float, 3> metalReflectance;
    std::array<float, 3> metalEta;
    std::array<float, 3> metalK;
    float metalRoughness = 0.0f;
};

// METALLICPAINT
struct metallicpaintMaterial {
    std::array<float, 3> metallicShadeColor;
    std::array<float, 3> metallicGlitterColor;
    float metallicGlitterSpread = 0.0f;
    float metallicEta = 0.0f;
};

// GLASS
struct glassMaterial {
    float glassEtaInside = 0.0f;
    float glassEtaOutside = 0.0f;
    std::array<float, 3> glassAttenuationColorInside;
    std::array<float, 3> glassAttenuationColorOutside;
    float glassAttenuationDistance = 0.0f;
};

// THINGLASS
struct thinglassMaterial {
    std::array<float, 3> thinglassTransmission;
    float thinglassEta = 0.0f;
    float thinglassThickness = 0.0f;
};

// PLASTIC
struct plasticMaterial {
    std::array<float, 3> plasticPigmentColor;
    float plasticEta = 0.0f;
    float plasticRoughness = 0.0f;
    float plasticThickness = 0.0f;
};


struct OSPRayMaterialContainer {
    materialTypeEnum materialType = materialTypeEnum::OBJMATERIAL;

    std::variant<objMaterial, luminousMaterial, velvetMaterial, matteMaterial, metalMaterial, metallicpaintMaterial,
        glassMaterial, thinglassMaterial, plasticMaterial>
        material;
    bool isValid = false;
};


class CallOSPRayMaterial : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "CallOSPRayMaterial";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call for an OSPRay material";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 1;
    }

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
            return "GetMaterialCall";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallOSPRayMaterial();

    /** Dtor. */
    ~CallOSPRayMaterial() override;

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


} // namespace megamol::ospray
