/*
* OSPRaySphereRenderer.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"

namespace megamol {
namespace ospray {

class OSPRaySphereGeometry : core::Module {

public:

    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRaySphereGeometry";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Creator for OSPRay sphere geometries.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor. */
    virtual ~OSPRaySphereGeometry(void);

    /** Ctor. */
    OSPRaySphereGeometry(void);

protected:
    /**
    * color transfer helper
    * @param array with gray scales
    * @param transferfunction table/texture
    * @param transferfunction table/texture size
    * @param target array (rgba)
    */
    void colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray);

    virtual bool create();
    virtual void release();

    bool readData(core::Call &call);

    bool getStructureCallback(core::Call& call);
    bool checkDatahashCallback(core::Call& call);
    bool InterfaceIsDirty();
    void OSPRaySphereGeometry::getClipData(float *clipDat, float *clipCol);

    // material
    // OBJMaterial/ScivisMaterial
    core::param::ParamSlot Kd;
    core::param::ParamSlot Ks;
    core::param::ParamSlot Ns;
    core::param::ParamSlot d;
    core::param::ParamSlot Tf;
    core::param::ParamSlot materialType;
    // LUMINOUS
    core::param::ParamSlot lumColor;
    core::param::ParamSlot lumIntensity;
    core::param::ParamSlot lumTransparency;
    // VELVET
    core::param::ParamSlot velvetReflectance;
    core::param::ParamSlot velvetBackScattering;
    core::param::ParamSlot velvetHorizonScatteringColor;
    core::param::ParamSlot velvetHorizonScatteringFallOff;
    // MATTE
    core::param::ParamSlot matteReflectance;
    // METAL
    core::param::ParamSlot metalReflectance;
    core::param::ParamSlot metalEta;
    core::param::ParamSlot metalK;
    core::param::ParamSlot metalRoughness;
    // METALLICPAINT
    core::param::ParamSlot metallicShadeColor;
    core::param::ParamSlot metallicGlitterColor;
    core::param::ParamSlot metallicGlitterSpread;
    core::param::ParamSlot metallicEta;
    // GLASS
    core::param::ParamSlot glassEtaInside;
    core::param::ParamSlot glassEtaOutside;
    core::param::ParamSlot glassAttenuationColorInside;
    core::param::ParamSlot glassAttenuationColorOutside;
    core::param::ParamSlot glassAttenuationDistance;
    //THINGLASS
    core::param::ParamSlot thinglassTransmission;
    core::param::ParamSlot thinglassEta;
    core::param::ParamSlot thinglassThickness;
    // PLASTIC
    core::param::ParamSlot plasticPigmentColor;
    core::param::ParamSlot plasticEta;
    core::param::ParamSlot plasticRoughness;
    core::param::ParamSlot plasticThickness;





    core::param::ParamSlot particleList;




    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The call for clipping plane */
    core::CallerSlot getClipPlaneSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    /** The callee for Structure */
    core::CalleeSlot deployStructureSlot;

    /** The call for additional structures */
    core::CallerSlot getStructureSlot;


private:


    //data objects
    std::vector<float> cd_rgba;
    std::vector<float> vd;

    // data conversion
    size_t vertexLength;
    size_t colorLength;
    // color transfer data
    unsigned int tex_size;
    SIZE_T datahash;

};

} // namespace ospray
} // namespace megamol