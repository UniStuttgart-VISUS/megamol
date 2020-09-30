/*
 * CallOSPRayStructure.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include <map>
#include <vector>
#include "OSPRay_plugin/CallOSPRayMaterial.h"
#include "OSPRay_plugin/OSPRay_plugin.h"
#include "mmcore/BoundingBoxes.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mesh/MeshCalls.h"
#include "CallOSPRayTransformation.h"

namespace megamol {
namespace ospray {

enum structureTypeEnum { UNINITIALIZED, GEOMETRY, VOLUME, OSPRAY_API_STRUCTURES };

enum geometryTypeEnum {
    SPHERES,
    NHSPHERES,
    TRIANGLES,
    STREAMLINES,
    CYLINDERS,
    QUADS
};

enum volumeTypeEnum { STRUCTUREDVOLUME, BLOCKBRICKEDVOLUME, GHOSTBLOCKBRICKEDVOLUME };

enum volumeRepresentationType { VOLUMEREP, ISOSURFACE, SLICE };

enum class voxelDataType : uint8_t { UCHAR = 0, SHORT = 1, USHORT = 2, FLOAT = 3, DOUBLE = 4 };

static std::string voxelDataTypeS[] = {"uchar", "short", "ushort", "float", "double"};

static uint32_t voxelDataTypeOSP[] = {2500, 3000, 3500, 6000, 7000};

// enum class voxelDataTypeOSP {
//    UCHAR = OSP_UCHAR,
//    SHORT = OSP_SHORT,
//    USHORT = OSP_USHORT,
//    FLOAT = OSP_FLOAT,
//    DOUBLE = OSP_DOUBLE
//};


class OSPRayStructureContainer {
public:
    structureTypeEnum type;
    std::shared_ptr<OSPRayMaterialContainer> materialContainer;
    geometryTypeEnum geometryType;
    volumeTypeEnum volumeType;
    volumeRepresentationType volRepType;

    std::shared_ptr<OSPRayTransformationContainer> transformationContainer = nullptr;
    bool transformationChanged = false;

    std::shared_ptr<std::vector<float>> vertexData;
    std::shared_ptr<std::vector<float>> colorData;
    std::shared_ptr<std::vector<unsigned int>> indexData;
    void* voxels;
    std::shared_ptr<std::vector<float>> gridOrigin;
    std::shared_ptr<std::vector<float>> gridSpacing;
    std::shared_ptr<std::vector<int>> dimensions;
    std::shared_ptr<std::vector<float>> clippingBoxLower;
    std::shared_ptr<std::vector<float>> clippingBoxUpper;
    std::shared_ptr<std::vector<float>> isoValue;
    std::shared_ptr<std::vector<float>> sliceData;
    std::shared_ptr<std::vector<float>> clipPlaneData;
    std::shared_ptr<std::vector<float>> clipPlaneColor;
    const void* raw;
    std::shared_ptr<void const*> raw2;
    std::shared_ptr<std::vector<float>> tfRGB;
    std::shared_ptr<std::vector<float>> tfA;
    std::shared_ptr<std::pair<float, float>> valueRange;
    std::shared_ptr<std::vector<float>> xData;
    std::shared_ptr<std::vector<float>> yData;
    std::shared_ptr<std::vector<float>> zData;
    std::shared_ptr<megamol::core::BoundingBoxes> boundingBox; //< TODO data duplicate to extent container ... however,
                                                               // this makes access more concise in the renderer

    std::pair<std::vector<void*>,structureTypeEnum> ospStructures;

    std::shared_ptr<mesh::MeshDataAccessCollection> mesh;
    std::shared_ptr<mesh::ImageDataAccessCollection> mesh_textures;

    unsigned int voxelCount;
    unsigned int maxDim;
    unsigned int vertexCount;
    unsigned int vertexLength;
    unsigned int vertexStride;
    unsigned int colorLength;
    unsigned int colorStride;
    int colorType;
    long long int partCount;
    float globalRadius;
    core::moldyn::SimpleSphericalParticles::ColourDataType mmpldColor;

    bool clippingBoxActive;
    bool dataChanged;
    bool materialChanged;
    bool parameterChanged;
    bool isValid;
    bool smooth; //< valid for lines

    // stuff that should be in OSPRayVolumetricStructure: AbstractOSPRayStructure
    // TODO: both actually.
    bool useMIP;
    bool useGradient;
    bool usePreIntegration;
    bool useAdaptiveSampling;
    float adaptiveFactor;
    float adaptiveMaxRate;

    voxelDataType voxelDType;

    float samplingRate;
    float aoThreshold;
    float aoRayOffsetFactor;

    OSPRayStructureContainer();
    ~OSPRayStructureContainer();
};

class OSPRayExtendContainer {
public:
    std::shared_ptr<megamol::core::BoundingBoxes_2> boundingBox;
    unsigned int timeFramesCount;
    bool isValid;

    OSPRayExtendContainer();
    ~OSPRayExtendContainer();
};


class OSPRAY_PLUGIN_API CallOSPRayStructure;
typedef std::map<CallOSPRayStructure*, OSPRayStructureContainer> OSPRayStrcutrureMap;
typedef std::map<CallOSPRayStructure*, OSPRayExtendContainer> OSPRayExtendMap;


class OSPRAY_PLUGIN_API CallOSPRayStructure : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallOSPRayStructure"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call for an OSPRay structure"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 2; }

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
            return "GetDataCall";
        case 1:
            return "GetExtendsCall";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallOSPRayStructure();

    /** Dtor. */
    virtual ~CallOSPRayStructure(void);

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallOSPRayStructure& operator=(const CallOSPRayStructure& rhs);

    void setStructureMap(OSPRayStrcutrureMap* sm);
    void addStructure(OSPRayStructureContainer& sc);
    bool fillStructureMap();

    void setExtendMap(OSPRayExtendMap* em);
    void addExtend(OSPRayExtendContainer& ec);
    bool fillExtendMap();

    void setTime(float time);
    float getTime();


private:
    OSPRayStrcutrureMap* structureMap;
    float time;
    OSPRayExtendMap* extendMap;
};
typedef core::factories::CallAutoDescription<CallOSPRayStructure> CallOSPRayStructureDescription;
} // namespace ospray
} // namespace megamol
