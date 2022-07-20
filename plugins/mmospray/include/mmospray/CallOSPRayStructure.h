/*
 * CallOSPRayStructure.h
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "ParticleDataAccessCollection.h"
#include "mesh/MeshCalls.h"
#include "mmcore/BoundingBoxes.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmospray/CallOSPRayMaterial.h"
#include "mmospray/CallOSPRayTransformation.h"
#include <map>
#include <vector>

namespace megamol {
namespace ospray {

enum structureTypeEnum { UNINITIALIZED, GEOMETRY, VOLUME, OSPRAY_API_STRUCTURES };

enum geometryTypeEnum { SPHERES, MESH, LINES, CURVES, CYLINDERS, TEST };

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

struct sphereStructure {
    std::shared_ptr<ParticleDataAccessCollection> spheres;
};

struct structuredVolumeStructure {
    std::shared_ptr<std::vector<float>> tfRGB;
    std::shared_ptr<std::vector<float>> tfA;
    std::array<float, 2> valueRange;

    const void* voxels;
    std::array<float, 3> gridOrigin;
    std::array<float, 3> gridSpacing;
    std::array<int, 3> dimensions;
    std::array<float, 3> clippingBoxLower;
    std::array<float, 3> clippingBoxUpper;
    float isoValue;
    bool clippingBoxActive;
    volumeRepresentationType volRepType;
    voxelDataType voxelDType;
    unsigned int voxelCount;
    unsigned int maxDim;
};

struct meshStructure {
    std::shared_ptr<mesh::MeshDataAccessCollection> mesh;
    std::shared_ptr<mesh::ImageDataAccessCollection> mesh_textures;
};

struct apiStructure {
    std::pair<std::vector<void*>, structureTypeEnum> ospStructures;
};

struct curveStructure {
    std::shared_ptr<mesh::MeshDataAccessCollection> mesh;

    std::shared_ptr<std::vector<float>> vertexData;
    std::shared_ptr<std::vector<float>> colorData;
    std::shared_ptr<std::vector<unsigned int>> indexData;
    unsigned int vertexLength;
    unsigned int dataStride;
    unsigned int colorLength;
    float globalRadius;
};

struct ClippingPlane {
    std::array<float, 4> coeff = {0, 0, 0, 0};
    bool isValid = false;
};

struct OSPRayStructureContainer {

    structureTypeEnum type = structureTypeEnum::UNINITIALIZED;
    std::shared_ptr<OSPRayMaterialContainer> materialContainer;

    geometryTypeEnum geometryType;
    volumeTypeEnum volumeType;

    std::shared_ptr<OSPRayTransformationContainer> transformationContainer = nullptr;
    bool transformationChanged = false;
    bool dataChanged;
    bool materialChanged;
    bool parameterChanged;
    bool isValid = false;
    ClippingPlane clippingPlane;
    bool clippingPlaneChanged = false;

    std::variant<sphereStructure, structuredVolumeStructure, meshStructure, apiStructure, curveStructure> structure;
};


class OSPRayExtendContainer {
public:
    std::shared_ptr<megamol::core::BoundingBoxes_2> boundingBox;
    unsigned int timeFramesCount;
    bool isValid = false;

    OSPRayExtendContainer() = default;
    ~OSPRayExtendContainer() = default;
};


class CallOSPRayStructure;
typedef std::map<CallOSPRayStructure*, OSPRayStructureContainer> OSPRayStrcutrureMap;
typedef std::map<CallOSPRayStructure*, OSPRayExtendContainer> OSPRayExtendMap;


class CallOSPRayStructure : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallOSPRayStructure";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call for an OSPRay structure";
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

    void setPickResult(uint32_t structure_id, uint32_t prim_id) {
        this->s_id = structure_id;
        this->p_id = prim_id;
    }
    std::tuple<uint32_t, uint32_t> getPickResult() const {
        return std::make_tuple(s_id, p_id);
    }

private:
    OSPRayStrcutrureMap* structureMap;
    float time;
    OSPRayExtendMap* extendMap;
    uint32_t s_id = -1;
    uint32_t p_id = -1;
};
typedef core::factories::CallAutoDescription<CallOSPRayStructure> CallOSPRayStructureDescription;
} // namespace ospray
} // namespace megamol
