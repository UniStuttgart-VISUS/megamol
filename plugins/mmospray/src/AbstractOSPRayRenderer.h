/*
 * AbstractOSPRayRenderer.h
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/CallOSPRayStructure.h"
#include "mmstd/light/CallLight.h"
#include "mmstd/renderer/CallRender3D.h"
#include "mmstd/renderer/Renderer3DModule.h"
#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
#include <map>
#include <stdint.h>

namespace megamol::ospray {

template<typename tPair>
struct second_t {
    typename tPair::second_type operator()(const tPair& p) const {
        return p.second;
    }
};

template<typename tMap>
second_t<typename tMap::value_type> second(const tMap& m) {
    return second_t<typename tMap::value_type>();
}


inline rkcommon::math::vec2f convertToVec2f(std::array<float, 2> inp) {
    return rkcommon::math::vec2f(inp[0], inp[1]);
}

inline rkcommon::math::vec3f convertToVec3f(std::array<float, 3> inp) {
    return rkcommon::math::vec3f(inp[0], inp[1], inp[2]);
}

inline rkcommon::math::vec3f convertToVec3f(std::array<float, 4> inp) {
    return rkcommon::math::vec3f(inp[0], inp[1], inp[2]);
}

inline rkcommon::math::vec4f convertToVec4f(std::array<float, 4> inp) {
    return rkcommon::math::vec4f(inp[0], inp[1], inp[2], inp[3]);
}

inline rkcommon::math::vec3f convertToVec3f(glm::vec3 inp) {
    return rkcommon::math::vec3f(inp[0], inp[1], inp[2]);
}

inline rkcommon::math::vec4f convertToVec4f(glm::vec4 inp) {
    return rkcommon::math::vec4f(inp[0], inp[1], inp[2], inp[3]);
}

inline rkcommon::math::vec3f convertToVec3f(glm::vec4 inp) {
    return rkcommon::math::vec3f(inp[0], inp[1], inp[2]);
}

struct baseStructureData {
    void emplace_back(::ospray::cpp::Geometry geo, structureTypeEnum type) {
        structures.emplace_back(geo);
        types.emplace_back(type);
    }
    void emplace_back(::ospray::cpp::Volume vol, structureTypeEnum type) {
        structures.emplace_back(vol);
        types.emplace_back(type);
    }
    void emplace_back(OSPGeometry geo, structureTypeEnum type) {
        structures.emplace_back(geo);
        types.emplace_back(type);
    }
    void emplace_back(OSPVolume vol, structureTypeEnum type) {
        structures.emplace_back(vol);
        types.emplace_back(type);
    }
    void clear() {
        structures.clear();
        types.clear();
    }
    size_t size() {
        assert(types.size() == structures.size());
        return structures.size();
    }


    std::vector<structureTypeEnum> types;
    std::vector<std::variant<::ospray::cpp::Geometry, ::ospray::cpp::Volume>> structures;
};


class AbstractOSPRayRenderer : public core::view::Renderer3DModule {
protected:
    // Ctor
    AbstractOSPRayRenderer();

    // Dtor
    ~AbstractOSPRayRenderer() override;

    /**
     * initializes OSPRay
     */
    void initOSPRay();

    /**
     * helper function for initializing OSPRay
     * @param OSPRay renderer object
     * @param OSPRay camera object
     * @param OSPRay world object
     * @param volume name/type
     * @param renderer type
     */
    void setupOSPRay(const char* renderer_name);

    /**
     * helper function for initializing OSPRays Camera
     * @param OSPRay camera object
     * @param CallRenderer3D object
     */
    void setupOSPRayCamera(megamol::core::view::Camera& mmcam);
    void clearOSPRayStuff();


    /**
     * Texture from file importer
     * @param file path
     * @return 2
     */
    ::ospray::cpp::Texture TextureFromFile(vislib::TString fileName);
    // helper function to write the rendered image as PPM file
    void writePPM(std::string fileName, const std::array<int, 2>& size, const uint32_t* pixel);
    void fillMaterialContainer(CallOSPRayStructure* entry_first, const OSPRayStructureContainer& element);

    // TODO: Documentation

    bool AbstractIsDirty();
    void AbstractResetDirty();
    void RendererSettings(glm::vec4 bg_color);

    /**
     * Reads the structure map and uses its parameteres to
     * create geometries and volumes.
     *
     */
    bool generateRepresentations();

    void createInstances();

    void changeMaterial();

    void changeTransformation();

    // Call slots
    megamol::core::CallerSlot _lightSlot;

    // Interface Variables
    core::param::ParamSlot _AOsamples;
    core::param::ParamSlot _AOdistance;
    core::param::ParamSlot _accumulateSlot;

    core::param::ParamSlot _rd_spp;
    core::param::ParamSlot _rd_maxRecursion;
    core::param::ParamSlot _rd_type;
    core::param::ParamSlot _rd_ptBackground;
    core::param::ParamSlot _shadows;
    core::param::ParamSlot _useDB;
    core::param::ParamSlot _numThreads;

    // Fix for deprecated material (ospNewMaterial2 now)
    std::string _rd_type_string;

    megamol::core::param::ParamSlot _deviceTypeSlot;

    // device type
    enum deviceType { DEFAULT, MPI_DISTRIBUTED };

    // renderer type
    enum rdenum { SCIVIS, PATHTRACER, MPI_RAYCAST };

    // light
    std::vector<::ospray::cpp::Light> _lightArray;

    // OSP objects
    std::shared_ptr<::ospray::cpp::FrameBuffer> _framebuffer;
    std::shared_ptr<::ospray::cpp::Camera> _camera;
    std::shared_ptr<::ospray::cpp::World> _world;
    // device
    std::shared_ptr<::ospray::cpp::Device> _device;
    // renderer
    std::shared_ptr<::ospray::cpp::Renderer> _renderer;
    ::ospray::cpp::Texture _maxDepthTexture;

    // structure vectors
    std::map<CallOSPRayStructure*, baseStructureData> _baseStructures;
    std::map<CallOSPRayStructure*, std::vector<::ospray::cpp::VolumetricModel>> _volumetricModels;
    std::map<CallOSPRayStructure*, std::vector<::ospray::cpp::GeometricModel>> _geometricModels;
    std::map<CallOSPRayStructure*, std::vector<::ospray::cpp::GeometricModel>> _clippingModels;

    std::map<CallOSPRayStructure*, ::ospray::cpp::Group> _groups;
    std::map<CallOSPRayStructure*, ::ospray::cpp::Instance> _instances;
    std::map<CallOSPRayStructure*, ::ospray::cpp::Material> _materials;


    // Structure map
    OSPRayStrcutrureMap _structureMap;
    // extend map
    OSPRayExtendMap _extendMap;

    void fillLightArray(std::array<float, 3> eyeDir);

    long long int _ispcLimit = 1ULL << 30;
    long long int _numCreateGeo;

    core::view::Camera::ProjectionType _currentProjectionType = core::view::Camera::ProjectionType::PERSPECTIVE;
};

} // namespace megamol::ospray
