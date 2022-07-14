/*
 * AbstractOSPRayRenderer.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "AbstractOSPRayRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/light/AmbientLight.h"
#include "mmcore/view/light/DistantLight.h"
#include "mmcore/view/light/HDRILight.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore/view/light/QuadLight.h"
#include "mmcore/view/light/SpotLight.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemInformation.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdio.h>

namespace megamol {
namespace ospray {

void ospErrorCallback(OSPError err, const char* details) {
    megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRay Error %u: %s", err, details);
}

void ospStatusCallback(const char* msg) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("OSPRay Device Status: %s", msg);
};

AbstractOSPRayRenderer::AbstractOSPRayRenderer(void)
        : core::view::Renderer3DModule()
        , _lightSlot("lights", "Lights are retrieved over this slot. If no light is connected")
        , _accumulateSlot("accumulate", "Activates the accumulation buffer")
        // general renderer parameters
        , _rd_spp("SamplesPerPixel", "Samples per pixel")
        , _rd_maxRecursion("maxRecursion", "Maximum ray recursion depth")
        , _rd_type("Type", "Select between SciVis and PathTracer")
        , _shadows("SciVis::Shadows", "Enables/Disables computation of hard shadows (scivis)")
        // scivis renderer parameters
        , _AOsamples("SciVis::AOsamples", "Number of rays per sample to compute ambient occlusion")
        , _AOdistance("SciVis::AOdistance", "Maximum distance to consider for ambient occlusion")
        // pathtracer renderer parameters
        , _rd_ptBackground(
              "PathTracer::BackgroundTexture", "Texture image used as background, replacing visible lights in infinity")
        // Use depth buffer component
        , _useDB("useDBcomponent", "activates depth composition with OpenGL content")
        , _deviceTypeSlot("device", "Set the type of the OSPRay device")
        , _numThreads("numThreads", "Number of threads used for rendering") {


    this->_lightSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->_lightSlot);

    core::param::EnumParam* rdt = new core::param::EnumParam(SCIVIS);
    rdt->SetTypePair(SCIVIS, "SciVis");
    rdt->SetTypePair(PATHTRACER, "PathTracer");
    rdt->SetTypePair(MPI_RAYCAST, "MPI_Raycast");

    // Ambient parameters
    this->_AOsamples << new core::param::IntParam(1);
    this->_AOdistance << new core::param::FloatParam(1e20f);
    this->_accumulateSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->_AOsamples);
    this->MakeSlotAvailable(&this->_AOdistance);
    this->MakeSlotAvailable(&this->_accumulateSlot);


    // General Renderer
    this->_rd_spp << new core::param::IntParam(1);
    this->_rd_maxRecursion << new core::param::IntParam(10);
    this->_rd_type << rdt;
    this->MakeSlotAvailable(&this->_rd_spp);
    this->MakeSlotAvailable(&this->_rd_maxRecursion);
    this->MakeSlotAvailable(&this->_rd_type);
    this->_shadows << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->_shadows);

    this->_rd_type.ForceSetDirty(); //< TODO HAZARD Dirty hack

    // PathTracer
    this->_rd_ptBackground << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->_rd_ptBackground);

    // Number of threads
    this->_numThreads << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->_numThreads);

    // Depth
    this->_useDB << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->_useDB);

    // Device
    auto deviceEp = new megamol::core::param::EnumParam(deviceType::DEFAULT);
    deviceEp->SetTypePair(deviceType::DEFAULT, "cpu");
    deviceEp->SetTypePair(deviceType::MPI_DISTRIBUTED, "mpi_distributed");
    this->_deviceTypeSlot << deviceEp;
    this->MakeSlotAvailable(&this->_deviceTypeSlot);
}


void AbstractOSPRayRenderer::initOSPRay() {
    if (!_device) {
        ospLoadModule("ispc");
        switch (this->_deviceTypeSlot.Param<megamol::core::param::EnumParam>()->Value()) {
        case deviceType::MPI_DISTRIBUTED: {
            ospLoadModule("mpi");
            _device = std::make_shared<::ospray::cpp::Device>("mpi_distributed");
            _device->setParam("masterRank", 0);
            if (this->_numThreads.Param<megamol::core::param::IntParam>()->Value() > 0) {
                _device->setParam(
                    "numThreads", static_cast<int>(this->_numThreads.Param<megamol::core::param::IntParam>()->Value()));
            }
        } break;
        default: {
            _device = std::make_shared<::ospray::cpp::Device>("cpu");
            if (this->_numThreads.Param<megamol::core::param::IntParam>()->Value() > 0) {
                _device->setParam(
                    "numThreads", static_cast<int>(this->_numThreads.Param<megamol::core::param::IntParam>()->Value()));
            } else {
                _device->setParam("numThreads", static_cast<int>(vislib::sys::SystemInformation::ProcessorCount() - 1));
            }
        }
        }
        _device->setErrorCallback([](void*, OSPError error, const char* errorDetails) {
            ospErrorCallback(error, errorDetails);
            exit(error);
        });
        _device->setStatusCallback([](void*, const char* msg) { ospStatusCallback(msg); });
#ifdef _DEBUG
        _device->setParam("logLevel", OSP_LOG_DEBUG);
#else
        _device->setParam("logLevel", OSP_LOG_INFO);
#endif
        _device->setParam("warnAsError", true);
        _device->commit();
        _device->setCurrent();
    }
    // this->deviceTypeSlot.MakeUnavailable(); //< TODO: Make sure you can set a device only once
}


void AbstractOSPRayRenderer::setupOSPRay(const char* renderer_name) {
    // create and setup renderer
    _renderer = std::make_shared<::ospray::cpp::Renderer>(renderer_name);
    _world = std::make_shared<::ospray::cpp::World>();
}


::ospray::cpp::Texture AbstractOSPRayRenderer::TextureFromFile(vislib::TString fileName) {

    fileName = vislib::sys::Path::Resolve(fileName);

    vislib::TString ext = vislib::TString("");
    size_t pos = fileName.FindLast('.');
    if (pos != std::string::npos)
        ext = fileName.Substring(pos + 1);

    FILE* file = fopen(vislib::StringA(fileName).PeekBuffer(), "rb");
    if (!file)
        throw std::runtime_error("Could not read file");


    if (ext == vislib::TString("ppm")) {
        try {
            int rc, peekchar;

            const int LINESZ = 10000;
            char lineBuf[LINESZ + 1];

            // read format specifier:
            int format = 0;
            rc = fscanf(file, "P%i\n", &format);
            if (format != 6)
                throw std::runtime_error("Wrong PPM format.");

            // skip all comment lines
            peekchar = getc(file);
            while (peekchar == '#') {
                auto tmp = fgets(lineBuf, LINESZ, file);
                (void)tmp;
                peekchar = getc(file);
            }
            ungetc(peekchar, file);

            // read width and height from first non-comment line
            int width = -1, height = -1;
            rc = fscanf(file, "%i %i\n", &width, &height);
            if (rc != 2)
                throw std::runtime_error("Could not read PPM width and height.");

            // skip all comment lines
            peekchar = getc(file);
            while (peekchar == '#') {
                auto tmp = fgets(lineBuf, LINESZ, file);
                (void)tmp;
                peekchar = getc(file);
            }
            ungetc(peekchar, file);

            // read maxval
            int maxVal = -1;
            rc = fscanf(file, "%i", &maxVal);
            peekchar = getc(file);

            unsigned char* data;
            data = new unsigned char[width * height * 3];
            rc = fread(data, width * height * 3, 1, file);
            // flip in y, because OSPRay's textures have the origin at the lower left corner
            unsigned char* texels = (unsigned char*)data;
            for (int y = 0; y < height / 2; y++)
                for (int x = 0; x < width * 3; x++)
                    std::swap(texels[y * width * 3 + x], texels[(height - 1 - y) * width * 3 + x]);

            ::ospray::cpp::Texture ret_tex("texture2d");
            ret_tex.setParam("format", OSP_TEXTURE_RGB8);
            ::ospray::cpp::Data<false> texel_data(texels, OSP_UCHAR, width * height);

            return ret_tex;

        } catch (std::runtime_error e) { std::cerr << e.what() << std::endl; }
    } else {
        std::cerr << "File type not supported. Only PPM file format allowed." << std::endl;
    }
}

bool AbstractOSPRayRenderer::AbstractIsDirty() {
    if (this->_AOsamples.IsDirty() || this->_AOdistance.IsDirty() || this->_accumulateSlot.IsDirty() ||
        this->_shadows.IsDirty() || this->_rd_type.IsDirty() || this->_rd_spp.IsDirty() ||
        this->_rd_maxRecursion.IsDirty() || this->_rd_ptBackground.IsDirty() || this->_useDB.IsDirty()) {
        return true;
    } else {
        return false;
    }
}

void AbstractOSPRayRenderer::AbstractResetDirty() {
    this->_AOsamples.ResetDirty();
    this->_AOdistance.ResetDirty();
    this->_accumulateSlot.ResetDirty();
    this->_shadows.ResetDirty();
    this->_rd_type.ResetDirty();
    this->_rd_spp.ResetDirty();
    this->_rd_maxRecursion.ResetDirty();
    this->_rd_ptBackground.ResetDirty();
    this->_useDB.ResetDirty();
}


void AbstractOSPRayRenderer::fillLightArray(std::array<float, 3> eyeDir) {
    // clear current lights
    _lightArray.clear();

    // create custom ospray light
    ::ospray::cpp::Light light;
    auto lights = core::view::light::LightCollection();

    auto call_light = _lightSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        lights = call_light->getData();
    }

    auto distant_lights = lights.get<core::view::light::DistantLightType>();
    auto point_lights = lights.get<core::view::light::PointLightType>();
    auto spot_lights = lights.get<core::view::light::SpotLightType>();
    auto quad_lights = lights.get<core::view::light::QuadLightType>();
    auto hdri_lights = lights.get<core::view::light::HDRILightType>();
    auto ambient_lights = lights.get<core::view::light::AmbientLightType>();

    for (auto dl : distant_lights) {
        light = ::ospray::cpp::Light("distant");
        if (dl.eye_direction == true) {
            light.setParam("direction", convertToVec3f(eyeDir));
        } else {
            light.setParam("direction", convertToVec3f(dl.direction));
        }
        light.setParam("angularDiameter", dl.angularDiameter);
        light.setParam("intensity", dl.intensity);
        light.setParam("color", convertToVec3f(dl.colour));
        light.commit();
        _lightArray.emplace_back(light);
    }

    for (auto pl : point_lights) {
        light = ::ospray::cpp::Light("point");
        light.setParam("position", convertToVec3f(pl.position));
        light.setParam("radius", pl.radius);
        light.setParam("intensity", pl.intensity);
        light.setParam("color", convertToVec3f(pl.colour));
        light.commit();
        _lightArray.emplace_back(light);
    }

    for (auto sl : spot_lights) {
        light = ::ospray::cpp::Light("spot");
        light.setParam("position", convertToVec3f(sl.position));
        light.setParam("direction", convertToVec3f(sl.direction));
        light.setParam("openingAngle", sl.openingAngle);
        light.setParam("penumbraAngle", sl.penumbraAngle);
        light.setParam("radius", sl.radius);
        light.setParam("intensity", sl.intensity);
        light.setParam("color", convertToVec3f(sl.colour));
        light.commit();
        _lightArray.emplace_back(light);
    }

    for (auto ql : quad_lights) {
        light = ::ospray::cpp::Light("quad");
        light.setParam("position", convertToVec3f(ql.position));
        light.setParam("edge1", convertToVec3f(ql.edgeOne));
        light.setParam("edge2", convertToVec3f(ql.edgeTwo));
        light.setParam("intensity", ql.intensity);
        light.setParam("color", convertToVec3f(ql.colour));
        light.commit();
        _lightArray.emplace_back(light);
    }

    for (auto hl : hdri_lights) {
        light = ::ospray::cpp::Light("hdri");
        light.setParam("up", convertToVec3f(hl.up));
        light.setParam("dir", convertToVec3f(hl.direction));
        if (hl.evnfile != vislib::TString("")) {
            ::ospray::cpp::Texture hdri_tex = this->TextureFromFile(hl.evnfile);
            _renderer->setParam("map_backplate", hdri_tex);
        }
        light.setParam("intensity", hl.intensity);
        light.setParam("color", convertToVec3f(hl.colour));
        light.commit();
        _lightArray.emplace_back(light);
    }

    for (auto al : ambient_lights) {
        light = ::ospray::cpp::Light("ambient");
        light.setParam("intensity", al.intensity);
        //light.setParam("color", convertToVec4f(al.colour));
        light.setParam("color", convertToVec3f(al.colour));
        light.commit();
        _lightArray.emplace_back(light);
    }
}


void AbstractOSPRayRenderer::RendererSettings(glm::vec4 bg_color) {
    // general renderer settings
    _renderer->setParam("pixelSamples", this->_rd_spp.Param<core::param::IntParam>()->Value());
    _renderer->setParam("maxPathLength", this->_rd_maxRecursion.Param<core::param::IntParam>()->Value());

    if (this->_rd_ptBackground.Param<core::param::FilePathParam>()->Value() != "") {
        ::ospray::cpp::Texture background_tex =
            this->TextureFromFile(this->_rd_ptBackground.Param<core::param::FilePathParam>()->Value().string().c_str());
        _renderer->setParam("map_backplate", background_tex);
    } else {
        _renderer->setParam("backgroundColor", convertToVec4f(bg_color));
    }

    switch (this->_rd_type.Param<core::param::EnumParam>()->Value()) {
    case SCIVIS:
        // scivis renderer settings
        _renderer->setParam("aoSamples", this->_AOsamples.Param<core::param::IntParam>()->Value());
        _renderer->setParam("shadows", this->_shadows.Param<core::param::BoolParam>()->Value());
        _renderer->setParam("aoDistance", this->_AOdistance.Param<core::param::FloatParam>()->Value());
        break;
    case PATHTRACER:
        _renderer->setParam("backgroundRefraction", true);
        // TODO: _renderer->setParam("roulettePathLength", );
        break;
    }
}


void AbstractOSPRayRenderer::setupOSPRayCamera(megamol::core::view::Camera& mmcam) {
    // calculate image parts for e.g. screenshooter
    std::array<float, 2> imgStart = {0, 0};
    std::array<float, 2> imgEnd = {0, 0};
    auto cam_pose = mmcam.get<core::view::Camera::Pose>();
    auto cam_proj_type = mmcam.get<core::view::Camera::ProjectionType>();

    if (cam_proj_type == core::view::Camera::ProjectionType::PERSPECTIVE) {
        if (_currentProjectionType != cam_proj_type || !_camera) {
            _camera = std::make_shared<::ospray::cpp::Camera>("perspective");
            _currentProjectionType = core::view::Camera::ProjectionType::PERSPECTIVE;
        }

        auto cam_intrinsics = mmcam.get<core::view::Camera::PerspectiveParameters>();
        imgStart[0] = cam_intrinsics.image_plane_tile.tile_start.x;
        imgStart[1] = cam_intrinsics.image_plane_tile.tile_start.y;
        imgEnd[0] = cam_intrinsics.image_plane_tile.tile_end.x;
        imgEnd[1] = cam_intrinsics.image_plane_tile.tile_end.x;

        _camera->setParam("aspect", static_cast<float>(cam_intrinsics.aspect));
        _camera->setParam("nearClip", static_cast<float>(cam_intrinsics.near_plane));
        _camera->setParam("fovy", glm::degrees(static_cast<float>(cam_intrinsics.fovy)));

    } else if (cam_proj_type == core::view::Camera::ProjectionType::ORTHOGRAPHIC) {
        if (_currentProjectionType != cam_proj_type || !_camera) {
            _camera = std::make_shared<::ospray::cpp::Camera>("orthographic");
            _currentProjectionType = core::view::Camera::ProjectionType::ORTHOGRAPHIC;
        }

        auto cam_intrinsics = mmcam.get<core::view::Camera::OrthographicParameters>();
        imgStart[0] = cam_intrinsics.image_plane_tile.tile_start.x;
        imgStart[1] = cam_intrinsics.image_plane_tile.tile_start.y;
        imgEnd[0] = cam_intrinsics.image_plane_tile.tile_end.x;
        imgEnd[1] = cam_intrinsics.image_plane_tile.tile_end.x;

        _camera->setParam("aspect", static_cast<float>(cam_intrinsics.aspect));
        _camera->setParam("nearClip", static_cast<float>(cam_intrinsics.near_plane));
        _camera->setParam("height", static_cast<float>(cam_intrinsics.frustrum_height));

    } else {
        core::utility::log::Log::DefaultLog.WriteWarn(
            "[AbstractOSPRayRenderer] Projection type not supported. Falling back to perspective.");
        if (_currentProjectionType != cam_proj_type || !_camera) {
            _camera = std::make_shared<::ospray::cpp::Camera>("perspective");
            _currentProjectionType = core::view::Camera::ProjectionType::PERSPECTIVE;
        }

        auto cam_intrinsics = mmcam.get<core::view::Camera::PerspectiveParameters>();
        imgStart[0] = cam_intrinsics.image_plane_tile.tile_start.x;
        imgStart[1] = cam_intrinsics.image_plane_tile.tile_start.y;
        imgEnd[0] = cam_intrinsics.image_plane_tile.tile_end.x;
        imgEnd[1] = cam_intrinsics.image_plane_tile.tile_end.x;

        _camera->setParam("aspect", static_cast<float>(cam_intrinsics.aspect));
        _camera->setParam("nearClip", static_cast<float>(cam_intrinsics.near_plane));
        _camera->setParam("fovy", glm::degrees(static_cast<float>(cam_intrinsics.fovy)));
    } //TODO: Implement panoramic camera


    // setup ospcam
    _camera->setParam("imageStart", convertToVec2f(imgStart));
    _camera->setParam("imageEnd", convertToVec2f(imgEnd));

    _camera->setParam("position", convertToVec3f(cam_pose.position));
    _camera->setParam("direction", convertToVec3f(cam_pose.direction));
    _camera->setParam("up", convertToVec3f(cam_pose.up));

    // ospSet1i(_camera, "architectural", 1);
    // TODO: ospSet1f(_camera, "apertureRadius", );
    // TODO: ospSet1f(_camera, "focalDistance", cr->GetCameraParameters()->FocalDistance());
}

void AbstractOSPRayRenderer::clearOSPRayStuff(void) {
    _lightArray.clear();
    // OSP objects
    _framebuffer.reset();
    _camera.reset();
    _world.reset();
    // device
    _device.reset();
    // renderer
    _renderer.reset();
    // structure vectors
    _baseStructures.clear();
    _volumetricModels.clear();
    _geometricModels.clear();
    _clippingModels.clear();

    _groups.clear();
    _instances.clear();
    _materials.clear();
}

AbstractOSPRayRenderer::~AbstractOSPRayRenderer(void) {}

// helper function to write the rendered image as PPM file
void AbstractOSPRayRenderer::writePPM(std::string fileName, const std::array<int, 2>& size, const uint32_t* pixel) {
    // std::ofstream file;
    // file << "P6\n" << size.x << " " << size.y << "\n255\n";
    FILE* file = fopen(fileName.c_str(), "wb");
    fprintf(file, "P6\n%i %i\n255\n", size[0], size[1]);
    unsigned char* out = (unsigned char*)alloca(3 * size[0]);
    for (int y = 0; y < size[1]; y++) {
        const unsigned char* in = (const unsigned char*)&pixel[(size[1] - 1 - y) * size[1]];
        for (int x = 0; x < size[0]; x++) {
            out[3 * x + 0] = in[4 * x + 0];
            out[3 * x + 1] = in[4 * x + 1];
            out[3 * x + 2] = in[4 * x + 2];
        }
        fwrite(out, 3 * size[0], sizeof(char), file);
    }
    fprintf(file, "\n");
    fclose(file);
}

void AbstractOSPRayRenderer::fillMaterialContainer(
    CallOSPRayStructure* entry_first, const OSPRayStructureContainer& element) {
    switch (element.materialContainer->materialType) {
    case OBJMATERIAL: {
        auto& container = std::get<objMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(this->_rd_type_string.c_str(), "obj");
        _materials[entry_first].setParam("kd", convertToVec3f(container.Kd));
        _materials[entry_first].setParam("ks", convertToVec3f(container.Ks));
        _materials[entry_first].setParam("ns", container.Ns);
        _materials[entry_first].setParam("d", container.d);
        _materials[entry_first].setParam("tf", convertToVec3f(container.Tf));
    } break;
    case LUMINOUS: {
        auto& container = std::get<luminousMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "luminous");
        _materials[entry_first].setParam("color", convertToVec3f(container.lumColor));
        _materials[entry_first].setParam("intensity", container.lumIntensity);
        _materials[entry_first].setParam("transparency", container.lumTransparency);
    } break;
    case GLASS: {
        auto& container = std::get<glassMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "glass");
        _materials[entry_first].setParam("etaInside", container.glassEtaInside);
        _materials[entry_first].setParam("etaOutside", container.glassEtaOutside);
        _materials[entry_first].setParam(
            "attenuationColorInside", convertToVec3f(container.glassAttenuationColorInside));
        _materials[entry_first].setParam(
            "attenuationColorOutside", convertToVec3f(container.glassAttenuationColorOutside));
        _materials[entry_first].setParam("attenuationDistance", container.glassAttenuationDistance);
    } break;
    case MATTE: {
        auto& container = std::get<matteMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "Matte");
        _materials[entry_first].setParam("reflectance", convertToVec3f(container.matteReflectance));
    } break;
    case METAL: {
        auto& container = std::get<metalMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "metal");
        _materials[entry_first].setParam("reflectance", convertToVec3f(container.metalReflectance));
        _materials[entry_first].setParam("eta", convertToVec3f(container.metalEta));
        _materials[entry_first].setParam("k", convertToVec3f(container.metalK));
        _materials[entry_first].setParam("roughness", container.metalRoughness);
    } break;
    case METALLICPAINT: {
        auto& container = std::get<metallicpaintMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "metallicPaint");
        _materials[entry_first].setParam("shadeColor", convertToVec3f(container.metallicShadeColor));
        _materials[entry_first].setParam("glitterColor", convertToVec3f(container.metallicGlitterColor));
        _materials[entry_first].setParam("glitterSpread", container.metallicGlitterSpread);
        _materials[entry_first].setParam("eta", container.metallicEta);
    } break;
    case PLASTIC: {
        auto& container = std::get<plasticMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "Plastic");
        _materials[entry_first].setParam("pigmentColor", convertToVec3f(container.plasticPigmentColor));
        _materials[entry_first].setParam("eta", container.plasticEta);
        _materials[entry_first].setParam("roughness", container.plasticRoughness);
        _materials[entry_first].setParam("thickness", container.plasticThickness);
    } break;
    case THINGLASS: {
        auto& container = std::get<thinglassMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "thinGlass");
        _materials[entry_first].setParam("transmission", convertToVec3f(container.thinglassTransmission));
        _materials[entry_first].setParam("eta", container.thinglassEta);
        _materials[entry_first].setParam("thickness", container.thinglassThickness);
    } break;
    case VELVET: {
        auto& container = std::get<velvetMaterial>(element.materialContainer->material);
        _materials[entry_first] = ::ospray::cpp::Material(_rd_type_string.c_str(), "Velvet");
        _materials[entry_first].setParam("reflectance", convertToVec3f(container.velvetReflectance));
        _materials[entry_first].setParam(
            "horizonScatteringColor", convertToVec3f(container.velvetHorizonScatteringColor));
        _materials[entry_first].setParam("backScattering", container.velvetBackScattering);
        _materials[entry_first].setParam("horizonScatteringFallOff", container.velvetHorizonScatteringFallOff);
    } break;
    }
}

void AbstractOSPRayRenderer::changeMaterial() {

    for (auto entry : this->_structureMap) {
        auto const& element = entry.second;

        // custom material settings
        if (this->_materials[entry.first] != NULL) {
            //ospRelease(this->_materials[entry.first]);
            this->_materials.erase(entry.first);
        }
        if (element.materialContainer != NULL) {
            fillMaterialContainer(entry.first, element);
            _materials[entry.first].commit();
        }

        if (this->_materials[entry.first] != NULL) {
            if (element.type == structureTypeEnum::GEOMETRY) {
                _geometricModels[entry.first].back().setParam(
                    "material", ::ospray::cpp::CopiedData(_materials[entry.first]));
                _geometricModels[entry.first].back().commit();
                _groups[entry.first].setParam("geometry", ::ospray::cpp::CopiedData(_geometricModels[entry.first]));
                _groups[entry.first].commit();
            }
        }
    }
}

void AbstractOSPRayRenderer::changeTransformation() {

    for (auto& entry : this->_baseStructures) {
        if (this->_structureMap[entry.first].transformationContainer == nullptr)
            continue;
        auto trafo = this->_structureMap[entry.first].transformationContainer;
        ::rkcommon::math::affine3f xfm;
        xfm.p.x = trafo->pos[0];
        xfm.p.y = trafo->pos[1];
        xfm.p.z = trafo->pos[2];
        xfm.l.vx.x = trafo->MX[0][0];
        xfm.l.vx.y = trafo->MX[0][1];
        xfm.l.vx.z = trafo->MX[0][2];
        xfm.l.vy.x = trafo->MX[1][0];
        xfm.l.vy.y = trafo->MX[1][1];
        xfm.l.vy.z = trafo->MX[1][2];
        xfm.l.vz.x = trafo->MX[2][0];
        xfm.l.vz.y = trafo->MX[2][1];
        xfm.l.vz.z = trafo->MX[2][2];

        _instances[entry.first].setParam("xfm", xfm);
        _instances[entry.first].commit();
    }
}


bool AbstractOSPRayRenderer::generateRepresentations() {

    bool returnValue = true;

    ::rkcommon::math::box3f _worldBounds;
    std::vector<::rkcommon::math::box3f> ghostRegions;
    std::vector<::rkcommon::math::box3f> regions;

    for (auto& entry : this->_structureMap) {

        _numCreateGeo = 1;
        auto const& element = entry.second;

        // check if structure should be released first
        if (element.dataChanged) {
            //for (int i = 0; i < _baseStructures[entry.first].size(); ++i) {
            //    if (_baseStructures[entry.first].types[i] == structureTypeEnum::GEOMETRY) {
            //        ospRelease(std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures[i]).handle());
            //    } else {
            //        ospRelease(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures[i]).handle());
            //    }
            //}
            _baseStructures[entry.first].clear();
            _baseStructures.erase(entry.first);

            //for (auto& georep : _geometricModels[entry.first]) {
            //    ospRelease(georep.handle());
            //}
            _geometricModels[entry.first].clear();
            _geometricModels.erase(entry.first);

            //for (auto& volrep : _volumetricModels[entry.first]) {
            //    ospRelease(volrep.handle());
            //}
            _volumetricModels[entry.first].clear();
            _volumetricModels.erase(entry.first);

            //for (auto& cliprep : _clippingModels[entry.first]) {
            //    ospRelease(cliprep.handle());
            //}
            _clippingModels[entry.first].clear();
            _clippingModels.erase(entry.first);

            //if (_groups[entry.first]) {
            //    ospRelease(_groups[entry.first].handle());
            //}
            //_groups[entry.first] = nullptr;
            _groups.erase(entry.first);

        } else {
            continue;
        }


        // custom material settings
        if (_materials[entry.first]) {
            _materials.erase(entry.first);
        }
        if (element.materialContainer &&
            this->_rd_type.Param<megamol::core::param::EnumParam>()->Value() != MPI_RAYCAST) {
            fillMaterialContainer(entry.first, element);
            _materials[entry.first].commit();
        }

        switch (element.type) {
        case structureTypeEnum::UNINITIALIZED:
            break;

        case structureTypeEnum::OSPRAY_API_STRUCTURES: {
            auto& container = std::get<apiStructure>(element.structure);
            if (container.ospStructures.first.empty()) {
                // returnValue = false;
                break;
            }

            for (auto structure : container.ospStructures.first) {
                if (container.ospStructures.second == structureTypeEnum::GEOMETRY) {
                    _baseStructures[entry.first].emplace_back(
                        reinterpret_cast<OSPGeometry>(structure), structureTypeEnum::GEOMETRY);
                    _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                        std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));
                    //_geometricModels[entry.first].back().commit();
                } else if (container.ospStructures.second == structureTypeEnum::VOLUME) {
                    _baseStructures[entry.first].emplace_back(
                        reinterpret_cast<OSPVolume>(structure), structureTypeEnum::VOLUME);
                    _volumetricModels[entry.first].emplace_back(::ospray::cpp::VolumetricModel(
                        std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.back())));
                    _volumetricModels[entry.first].back().commit();
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "OSPRAY_API_STRUCTURE: Something went wrong.");
                    break;
                }
            }

            // General geometry execution
            for (unsigned int i = 0; i < container.ospStructures.first.size(); i++) {
                auto idx = _baseStructures[entry.first].size() - 1 - i;
                if (_materials[entry.first] != NULL && _baseStructures[entry.first].size() > 0) {
                    _geometricModels[entry.first][idx].setParam(
                        "material", ::ospray::cpp::CopiedData(_materials[entry.first]));
                    _geometricModels[entry.first][idx].commit();
                }
            }

            _groups[entry.first] = ::ospray::cpp::Group();
            if (!_geometricModels[entry.first].empty()) {
                _groups[entry.first].setParam("geometry", ::ospray::cpp::CopiedData(_geometricModels[entry.first]));
            }
            if (!_volumetricModels[entry.first].empty()) {
                _groups[entry.first].setParam("volume", ::ospray::cpp::CopiedData(_volumetricModels[entry.first]));
            }
            _groups[entry.first].commit();
        } break;
        case structureTypeEnum::GEOMETRY:
            switch (element.geometryType) {
            case geometryTypeEnum::TEST:

                using namespace rkcommon::math;
                using namespace ::ospray::cpp;
                {
                    // triangle mesh data
                    std::vector<vec3f> vertex = {vec3f(-1.0f, -1.0f, 3.0f), vec3f(-1.0f, 1.0f, 3.0f),
                        vec3f(1.0f, -1.0f, 3.0f), vec3f(0.1f, 0.1f, 0.3f)};

                    std::vector<vec4f> color = {vec4f(0.9f, 0.5f, 0.5f, 1.0f), vec4f(0.8f, 0.8f, 0.8f, 1.0f),
                        vec4f(0.8f, 0.8f, 0.8f, 1.0f), vec4f(0.5f, 0.9f, 0.5f, 1.0f)};

                    std::vector<vec3ui> index = {vec3ui(0, 1, 2), vec3ui(1, 2, 3)};

                    // create and setup model and mesh
                    _baseStructures[entry.first].emplace_back(
                        ::ospray::cpp::Geometry("mesh"), structureTypeEnum::GEOMETRY);
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("vertex.position", CopiedData(vertex));
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("vertex.color", CopiedData(color));
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("index", CopiedData(index));
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();

                    // put the mesh into a model
                    _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                        std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));
                    _geometricModels[entry.first].back().commit();
                }

                break;

            case geometryTypeEnum::SPHERES: {
                auto& container = std::get<sphereStructure>(element.structure);
                if (container.spheres == NULL) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRay:generateRepresentations] Representation SPHERES active but no data provided.");
                    // returnValue = false;
                    break;
                }

                _numCreateGeo = container.spheres->accessSphereCollections().size();

                for (auto& spheres : container.spheres->accessSphereCollections()) {
                    _baseStructures[entry.first].emplace_back(
                        ::ospray::cpp::Geometry("sphere"), structureTypeEnum::GEOMETRY);

                    bool radius_found = false;
                    bool color_found = false;

                    for (auto& attrib : spheres.second.attributes) {

                        if (attrib.semantic == ParticleDataAccessCollection::POSITION) {
                            auto count = attrib.byte_size / attrib.stride;

                            auto vertexData = ::ospray::cpp::CopiedData(attrib.data, OSP_VEC3F, count, attrib.stride);
                            vertexData.commit();
                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                .setParam("sphere.position", vertexData);
                        }

                        // check for radius data
                        if (attrib.semantic == ParticleDataAccessCollection::RADIUS) {
                            radius_found = true;
                            auto count = attrib.byte_size / attrib.stride;

                            auto radiusData =
                                ::ospray::cpp::SharedData(&attrib.data[attrib.offset], OSP_FLOAT, count, attrib.stride);
                            radiusData.commit();
                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                .setParam("sphere.radius", radiusData);
                        }
                    }

                    if (!radius_found) {
                        std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                            .setParam("radius", static_cast<float>(spheres.second.global_radius));
                    }
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();
                    _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                        std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));

                    for (auto& attrib : spheres.second.attributes) {

                        // check colorpointer and convert to rgba
                        if (attrib.semantic == ParticleDataAccessCollection::COLOR) {
                            color_found = true;
                            ::ospray::cpp::SharedData colorData;
                            if (attrib.component_type == ParticleDataAccessCollection::ValueType::FLOAT) {
                                auto count = attrib.byte_size / attrib.stride;
                                auto osp_type = OSP_VEC3F;
                                if (attrib.component_cnt == 4)
                                    osp_type = OSP_VEC4F;
                                colorData = ::ospray::cpp::SharedData(
                                    &attrib.data[attrib.offset], osp_type, count, attrib.stride);
                            } else {
                                core::utility::log::Log::DefaultLog.WriteError(
                                    "[OSPRayRenderer][SPHERES] Color type not supported.");
                            }
                            colorData.commit();
                            _geometricModels[entry.first].back().setParam("color", colorData);
                        }
                    }

                    if (!color_found) {
                        auto col = rkcommon::math::vec4f(spheres.second.global_color[0], spheres.second.global_color[1],
                            spheres.second.global_color[2], spheres.second.global_color[3]);
                        _geometricModels[entry.first].back().setParam("color", col);
                    }


                } // end for num geometies
            } break;
            case geometryTypeEnum::MESH: {
                auto& container = std::get<meshStructure>(element.structure);
                if (container.mesh == nullptr) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRay:generateRepresentations] Representation MESH active but no data provided.");
                    // returnValue = false;
                    break;
                }
                {
                    std::shared_ptr<mesh::ImageDataAccessCollection> mesh_texture_collection(nullptr);
                    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_collection(nullptr);

                    _numCreateGeo = 0;

                    // outer loop over mesh collection
                    for (int mc_index = 0; mc_index < container.mesh->size(); ++mc_index) {

                        mesh_collection = (*container.mesh)[mc_index];

                        if (container.mesh_textures != nullptr && container.mesh_textures->size() > mc_index) {
                            mesh_texture_collection = (*container.mesh_textures)[mc_index];
                        } else {
                            mesh_texture_collection = nullptr;
                        }

                        std::vector<mesh::ImageDataAccessCollection::Image> tex_vec;
                        if (mesh_texture_collection != nullptr) {
                            assert(mesh_collection->accessMeshes().size() ==
                                   mesh_texture_collection->accessImages().size());
                            tex_vec = mesh_texture_collection->accessImages();
                        }

                        _numCreateGeo += mesh_collection->accessMeshes().size();

                        // inner loop over meshes per collection
                        uint32_t mesh_index = 0;
                        for (auto const& mesh : mesh_collection->accessMeshes()) {

                            _baseStructures[entry.first].emplace_back(
                                ospNewGeometry("mesh"), structureTypeEnum::GEOMETRY);

                            auto mesh_type = mesh.second.primitive_type;

                            for (auto& attrib : mesh.second.attributes) {

                                if (attrib.semantic == mesh::MeshDataAccessCollection::POSITION) {
                                    auto count = attrib.byte_size /
                                                 (mesh::MeshDataAccessCollection::getByteSize(attrib.component_type) *
                                                     attrib.component_cnt);

                                    auto vertexData =
                                        ::ospray::cpp::SharedData(attrib.data, OSP_VEC3F, count, attrib.stride);
                                    vertexData.commit();
                                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                        .setParam("vertex.position", vertexData);
                                }

                                // check normal pointer
                                if (attrib.semantic == mesh::MeshDataAccessCollection::NORMAL) {
                                    auto count = attrib.byte_size / attrib.stride;
                                    auto normalData =
                                        ::ospray::cpp::SharedData(attrib.data, OSP_VEC3F, count, attrib.stride);
                                    normalData.commit();
                                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                        .setParam("vertex.normal", normalData);
                                }

                                // check colorpointer and convert to rgba
                                if (attrib.semantic == mesh::MeshDataAccessCollection::COLOR) {
                                    ::ospray::cpp::SharedData colorData;
                                    if (attrib.component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT) {
                                        auto count = attrib.byte_size / (mesh::MeshDataAccessCollection::getByteSize(
                                                                             attrib.component_type) *
                                                                            attrib.component_cnt);
                                        auto osp_type = OSP_VEC3F;
                                        if (attrib.component_cnt == 4)
                                            osp_type = OSP_VEC4F;
                                        colorData =
                                            ::ospray::cpp::SharedData(attrib.data, osp_type, count, attrib.stride);
                                    } else {
                                        core::utility::log::Log::DefaultLog.WriteError(
                                            "[OSPRayRenderer][MESH] Color type not supported.");
                                    }
                                    colorData.commit();
                                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                        .setParam("vertex.color", colorData);
                                }

                                // check texture array
                                if (attrib.semantic == mesh::MeshDataAccessCollection::TEXCOORD) {
                                    auto count = attrib.byte_size /
                                                 (mesh::MeshDataAccessCollection::getByteSize(attrib.component_type) *
                                                     attrib.component_cnt);
                                    auto texData =
                                        ::ospray::cpp::SharedData(attrib.data, OSP_VEC2F, count, attrib.stride);
                                    texData.commit();
                                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                        .setParam("vertex.texcoord", texData);
                                }
                            }
                            // check index pointer
                            if (mesh.second.indices.data != nullptr) {
                                auto count = mesh.second.indices.byte_size /
                                             mesh::MeshDataAccessCollection::getByteSize(mesh.second.indices.type);

                                size_t stride = 3 * sizeof(unsigned int);
                                auto osp_type = OSP_VEC3UI;
                                auto divider = 3ull;

                                if (mesh_type == mesh::MeshDataAccessCollection::QUADS) {
                                    stride = 4 * sizeof(unsigned int);
                                    osp_type = OSP_VEC4UI;
                                    divider = 4ull;
                                }
                                count /= divider;

                                auto indexData =
                                    ::ospray::cpp::SharedData(mesh.second.indices.data, osp_type, count, stride);
                                indexData.commit();
                                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                    .setParam("index", indexData);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "OSPRay cannot render meshes without index array");
                                returnValue = false;
                            }

                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();
                            _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));

                            if (container.mesh_textures != nullptr) {
                                auto osp_tex_format = OSP_TEXTURE_FORMAT_INVALID;
                                auto osp_data_format = OSP_BYTE;
                                switch (tex_vec[mesh_index].format) {
                                case mesh::ImageDataAccessCollection::TextureFormat::RGBA8:
                                    osp_tex_format = OSP_TEXTURE_RGBA8;
                                    break;
                                case mesh::ImageDataAccessCollection::TextureFormat::RGB32F:
                                    osp_tex_format = OSP_TEXTURE_RGB32F;
                                    osp_data_format = OSP_FLOAT;
                                    break;
                                case mesh::ImageDataAccessCollection::TextureFormat::RGB8:
                                    osp_tex_format = OSP_TEXTURE_RGB8;
                                    break;
                                case mesh::ImageDataAccessCollection::TextureFormat::RGBA32F:
                                    osp_tex_format = OSP_TEXTURE_RGBA32F;
                                    osp_data_format = OSP_FLOAT;
                                    break;
                                default:
                                    osp_tex_format = OSP_TEXTURE_RGB8;
                                    break;
                                }

                                auto ospTexture = ::ospray::cpp::Texture("texture2d");
                                rkcommon::math::vec2i width_height = {
                                    tex_vec[mesh_index].width, tex_vec[mesh_index].height};

                                auto textureData =
                                    ::ospray::cpp::SharedData(tex_vec[mesh_index].data, osp_data_format, width_height);
                                textureData.commit();

                                ospTexture.setParam("format", osp_tex_format);
                                ospTexture.setParam("data", textureData);
                                ospTexture.commit();

                                auto ospMat = ::ospray::cpp::Material(_rd_type_string.c_str(), "obj");
                                ospMat.setParam("map_Kd", ospTexture);
                                // ospSetObject(ospMat, "map_Ks", ospTexture);
                                // ospSetObject(ospMat, "map_d", ospTexture);
                                ospMat.commit();
                                _geometricModels[entry.first].back().setParam(
                                    "material", ::ospray::cpp::CopiedData(ospMat));
                            }
                            mesh_index++;
                        }
                    }
                }
            } break;
            case geometryTypeEnum::LINES:
            case geometryTypeEnum::CURVES: {
                auto& container = std::get<curveStructure>(element.structure);
                if (container.vertexData == nullptr && container.mesh == nullptr) {
                    // returnValue = false;
                    core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRay:generateRepresentations] Representation CURVES active but no data provided.");
                    break;
                }
                if (container.mesh != nullptr) {

                    std::shared_ptr<mesh::MeshDataAccessCollection> mesh_collection(nullptr);

                    _numCreateGeo = 0;

                    // outer loop over mesh collection
                    for (int mc_index = 0; mc_index < container.mesh->size(); ++mc_index) {
                        mesh_collection = (*container.mesh)[mc_index];

                        this->_numCreateGeo += mesh_collection->accessMeshes().size();
                        for (auto& mesh : mesh_collection->accessMeshes()) {

                            _baseStructures[entry.first].emplace_back(
                                ::ospray::cpp::Geometry("curve"), structureTypeEnum::GEOMETRY);

                            for (auto& attrib : mesh.second.attributes) {

                                if (attrib.semantic == mesh::MeshDataAccessCollection::POSITION) {
                                    size_t count = attrib.byte_size / attrib.stride;
                                    auto vertexData =
                                        ::ospray::cpp::SharedData(attrib.data, OSP_VEC3F, count, attrib.stride);
                                    vertexData.commit();
                                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                        .setParam("vertex.position", vertexData);
                                }

                                // check colorpointer and convert to rgba
                                if (attrib.semantic == mesh::MeshDataAccessCollection::COLOR) {
                                    ::ospray::cpp::SharedData colorData;
                                    if (attrib.component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT) {
                                        size_t count = attrib.byte_size / attrib.stride;
                                        colorData =
                                            ::ospray::cpp::SharedData(attrib.data, OSP_VEC3F, count, attrib.stride);
                                    } else {
                                        core::utility::log::Log::DefaultLog.WriteError(
                                            "[OSPRayRenderer][CURVE] Color type not supported.");
                                    }
                                    colorData.commit();
                                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                        .setParam("vertex.color", colorData);
                                }
                            }
                            // check index pointer
                            if (mesh.second.indices.data != nullptr) {
                                size_t count = mesh.second.indices.byte_size /
                                               mesh::MeshDataAccessCollection::getByteSize(mesh.second.indices.type);
                                auto indexData = ::ospray::cpp::SharedData(mesh.second.indices.data, OSP_UINT, count);
                                indexData.commit();
                                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                    .setParam("index", indexData);
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "OSPRay cannot render curves without index array");
                                returnValue = false;
                            }

                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                .setParam("radius", container.globalRadius);
                            // TODO: Add user input support for this
                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                .setParam("type", OSP_ROUND);
                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                                .setParam("basis", OSP_LINEAR);

                            std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();
                            _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));

                        } // end for geometry
                    }
                } else {
                    _baseStructures[entry.first].emplace_back(
                        ::ospray::cpp::Geometry("curve"), structureTypeEnum::GEOMETRY);

                    this->_numCreateGeo = 1;

                    auto vertexData = ::ospray::cpp::SharedData(
                        container.vertexData->data(), OSP_VEC3F, container.vertexData->size() / 3);
                    vertexData.commit();
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("vertex.position", vertexData);

                    auto indexData = ::ospray::cpp::SharedData(*container.indexData);
                    indexData.commit();
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("index", indexData);

                    if (container.colorData->size() > 0) {
                        auto colorData = ::ospray::cpp::SharedData(
                            container.colorData->data(), OSP_VEC4F, container.colorData->size() / 4);
                        colorData.commit();
                        std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                            .setParam("vertex.color", colorData);
                    }

                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("radius", container.globalRadius);
                    // TODO: Add user input support for this
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("type", OSP_ROUND);
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                        .setParam("basis", OSP_LINEAR);

                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();
                    _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                        std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));
                }
            } break;
            }

            // General geometry execution
            for (unsigned int i = 0; i < this->_numCreateGeo; i++) {
                if (_materials[entry.first] != NULL && _geometricModels[entry.first].size() > 0) {
                    _geometricModels[entry.first].rbegin()[i].setParam(
                        "material", ::ospray::cpp::CopiedData(_materials[entry.first]));
                }

                if (_geometricModels[entry.first].size() > 0) {
                    _geometricModels[entry.first].rbegin()[i].commit();
                }
            }

            _groups[entry.first] = ::ospray::cpp::Group();
            _groups[entry.first].setParam("geometry", ::ospray::cpp::CopiedData(_geometricModels[entry.first]));
            if (entry.second.clippingPlane.isValid) {
                _baseStructures[entry.first].emplace_back(::ospray::cpp::Geometry("plane"), GEOMETRY);

                ::rkcommon::math::vec4f plane;
                plane[0] = entry.second.clippingPlane.coeff[0];
                plane[1] = entry.second.clippingPlane.coeff[1];
                plane[2] = entry.second.clippingPlane.coeff[2];
                plane[3] = entry.second.clippingPlane.coeff[3];
                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                    .setParam("plane.coefficients", ::ospray::cpp::CopiedData(plane));
                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();

                _clippingModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));
                _clippingModels[entry.first].back().commit();

                _groups[entry.first].setParam(
                    "clippingGeometry", ::ospray::cpp::CopiedData(_clippingModels[entry.first]));
            }
            _groups[entry.first].commit();
            break;

        case structureTypeEnum::VOLUME: {
            auto& container = std::get<structuredVolumeStructure>(element.structure);
            if (container.voxels == NULL) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[OSPRay:generateRepresentations] Representation VOLUME active but no data provided.");
                break;
            }

            _baseStructures[entry.first].emplace_back(
                ::ospray::cpp::Volume("structuredRegular"), structureTypeEnum::VOLUME);

            auto type = static_cast<OSPDataType>(voxelDataTypeOSP[static_cast<uint8_t>(container.voxelDType)]);

            // add data
            rkcommon::math::vec3i dims = {container.dimensions[0], container.dimensions[1], container.dimensions[2]};
            auto voxelData = ::ospray::cpp::SharedData(container.voxels, type, dims);
            voxelData.commit();


            std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.back()).setParam("data", voxelData);

            //ospSet3iv(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "dimensions",element.dimensions->data());
            std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.back())
                .setParam("gridOrigin", convertToVec3f(container.gridOrigin));
            std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.back())
                .setParam("gridSpacing", convertToVec3f(container.gridSpacing));


            //std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()).setParam("voxelRange", element.valueRange);
            //ospSet1b(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "singleShade", element.useMIP);
            //ospSet1b(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "gradientShadingEnables",
            //    element.useGradient);
            //ospSet1b(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "preIntegration",
            //    element.usePreIntegration);
            //ospSet1b(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "adaptiveSampling",
            //    element.useAdaptiveSampling);
            //ospSet1f(
            //    std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "adaptiveScalar", element.adaptiveFactor);
            //ospSet1f(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "adaptiveMaxSamplingRate",
            //    element.adaptiveMaxRate);
            //ospSet1f(std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].back()), "samplingRate", element.samplingRate);

            auto tf = ::ospray::cpp::TransferFunction("piecewiseLinear");

            auto tf_rgb = ::ospray::cpp::SharedData(container.tfRGB->data(), OSP_VEC3F, container.tfRGB->size() / 3);
            auto tf_opa = ::ospray::cpp::SharedData(container.tfA->data(), OSP_FLOAT, container.tfA->size());
            tf.setParam("color", tf_rgb);
            tf.setParam("opacity", tf_opa);
            rkcommon::math::vec2f valrange = {container.valueRange[0], container.valueRange[1]};
            tf.setParam("valueRange", valrange);

            tf.commit();

            std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.back()).commit();
            _volumetricModels[entry.first].emplace_back(::ospray::cpp::VolumetricModel(
                std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.back())));

            _volumetricModels[entry.first].back().setParam("transferFunction", tf);
            _volumetricModels[entry.first].back().commit();

            // ClippingBox
            if (container.clippingBoxActive) {
                _baseStructures[entry.first].emplace_back(::ospray::cpp::Geometry("box"), GEOMETRY);

                ::rkcommon::math::box3f box;
                box.lower = {
                    container.clippingBoxLower[0], container.clippingBoxLower[1], container.clippingBoxLower[2]};
                box.upper = {
                    container.clippingBoxUpper[0], container.clippingBoxUpper[1], container.clippingBoxUpper[2]};
                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                    .setParam("box", ::ospray::cpp::CopiedData(box));
                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();

                _clippingModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));
                _clippingModels[entry.first].back().commit();
            }

            switch (container.volRepType) {
            case volumeRepresentationType::VOLUMEREP:
                _groups[entry.first] = ::ospray::cpp::Group();
                _groups[entry.first].setParam("volume", ::ospray::cpp::CopiedData(_volumetricModels[entry.first]));
                if (container.clippingBoxActive) {
                    _groups[entry.first].setParam(
                        "clippingGeometry", ::ospray::cpp::CopiedData(_clippingModels[entry.first]));
                }
                _groups[entry.first].commit();
                break;

            case volumeRepresentationType::ISOSURFACE:
                // isosurface
                _baseStructures[entry.first].emplace_back(::ospray::cpp::Geometry("isosurface"), GEOMETRY);

                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                    .setParam("isovalue", container.isoValue);

                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())
                    .setParam(
                        "volume", std::get<::ospray::cpp::Volume>(_baseStructures[entry.first].structures.front()));

                std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back()).commit();
                _geometricModels[entry.first].emplace_back(::ospray::cpp::GeometricModel(
                    std::get<::ospray::cpp::Geometry>(_baseStructures[entry.first].structures.back())));


                if (_materials[entry.first] != NULL) {
                    _geometricModels[entry.first].back().setParam(
                        "material", ::ospray::cpp::CopiedData(_materials[entry.first]));
                }
                _geometricModels[entry.first].back().commit();

                _groups[entry.first] = ::ospray::cpp::Group();
                _groups[entry.first].setParam("geometry", ::ospray::cpp::CopiedData(_geometricModels[entry.first]));
                if (container.clippingBoxActive) {
                    _groups[entry.first].setParam(
                        "clippingGeometry", ::ospray::cpp::CopiedData(_clippingModels[entry.first]));
                }
                _groups[entry.first].commit();
                break;
            }
        } break;
        }

    } // for element loop

    //if (this->_rd_type.Param<megamol::core::param::EnumParam>()->Value() == MPI_RAYCAST && ghostRegions.size() > 0 &&
    //    regions.size() > 0) {
    //    for (auto const& el : regions) {
    //        ghostRegions.push_back(_worldBounds);
    //    }
    //    auto ghostRegionData = ospNewData(2 * ghostRegions.size(), OSP_FLOAT3, ghostRegions.data());
    //    auto regionData = ospNewData(2 * regions.size(), OSP_FLOAT3, ghostRegions.data());
    //    ospCommit(ghostRegionData);
    //    ospCommit(regionData);
    //    ospSetData(_world, "ghostRegions", ghostRegionData);
    //    ospSetData(_world, "regions", ghostRegionData);
    //}

    return returnValue;
}

void AbstractOSPRayRenderer::createInstances() {

    for (auto& entry : this->_structureMap) {

        /*if (_instances[entry.first]) {
            ospRelease(_instances[entry.first].handle());
        }*/
        _instances.erase(entry.first);

        auto const& element = entry.second;

        _instances[entry.first] = ::ospray::cpp::Instance(_groups[entry.first]);

        if (element.transformationContainer) {

            auto trafo = element.transformationContainer;
            ::rkcommon::math::affine3f xfm;
            xfm.p.x = trafo->pos[0];
            xfm.p.y = trafo->pos[1];
            xfm.p.z = trafo->pos[2];
            xfm.l.vx.x = trafo->MX[0][0];
            xfm.l.vx.y = trafo->MX[0][1];
            xfm.l.vx.z = trafo->MX[0][2];
            xfm.l.vy.x = trafo->MX[1][0];
            xfm.l.vy.y = trafo->MX[1][1];
            xfm.l.vy.z = trafo->MX[1][2];
            xfm.l.vz.x = trafo->MX[2][0];
            xfm.l.vz.y = trafo->MX[2][1];
            xfm.l.vz.z = trafo->MX[2][2];

            _instances[entry.first].setParam("xfm", xfm);
        }

        _instances[entry.first].commit();
    }
}
} // end namespace ospray
} // end namespace megamol
