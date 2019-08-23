/*
 * AbstractOSPRayRenderer.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractOSPRayRenderer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallRender3D.h"
#include "ospcommon/box.h"
#include "ospray/ospray.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemInformation.h"


#include <stdio.h>


using namespace megamol::ospray;

void ospErrorCallback(OSPError err, const char* details) {
    vislib::sys::Log::DefaultLog.WriteError("OSPRay Error %u: %s", err, details);
}

AbstractOSPRayRenderer::AbstractOSPRayRenderer(void)
    : core::view::Renderer3DModule()
    , accumulateSlot("accumulate", "Activates the accumulation buffer")
    ,
    // general renderer parameters
    rd_epsilon("Epsilon", "Ray epsilon to avoid self-intersections")
    , rd_spp("SamplesPerPixel", "Samples per pixel")
    , rd_maxRecursion("maxRecursion", "Maximum ray recursion depth")
    , rd_type("Type", "Select between SciVis and PathTracer")
    , shadows("SciVis::Shadows", "Enables/Disables computation of hard shadows (scivis)")
    ,
    // scivis renderer parameters
    AOtransparencyEnabled("SciVis::AOtransparencyEnabled", "Enables or disables AO transparency")
    , AOsamples("SciVis::AOsamples", "Number of rays per sample to compute ambient occlusion")
    , AOdistance("SciVis::AOdistance", "Maximum distance to consider for ambient occlusion")
    ,
    // pathtracer renderer parameters
    rd_ptBackground(
        "PathTracer::BackgroundTexture", "Texture image used as background, replacing visible lights in infinity")
    ,
    // Call lights
    getLightSlot("getLight", "Connects to a light source")
    ,
    // Use depth buffer component
    useDB("useDBcomponent", "activates depth composition with OpenGL content")
    , deviceTypeSlot("device", "Set the type of the OSPRay device")
    , numThreads("numThreads", "Number of threads used for rendering") {

    // ospray lights
    lightsToRender = NULL;
    this->getLightSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightSlot);
    // ospray device and framebuffer
    device = NULL;
    framebufferIsDirty = true;
    maxDepthTexture = NULL;

    core::param::EnumParam* rdt = new core::param::EnumParam(SCIVIS);
    rdt->SetTypePair(SCIVIS, "SciVis");
    rdt->SetTypePair(PATHTRACER, "PathTracer");
    rdt->SetTypePair(MPI_RAYCAST, "MPI_Raycast");

    // Ambient parameters
    this->AOtransparencyEnabled << new core::param::BoolParam(false);
    this->AOsamples << new core::param::IntParam(1);
    this->AOdistance << new core::param::FloatParam(1e20f);
    this->accumulateSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->AOtransparencyEnabled);
    this->MakeSlotAvailable(&this->AOsamples);
    this->MakeSlotAvailable(&this->AOdistance);
    this->MakeSlotAvailable(&this->accumulateSlot);


    // General Renderer
    this->rd_epsilon << new core::param::FloatParam(1e-4f);
    this->rd_spp << new core::param::IntParam(1);
    this->rd_maxRecursion << new core::param::IntParam(10);
    this->rd_type << rdt;
    this->MakeSlotAvailable(&this->rd_epsilon);
    this->MakeSlotAvailable(&this->rd_spp);
    this->MakeSlotAvailable(&this->rd_maxRecursion);
    this->MakeSlotAvailable(&this->rd_type);
    this->shadows << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->shadows);

    this->rd_type.ForceSetDirty(); //< TODO HAZARD Dirty hack

    // PathTracer
    this->rd_ptBackground << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->rd_ptBackground);

    // Number of threads
    this->numThreads << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->numThreads);

    // Depth
    this->useDB << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useDB);

    // Device
    auto deviceEp = new megamol::core::param::EnumParam(deviceType::DEFAULT);
    deviceEp->SetTypePair(deviceType::DEFAULT, "default");
    deviceEp->SetTypePair(deviceType::MPI_DISTRIBUTED, "mpi_distributed");
    this->deviceTypeSlot << deviceEp;
    this->MakeSlotAvailable(&this->deviceTypeSlot);
}

void AbstractOSPRayRenderer::renderTexture2D(vislib::graphics::gl::GLSLShader& shader, const uint32_t* fb,
    const float* db, int& width, int& height, megamol::core::view::CallRender3D& cr) {

    auto fbo = cr.FrameBufferObject();
    if (fbo != NULL) {

        if (fbo->IsValid()) {
            if ((fbo->GetWidth() != width) || (fbo->GetHeight() != height)) {
                fbo->Release();
            }
        }
        if (!fbo->IsValid()) {
            fbo->Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
        }
        if (fbo->IsValid() && !fbo->IsEnabled()) {
            fbo->Enable();
        }

        fbo->BindColourTexture();
        glClear(GL_COLOR_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);
        glBindTexture(GL_TEXTURE_2D, 0);

        fbo->BindDepthTexture();
        glClear(GL_DEPTH_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, db);
        glBindTexture(GL_TEXTURE_2D, 0);

        if (fbo->IsValid()) {
            fbo->Disable();
            // fbo->DrawColourTexture();
            // fbo->DrawDepthTexture();
        }
    } else {
        /*
        if (this->new_fbo.IsValid()) {
            if ((this->new_fbo.GetWidth() != width) || (this->new_fbo.GetHeight() != height)) {
                this->new_fbo.Release();
            }
        }
        if (!this->new_fbo.IsValid()) {
            this->new_fbo.Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
        vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
        }
        if (this->new_fbo.IsValid() && !this->new_fbo.IsEnabled()) {
            this->new_fbo.Enable();
        }

        this->new_fbo.BindColourTexture();
        glClear(GL_COLOR_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);
        glBindTexture(GL_TEXTURE_2D, 0);

        this->new_fbo.BindDepthTexture();
        glClear(GL_DEPTH_BUFFER_BIT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, db);
        glBindTexture(GL_TEXTURE_2D, 0);


        glBlitNamedFramebuffer(this->new_fbo.GetID(), 0, 0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT, GL_NEAREST);

        this->new_fbo.Disable();
        */
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->depth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, db);
        glBindTexture(GL_TEXTURE_2D, 0);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->tex);
        glUniform1i(shader.ParameterLocation("tex"), 0);


        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->depth);
        glUniform1i(shader.ParameterLocation("depth"), 1);


        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
    }
}


void AbstractOSPRayRenderer::setupTextureScreen() {
    // setup color texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &this->tex);
    glBindTexture(GL_TEXTURE_2D, this->tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    //// setup depth texture
    glGenTextures(1, &this->depth);
    glBindTexture(GL_TEXTURE_2D, this->depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void AbstractOSPRayRenderer::releaseTextureScreen() {
    glDeleteTextures(1, &this->tex);
    glDeleteTextures(1, &this->depth);
}


void AbstractOSPRayRenderer::initOSPRay(OSPDevice& dvce) {
    if (dvce == nullptr) {
        ospLoadModule("ispc");
        switch (this->deviceTypeSlot.Param<megamol::core::param::EnumParam>()->Value()) {
        case deviceType::MPI_DISTRIBUTED: {
            ospLoadModule("mpi");
            dvce = ospNewDevice("mpi_distributed");
            ospDeviceSet1i(dvce, "masterRank", 0);
            if (this->numThreads.Param<megamol::core::param::IntParam>()->Value() > 0) {
                ospDeviceSet1i(dvce, "numThreads", this->numThreads.Param<megamol::core::param::IntParam>()->Value());
            }
        } break;
        default: {
            dvce = ospNewDevice("default");
            if (this->numThreads.Param<megamol::core::param::IntParam>()->Value() > 0) {
                ospDeviceSet1i(dvce, "numThreads", this->numThreads.Param<megamol::core::param::IntParam>()->Value());
            }
            else {
                ospDeviceSet1i(dvce, "numThreads", vislib::sys::SystemInformation::ProcessorCount() - 1);
            }
        }
        }
        ospDeviceSetErrorFunc(dvce, ospErrorCallback);
        ospDeviceCommit(dvce);
        ospSetCurrentDevice(dvce);
    }
    // this->deviceTypeSlot.MakeUnavailable(); //< TODO: Make sure you can set a device only once
}


void AbstractOSPRayRenderer::setupOSPRay(
    OSPRenderer& renderer, OSPCamera& camera, OSPModel& world, const char* renderer_name) {
    // create and setup renderer
    renderer = ospNewRenderer(renderer_name);
    camera = ospNewCamera("perspective");
    world = ospNewModel();
    ospSetObject(renderer, "model", world);
    ospSetObject(renderer, "camera", camera);
}


OSPTexture2D AbstractOSPRayRenderer::TextureFromFile(vislib::TString fileName) {

    fileName = vislib::sys::Path::Resolve(fileName);

    vislib::TString ext = vislib::TString("");
    size_t pos = fileName.FindLast('.');
    if (pos != std::string::npos) ext = fileName.Substring(pos + 1);

    FILE* file = fopen(vislib::StringA(fileName).PeekBuffer(), "rb");
    if (!file) throw std::runtime_error("Could not read file");


    if (ext == vislib::TString("ppm")) {
        try {
            int rc, peekchar;

            const int LINESZ = 10000;
            char lineBuf[LINESZ + 1];

            // read format specifier:
            int format = 0;
            rc = fscanf(file, "P%i\n", &format);
            if (format != 6) throw std::runtime_error("Wrong PPM format.");

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
            if (rc != 2) throw std::runtime_error("Could not read PPM width and height.");

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

            OSPTexture2D ret_tex = ospNewTexture2D({width, height}, OSP_TEXTURE_RGB8, texels);
            return ret_tex;
        } catch (std::runtime_error e) {
            std::cerr << e.what() << std::endl;
        }

    } else if (ext == vislib::TString("pfm")) {
        try {
            // Note: the PFM file specification does not support comments thus we don't skip any
            // http://netpbm.sourceforge.net/doc/pfm.html
            int rc = 0;


            // read format specifier:
            // PF: color floating point image
            // Pf: grayscae floating point image
            char format[2] = {0};
            if (fscanf(file, "%c%c\n", &format[0], &format[1]) != 2) throw std::runtime_error("could not fscanf");

            if (format[0] != 'P' || (format[1] != 'F' && format[1] != 'f')) {
                throw std::runtime_error("Invalid pfm texture file, header is not PF or Pf");
            }
            int numChannels = 3;
            if (format[1] == 'f') {
                numChannels = 1;
            }

            // read width and height
            int width = -1;
            int height = -1;
            rc = fscanf(file, "%i %i\n", &width, &height);
            if (rc != 2) {
                throw std::runtime_error("Could not parse width and height in PF PFM file");
            }

            // read scale factor/endiannes
            float scaleEndian = 0.0;
            rc = fscanf(file, "%f\n", &scaleEndian);

            if (rc != 1) {
                throw std::runtime_error("Could not parse scale factor/endianness in PF PFM file");
            }
            if (scaleEndian == 0.0) {
                throw std::runtime_error("Scale factor/endianness in PF PFM file can not be 0");
            }
            if (scaleEndian > 0.0) {
                throw std::runtime_error("Could not parse PF PFM file");
            }
            float scaleFactor = std::abs(scaleEndian);

            int depth = sizeof(float);
            float* data;
            data = new float[width * height * numChannels];
            if (fread(data, sizeof(float), width * height * numChannels, file) != width * height * numChannels) {
                throw std::runtime_error("could not fread");
            }
            // flip in y, because OSPRay's textures have the origin at the lower left corner
            float* texels = (float*)data;
            for (int y = 0; y < height / 2; ++y) {
                for (int x = 0; x < width * numChannels; ++x) {
                    // Scale the pixels by the scale factor
                    texels[y * width * numChannels + x] = texels[y * width * numChannels + x] * scaleFactor;
                    texels[(height - 1 - y) * width * numChannels + x] =
                        texels[(height - 1 - y) * width * numChannels + x] * scaleFactor;
                    std::swap(texels[y * width * numChannels + x], texels[(height - 1 - y) * width * numChannels + x]);
                }
            }
            OSPTextureFormat type = OSP_TEXTURE_R8;

            if (numChannels == 1) type = OSP_TEXTURE_R32F;
            if (numChannels == 3) type = OSP_TEXTURE_RGB32F;
            if (numChannels == 4) type = OSP_TEXTURE_RGBA32F;

            OSPTexture2D ret_tex = ospNewTexture2D({width, height}, type, texels);
            return ret_tex;
        } catch (std::runtime_error e) {
            std::cerr << e.what() << std::endl;
        }
    } else {
        std::cerr << "File type not supported. Only PPM and PFM file formats allowed." << std::endl;
    }
}

bool AbstractOSPRayRenderer::AbstractIsDirty() {
    if (this->AOsamples.IsDirty() || this->AOtransparencyEnabled.IsDirty() || this->AOdistance.IsDirty() ||
        this->accumulateSlot.IsDirty() || this->shadows.IsDirty() || this->rd_type.IsDirty() ||
        this->rd_epsilon.IsDirty() || this->rd_spp.IsDirty() || this->rd_maxRecursion.IsDirty() ||
        this->rd_ptBackground.IsDirty() || this->useDB.IsDirty() || this->framebufferIsDirty) {
        return true;
    } else {
        return false;
    }
}

void AbstractOSPRayRenderer::AbstractResetDirty() {
    this->AOsamples.ResetDirty();
    this->AOtransparencyEnabled.ResetDirty();
    this->AOdistance.ResetDirty();
    this->accumulateSlot.ResetDirty();
    this->shadows.ResetDirty();
    this->rd_type.ResetDirty();
    this->rd_epsilon.ResetDirty();
    this->rd_spp.ResetDirty();
    this->rd_maxRecursion.ResetDirty();
    this->rd_ptBackground.ResetDirty();
    this->useDB.ResetDirty();
    this->framebufferIsDirty = false;
}


void AbstractOSPRayRenderer::fillLightArray(float* eyeDir) {

    // create custom ospray light
    OSPLight light;

    this->lightArray.clear();

    for (auto const& entry : this->lightMap) {
        auto const& lc = entry.second;

        switch (lc.lightType) {
        case core::view::light::lightenum::NONE:
            light = NULL;
            break;
        case core::view::light::lightenum::DISTANTLIGHT:
            light = ospNewLight(this->renderer, "distant");
            if (lc.dl_eye_direction == true) {
                ospSet3fv(light, "direction", eyeDir);
            } else {
                ospSet3fv(light, "direction", lc.dl_direction.data());
            }
            ospSet1f(light, "angularDiameter", lc.dl_angularDiameter);
            break;
        case core::view::light::lightenum::POINTLIGHT:
            light = ospNewLight(this->renderer, "point");
            ospSet3fv(light, "position", lc.pl_position.data());
            ospSet1f(light, "radius", lc.pl_radius);
            break;
        case core::view::light::lightenum::SPOTLIGHT:
            light = ospNewLight(this->renderer, "spot");
            ospSet3fv(light, "position", lc.sl_position.data());
            ospSet3fv(light, "direction", lc.sl_direction.data());
            ospSet1f(light, "openingAngle", lc.sl_openingAngle);
            ospSet1f(light, "penumbraAngle", lc.sl_penumbraAngle);
            ospSet1f(light, "radius", lc.sl_radius);
            break;
        case core::view::light::lightenum::QUADLIGHT:
            light = ospNewLight(this->renderer, "quad");
            ospSet3fv(light, "position", lc.ql_position.data());
            ospSet3fv(light, "edge1", lc.ql_edgeOne.data());
            ospSet3fv(light, "edge2", lc.ql_edgeTwo.data());
            break;
        case core::view::light::lightenum::HDRILIGHT:
            light = ospNewLight(this->renderer, "hdri");
            ospSet3fv(light, "up", lc.hdri_up.data());
            ospSet3fv(light, "dir", lc.hdri_direction.data());
            if (lc.hdri_evnfile != vislib::TString("")) {
                OSPTexture2D hdri_tex = this->TextureFromFile(lc.hdri_evnfile);
                ospSetObject(this->renderer, "backplate", hdri_tex);
            }
            break;
        case core::view::light::lightenum::AMBIENTLIGHT:
            light = ospNewLight(this->renderer, "ambient");
            break;
        }
        if (lc.isValid && light != NULL) {
            ospSet1f(light, "intensity", lc.lightIntensity);
            ospSet3fv(light, "color", lc.lightColor.data());
            ospCommit(light);
            this->lightArray.push_back(light);
        }
    }
}


void AbstractOSPRayRenderer::RendererSettings(OSPRenderer& renderer) {
    // general renderer settings
    ospSet1f(renderer, "epsilon", this->rd_epsilon.Param<core::param::FloatParam>()->Value());
    ospSet1i(renderer, "spp", this->rd_spp.Param<core::param::IntParam>()->Value());
    ospSet1i(renderer, "maxDepth", this->rd_maxRecursion.Param<core::param::IntParam>()->Value());
    ospSetObject(renderer, "maxDepthTexture", this->maxDepthTexture);

    switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
    case SCIVIS:
        // scivis renderer settings
        ospSet1i(
            renderer, "aoTransparencyEnabled", this->AOtransparencyEnabled.Param<core::param::BoolParam>()->Value());
        ospSet1i(renderer, "aoSamples", this->AOsamples.Param<core::param::IntParam>()->Value());
        ospSet1i(renderer, "shadowsEnabled", this->shadows.Param<core::param::BoolParam>()->Value());
        ospSet1f(renderer, "aoDistance", this->AOdistance.Param<core::param::FloatParam>()->Value());
        //ospSet1i(renderer, "backgroundEnabled", 0);
        
        GLfloat bgcolor[4];
        glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
        ospSet3fv(renderer, "bgColor", bgcolor);
        ospSet1i(renderer, "oneSidedLighting", true);
        
        break;
    case PATHTRACER:
        if (this->rd_ptBackground.Param<core::param::FilePathParam>()->Value() != vislib::TString("")) {
            OSPTexture2D bkgnd_tex =
                this->TextureFromFile(this->rd_ptBackground.Param<core::param::FilePathParam>()->Value());
            ospSetObject(renderer, "backplate", bkgnd_tex);
        } else {
            ospSet1i(renderer, "backgroundEnabled", 0);
        }
        break;
    }
}


void AbstractOSPRayRenderer::setupOSPRayCamera(
    OSPCamera& camera, megamol::core::view::CallRender3D* cr, float scaling) {


    // calculate image parts for e.g. screenshooter
    std::vector<float> imgStart(2, 0);
    std::vector<float> imgEnd(2, 0);
    imgStart[0] = cr->GetCameraParameters()->TileRect().GetLeft() /
                  static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetWidth());
    imgStart[1] = cr->GetCameraParameters()->TileRect().GetBottom() /
                  static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetHeight());

    imgEnd[0] = (cr->GetCameraParameters()->TileRect().GetLeft() + cr->GetCameraParameters()->TileRect().Width()) /
                static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetWidth());
    imgEnd[1] = (cr->GetCameraParameters()->TileRect().GetBottom() + cr->GetCameraParameters()->TileRect().Height()) /
                static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetHeight());

    // setup camera
    ospSet2fv(camera, "imageStart", imgStart.data());
    ospSet2fv(camera, "imageEnd", imgEnd.data());
    ospSetf(camera, "aspect",
        static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetWidth()) /
            static_cast<float>(cr->GetCameraParameters()->VirtualViewSize().GetHeight()));
    // ospSetf(camera, "aspect", cr->GetCameraParameters()->TileRect().AspectRatio());

    // undo scaling
    auto bc = cr->AccessBoundingBoxes().ObjectSpaceBBox().CalcCenter();
    auto mmpos = cr->GetCameraParameters()->EyePosition().PeekCoordinates();
    std::vector<float> ospPos = {mmpos[0] / scaling, //+bc.GetX() / scaling,
        mmpos[1] / scaling,                          //+bc.GetY() / scaling,
        mmpos[2] / scaling};                         //+bc.GetZ() / scaling


    ospSet3fv(camera, "pos", ospPos.data());

    // ospSet3fv(camera, "pos", cr->GetCameraParameters()->EyePosition().PeekCoordinates());
    ospSet3fv(camera, "dir", cr->GetCameraParameters()->EyeDirection().PeekComponents());
    ospSet3fv(camera, "up", cr->GetCameraParameters()->EyeUpVector().PeekComponents());
    ospSet1f(camera, "fovy", cr->GetCameraParameters()->ApertureAngle());

    // ospSet1i(camera, "architectural", 1);
    // ospSet1f(camera, "nearClip", cr->GetCameraParameters()->NearClip());
    // ospSet1f(camera, "farClip", cr->GetCameraParameters()->FarClip());
    // ospSet1f(camera, "apertureRadius", cr->GetCameraParameters()->ApertureAngle);
    // ospSet1f(camera, "focalDistance", cr->GetCameraParameters()->FocalDistance());
}

OSPFrameBuffer AbstractOSPRayRenderer::newFrameBuffer(
    osp::vec2i& imgSize, const OSPFrameBufferFormat format, const uint32_t frameBufferChannels) {
    OSPFrameBuffer frmbuff = ospNewFrameBuffer(imgSize, format, frameBufferChannels);
    this->framebufferIsDirty = true;
    return frmbuff;
}


AbstractOSPRayRenderer::~AbstractOSPRayRenderer(void) {
    if (lightsToRender != NULL) ospRelease(lightsToRender);
    this->Release();
}

// helper function to write the rendered image as PPM file
void AbstractOSPRayRenderer::writePPM(const char* fileName, const osp::vec2i& size, const uint32_t* pixel) {
    // std::ofstream file;
    // file << "P6\n" << size.x << " " << size.y << "\n255\n";
    FILE* file = fopen(fileName, "wb");
    fprintf(file, "P6\n%i %i\n255\n", size.x, size.y);
    unsigned char* out = (unsigned char*)alloca(3 * size.x);
    for (int y = 0; y < size.y; y++) {
        const unsigned char* in = (const unsigned char*)&pixel[(size.y - 1 - y) * size.x];
        for (int x = 0; x < size.x; x++) {
            out[3 * x + 0] = in[4 * x + 0];
            out[3 * x + 1] = in[4 * x + 1];
            out[3 * x + 2] = in[4 * x + 2];
        }
        fwrite(out, 3 * size.x, sizeof(char), file);
    }
    fprintf(file, "\n");
    fclose(file);
}


void AbstractOSPRayRenderer::changeMaterial() {

    for (auto entry : this->structureMap) {
        auto const& element = entry.second;

        // custom material settings
        OSPMaterial material;
        material = NULL;
        if (element.materialContainer != NULL) {
            switch (element.materialContainer->materialType) {
            case OBJMATERIAL:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "OBJMaterial");
                ospSet3fv(material, "Kd", element.materialContainer->Kd.data());
                ospSet3fv(material, "Ks", element.materialContainer->Ks.data());
                ospSet1f(material, "Ns", element.materialContainer->Ns);
                ospSet1f(material, "d", element.materialContainer->d);
                ospSet3fv(material, "Tf", element.materialContainer->Tf.data());
                break;
            case LUMINOUS:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Luminous");
                ospSet3fv(material, "color", element.materialContainer->lumColor.data());
                ospSet1f(material, "intensity", element.materialContainer->lumIntensity);
                ospSet1f(material, "transparency", element.materialContainer->lumTransparency);
                break;
            case GLASS:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Glass");
                ospSet1f(material, "etaInside", element.materialContainer->glassEtaInside);
                ospSet1f(material, "etaOutside", element.materialContainer->glassEtaOutside);
                ospSet3fv(
                    material, "attenuationColorInside", element.materialContainer->glassAttenuationColorInside.data());
                ospSet3fv(material, "attenuationColorOutside",
                    element.materialContainer->glassAttenuationColorOutside.data());
                ospSet1f(material, "attenuationDistance", element.materialContainer->glassAttenuationDistance);
                break;
            case MATTE:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Matte");
                ospSet3fv(material, "reflectance", element.materialContainer->matteReflectance.data());
                break;
            case METAL:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Metal");
                ospSet3fv(material, "reflectance", element.materialContainer->metalReflectance.data());
                ospSet3fv(material, "eta", element.materialContainer->metalEta.data());
                ospSet3fv(material, "k", element.materialContainer->metalK.data());
                ospSet1f(material, "roughness", element.materialContainer->metalRoughness);
                break;
            case METALLICPAINT:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "MetallicPaint");
                ospSet3fv(material, "shadeColor", element.materialContainer->metallicShadeColor.data());
                ospSet3fv(material, "glitterColor", element.materialContainer->metallicGlitterColor.data());
                ospSet1f(material, "glitterSpread", element.materialContainer->metallicGlitterSpread);
                ospSet1f(material, "eta", element.materialContainer->metallicEta);
                break;
            case PLASTIC:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Plastic");
                ospSet3fv(material, "pigmentColor", element.materialContainer->plasticPigmentColor.data());
                ospSet1f(material, "eta", element.materialContainer->plasticEta);
                ospSet1f(material, "roughness", element.materialContainer->plasticRoughness);
                ospSet1f(material, "thickness", element.materialContainer->plasticThickness);
                break;
            case THINGLASS:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "ThinGlass");
                ospSet3fv(material, "transmission", element.materialContainer->thinglassTransmission.data());
                ospSet1f(material, "eta", element.materialContainer->thinglassEta);
                ospSet1f(material, "thickness", element.materialContainer->thinglassThickness);
                break;
            case VELVET:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Velvet");
                ospSet3fv(material, "reflectance", element.materialContainer->velvetReflectance.data());
                ospSet3fv(
                    material, "horizonScatteringColor", element.materialContainer->velvetHorizonScatteringColor.data());
                ospSet1f(material, "backScattering", element.materialContainer->velvetBackScattering);
                ospSet1f(
                    material, "horizonScatteringFallOff", element.materialContainer->velvetHorizonScatteringFallOff);
                break;
            }
            ospCommit(material);
        }

        if (material != NULL) {
            ospSetMaterial(geo.back(), material);
        }
        ospCommit(geo.back());
    }
}


bool AbstractOSPRayRenderer::fillWorld() {

    bool returnValue = true;

    if (this->geo.size() != 0) {
        for (auto element : this->geo) {
            ospRemoveGeometry(this->world, element);
            ospRelease(element);
        }
        this->geo.clear();
    }
    if (this->vol.size() != 0) {
        for (auto element : this->vol) {
            ospRemoveVolume(this->world, element);
            ospRelease(element);
        }
        this->vol.clear();
    }
    // ospRelease(this->world);


    ospcommon::box3f worldBounds;
    std::vector<ospcommon::box3f> ghostRegions;
    std::vector<ospcommon::box3f> regions;

    for (auto entry : this->structureMap) {

        numCreateGeo = 1;
        auto const& element = entry.second;

        // custom material settings
        OSPMaterial material = NULL;
        if (element.materialContainer != NULL &&
            this->rd_type.Param<megamol::core::param::EnumParam>()->Value() != MPI_RAYCAST) {
            switch (element.materialContainer->materialType) {
            case OBJMATERIAL:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "OBJMaterial");
                ospSet3fv(material, "Kd", element.materialContainer->Kd.data());
                ospSet3fv(material, "Ks", element.materialContainer->Ks.data());
                ospSet1f(material, "Ns", element.materialContainer->Ns);
                ospSet1f(material, "d", element.materialContainer->d);
                ospSet3fv(material, "Tf", element.materialContainer->Tf.data());
                break;
            case LUMINOUS:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Luminous");
                ospSet3fv(material, "color", element.materialContainer->lumColor.data());
                ospSet1f(material, "intensity", element.materialContainer->lumIntensity);
                ospSet1f(material, "transparency", element.materialContainer->lumTransparency);
                break;
            case GLASS:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Glass");
                ospSet1f(material, "etaInside", element.materialContainer->glassEtaInside);
                ospSet1f(material, "etaOutside", element.materialContainer->glassEtaOutside);
                ospSet3fv(
                    material, "attenuationColorInside", element.materialContainer->glassAttenuationColorInside.data());
                ospSet3fv(material, "attenuationColorOutside",
                    element.materialContainer->glassAttenuationColorOutside.data());
                ospSet1f(material, "attenuationDistance", element.materialContainer->glassAttenuationDistance);
                break;
            case MATTE:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Matte");
                ospSet3fv(material, "reflectance", element.materialContainer->matteReflectance.data());
                break;
            case METAL:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Metal");
                ospSet3fv(material, "reflectance", element.materialContainer->metalReflectance.data());
                ospSet3fv(material, "eta", element.materialContainer->metalEta.data());
                ospSet3fv(material, "k", element.materialContainer->metalK.data());
                ospSet1f(material, "roughness", element.materialContainer->metalRoughness);
                break;
            case METALLICPAINT:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "MetallicPaint");
                ospSet3fv(material, "shadeColor", element.materialContainer->metallicShadeColor.data());
                ospSet3fv(material, "glitterColor", element.materialContainer->metallicGlitterColor.data());
                ospSet1f(material, "glitterSpread", element.materialContainer->metallicGlitterSpread);
                ospSet1f(material, "eta", element.materialContainer->metallicEta);
                break;
            case PLASTIC:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Plastic");
                ospSet3fv(material, "pigmentColor", element.materialContainer->plasticPigmentColor.data());
                ospSet1f(material, "eta", element.materialContainer->plasticEta);
                ospSet1f(material, "roughness", element.materialContainer->plasticRoughness);
                ospSet1f(material, "thickness", element.materialContainer->plasticThickness);
                break;
            case THINGLASS:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "ThinGlass");
                ospSet3fv(material, "transmission", element.materialContainer->thinglassTransmission.data());
                ospSet1f(material, "eta", element.materialContainer->thinglassEta);
                ospSet1f(material, "thickness", element.materialContainer->thinglassThickness);
                break;
            case VELVET:
                material = ospNewMaterial2(this->rd_type_string.c_str(), "Velvet");
                ospSet3fv(material, "reflectance", element.materialContainer->velvetReflectance.data());
                ospSet3fv(
                    material, "horizonScatteringColor", element.materialContainer->velvetHorizonScatteringColor.data());
                ospSet1f(material, "backScattering", element.materialContainer->velvetBackScattering);
                ospSet1f(
                    material, "horizonScatteringFallOff", element.materialContainer->velvetHorizonScatteringFallOff);
                break;
            }
            ospCommit(material);
        }

        OSPData vertexData = NULL;
        OSPData colorData = NULL;
        OSPData normalData = NULL;
        OSPData texData = NULL;
        OSPData indexData = NULL;
        OSPData voxels = NULL;
        OSPData isovalues = NULL;
        OSPData planes = NULL;
        OSPData xData = NULL;
        OSPData yData = NULL;
        OSPData zData = NULL;
        OSPData bboxData = NULL;
        OSPVolume aovol = NULL;
        OSPError error;

        // OSPPlane pln       = NULL; //TEMPORARILY DISABLED
        switch (element.type) {
        case structureTypeEnum::UNINITIALIZED:
            break;

        case structureTypeEnum::OSPRAY_API_STRUCTURES:
             if (element.ospStructures.first.empty()) {
                // returnValue = false;
                break;
            }
            for (auto structure : element.ospStructures.first) {
                if (element.ospStructures.second == structureTypeEnum::GEOMETRY) {
                    geo.push_back(reinterpret_cast<OSPGeometry>(structure));
                } else if (element.ospStructures.second == structureTypeEnum::VOLUME) {
                    vol.push_back(reinterpret_cast<OSPVolume>(structure));
                } else {
                    vislib::sys::Log::DefaultLog.WriteError("OSPRAY_API_STRUCTURE: Something went wrong.");
                }
            }
            // General geometry execution
            for (unsigned int i = 0; i < element.ospStructures.first.size(); i++) {
                auto idx = geo.size() - 1 - i;
                if (material != NULL && geo.size() > 0) {
                    ospSetMaterial(geo[idx], material);
                }

                if (geo.size() > 0) {
                    ospCommit(geo[idx]);
                    ospAddGeometry(world, geo[idx]);
                }
            }
            break;
        case structureTypeEnum::GEOMETRY:
            switch (element.geometryType) {
             case geometryTypeEnum::SPHERES:
                if (element.vertexData == NULL) {
                    // returnValue = false;
                    break;
                }

                numCreateGeo = element.partCount * element.vertexLength * sizeof(float) / ispcLimit + 1;

                for (unsigned int i = 0; i < numCreateGeo; i++) {
                    geo.push_back(ospNewGeometry("spheres"));

                    long long int vertexFloatsToRead = element.partCount * element.vertexLength / numCreateGeo;
                    vertexFloatsToRead -= vertexFloatsToRead % element.vertexLength;
                    if (vertexData != NULL) ospRelease(vertexData);
                    vertexData = ospNewData(vertexFloatsToRead, OSP_FLOAT,
                        &element.vertexData->operator[](i* vertexFloatsToRead), OSP_DATA_SHARED_BUFFER);

                    ospSet1i(geo.back(), "bytes_per_sphere", element.vertexLength * sizeof(float));

                    if (element.vertexLength > 3) {
                        // ospRemoveParam(geo.back(), "radius");
                        ospSet1f(geo.back(), "offset_radius", 3 * sizeof(float));
                        // TODO: HACK
                        ospSet1f(geo.back(), "radius", 1);
                    } else {
                        ospSet1f(geo.back(), "radius", element.globalRadius);
                    }
                    ospCommit(vertexData);
                    ospSetData(geo.back(), "spheres", vertexData);

                    if (element.colorLength == 4) {
                        long long int colorFloatsToRead = element.partCount * element.colorLength / numCreateGeo;
                        colorFloatsToRead -= colorFloatsToRead % element.colorLength;
                        if (colorData != NULL) ospRelease(colorData);
                        colorData = ospNewData(colorFloatsToRead, OSP_FLOAT,
                            &element.colorData->operator[](i* colorFloatsToRead), OSP_DATA_SHARED_BUFFER);
                        ospCommit(colorData);
                        ospSetData(geo.back(), "color", colorData);
                        // ospSet1i(geo.back(), "color_components", 4);
                        ospSet1i(geo.back(), "color_format", OSP_FLOAT4);
                        // ospSet1i(geo.back(), "color_offset", 0);
                        // ospSet1i(geo.back(), "color_stride", 4 * sizeof(float));
                    }
                }
                // clipPlane setup
                /* TEMPORARILY DISABLED
                if (!std::all_of(element.clipPlaneData->begin(), element.clipPlaneData->end() - 1, [](float i) { return
                i == 0; })) { pln = ospNewPlane("clipPlane"); ospSet1f(pln, "dist", element.clipPlaneData->data()[3]);
                ospSet3fv(pln, "normal", element.clipPlaneData->data());
                ospSet4fv(pln, "color", element.clipPlaneColor->data());
                ospCommit(pln);
                ospSetObject(geo, "clipPlane", pln);
                } else {
                ospSetObject(geo, "clipPlane", NULL);
                }
                */
                break;

            case geometryTypeEnum::NHSPHERES:
                if (element.raw == NULL) {
                    // returnValue = false;
                    break;
                }

                numCreateGeo = element.partCount * element.vertexStride / ispcLimit + 1;

                for (unsigned int i = 0; i < numCreateGeo; i++) {
                    geo.push_back(ospNewGeometry("spheres"));


                    long long int floatsToRead =
                        element.partCount * element.vertexStride / (numCreateGeo * sizeof(float));
                    floatsToRead -= floatsToRead % (element.vertexStride / sizeof(float));

                    if (vertexData != NULL) ospRelease(vertexData);
                    vertexData = ospNewData(floatsToRead, OSP_FLOAT,
                        &static_cast<const float*>(element.raw)[i * floatsToRead], OSP_DATA_SHARED_BUFFER);
                    ospCommit(vertexData);
                    ospSet1i(geo.back(), "bytes_per_sphere", element.vertexStride);
                    ospSetData(geo.back(), "spheres", vertexData);
                    ospSetData(geo.back(), "color", NULL);

                    if (element.vertexLength > 3) {
                        ospSet1f(geo.back(), "offset_radius", 3 * sizeof(float));
                    } else {
                        ospSet1f(geo.back(), "radius", element.globalRadius);
                    }
                    if (element.mmpldColor ==
                            core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB ||
                        element.mmpldColor ==
                            core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGBA) {

                        ospSet1i(geo.back(), "color_offset",
                            element.vertexLength *
                                sizeof(float)); // TODO: This won't work if there are radii in the array
                        ospSet1i(geo.back(), "color_stride", element.colorStride);
                        ospSetData(geo.back(), "color", vertexData);
                        if (element.mmpldColor ==
                            core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB) {
                            // ospSet1i(geo.back(), "color_components", 3);
                            ospSet1i(geo.back(), "color_format", OSP_FLOAT3);
                        } else {
                            // ospSet1i(geo.back(), "color_components", 4);
                            ospSet1i(geo.back(), "color_format", OSP_FLOAT4);
                        }
                    }
                }
                break;
            case geometryTypeEnum::PBS:
                if (element.xData == NULL || element.yData == NULL || element.zData == NULL) {
                    // returnValue = false;
                    break;
                }
                {
                    auto ret = ospLoadModule("ngpf_spheres");
                    if (ret != OSP_NO_ERROR) {
                        vislib::sys::Log::DefaultLog.WriteError("Could not load ngpfSpheres module of OSPRay");
                        throw std::runtime_error("Could not load ngpfSpheres module of OSPRay");
                    }
                }
                geo.push_back(ospNewGeometry("ngpf_spheres"));

                {

                    xData = ospNewData(element.partCount, OSP_FLOAT, element.xData->data());
                    yData = ospNewData(element.partCount, OSP_FLOAT, element.yData->data());
                    zData = ospNewData(element.partCount, OSP_FLOAT, element.zData->data());

                    ospCommit(xData);
                    ospCommit(yData);
                    ospCommit(zData);

                    ospSetData(geo.back(), "x_data", xData);
                    ospSetData(geo.back(), "y_data", yData);
                    ospSetData(geo.back(), "z_data", zData);

                    ospSet1f(geo.back(), "radius", element.globalRadius);
                }
                break;
            case geometryTypeEnum::TRIANGLES:
                if (element.vertexData == NULL) {
                    // returnValue = false;
                    break;
                }

                geo.push_back(ospNewGeometry("triangles"));

                // check vertex data type
                if (element.vertexData->size() != 0) {
                    vertexData = ospNewData(element.vertexCount, OSP_FLOAT3, element.vertexData->data());
                    ospCommit(vertexData);
                    ospSetData(geo.back(), "vertex", vertexData);
                } else {
                    vislib::sys::Log::DefaultLog.WriteError("OSPRay cannot render meshes without vertex array");
                    returnValue = false;
                }

                // check normal pointer
                if (element.normalData->size() != 0) {
                    normalData = ospNewData(element.vertexCount, OSP_FLOAT3, element.normalData->data());
                    ospCommit(normalData);
                    ospSetData(geo.back(), "vertex.normal", normalData);
                }

                // check colorpointer and convert to rgba
                if (element.colorData->size() != 0) {
                    colorData = ospNewData(element.vertexCount, OSP_FLOAT4, element.colorData->data());
                    ospCommit(colorData);
                    ospSetData(geo.back(), "vertex.color", colorData);
                }

                // check texture array
                if (element.texData->size() != 0) {
                    texData = ospNewData(element.triangleCount, OSP_FLOAT2, element.texData->data());
                    ospCommit(texData);
                    ospSetData(geo.back(), "vertex.texcoord", texData);
                }

                // check index pointer
                if (element.indexData->size() != 0) {
                    indexData = ospNewData(element.triangleCount, OSP_INT3, element.indexData->data());
                    ospCommit(indexData);
                    ospSetData(geo.back(), "index", indexData);
                } else {
                    vislib::sys::Log::DefaultLog.WriteError("OSPRay cannot render meshes without index array");
                    returnValue = false;
                }

                break;
            case geometryTypeEnum::STREAMLINES:
                if (element.vertexData == NULL) {
                    // returnValue = false;
                    break;
                }
                {
                    geo.push_back(ospNewGeometry("streamlines"));

                    osp::vec3fa* data = new osp::vec3fa[element.vertexData->size() / 3];

                    // fill aligned array with vertex data
                    for (unsigned int i = 0; i < element.vertexData->size() / 3; i++) {
                        data[i].x = element.vertexData->data()[3 * i + 0];
                        data[i].y = element.vertexData->data()[3 * i + 1];
                        data[i].z = element.vertexData->data()[3 * i + 2];
                    }


                    vertexData = ospNewData(element.vertexData->size() / 3, OSP_FLOAT3A, data, OSP_DATA_SHARED_BUFFER);
                    ospCommit(vertexData);
                    ospSetData(geo.back(), "vertex", vertexData);

                    indexData = ospNewData(
                        element.indexData->size(), OSP_INT, element.indexData->data(), OSP_DATA_SHARED_BUFFER);
                    ospCommit(indexData);
                    ospSetData(geo.back(), "index", indexData);

                    if (element.colorData->size() > 0) {
                        colorData = ospNewData(element.colorData->size() / element.colorLength, OSP_FLOAT4,
                            element.colorData->data(), OSP_DATA_SHARED_BUFFER);
                        ospCommit(colorData);
                        ospSetData(geo.back(), "vertex.color", colorData);
                    }

                    ospSet1f(geo.back(), "radius", element.globalRadius);
                    ospSet1i(geo.back(), "smooth", element.smooth);
                }
                break;
            case geometryTypeEnum::CYLINDERS:
                break;
            }

            // General geometry execution
            for (unsigned int i = 0; i < this->numCreateGeo; i++) {
                if (material != NULL && geo.size() > 0) {
                    ospSetMaterial(geo.rbegin()[i], material);
                }

                if (geo.size() > 0) {
                    ospCommit(geo.rbegin()[i]);
                    ospAddGeometry(world, geo.rbegin()[i]);
                }
            }

            if (vertexData != NULL) ospRelease(vertexData);
            if (colorData != NULL) ospRelease(colorData);
            if (normalData != NULL) ospRelease(normalData);
            if (texData != NULL) ospRelease(texData);
            if (indexData != NULL) ospRelease(indexData);
            if (xData != NULL) ospRelease(xData);
            if (yData != NULL) ospRelease(yData);
            if (zData != NULL) ospRelease(zData);
            if (bboxData != NULL) ospRelease(bboxData);
            if (aovol != NULL) ospRelease(aovol);
            if (voxels != NULL) ospRelease(voxels);

            break;

        case structureTypeEnum::VOLUME:

                if (element.voxels == NULL) {
                    // returnValue = false;
                    break;
                }

                vol.push_back(ospNewVolume("shared_structured_volume"));

                auto type = static_cast<uint8_t>(element.voxelDType);

                ospSetString(vol.back(), "voxelType", voxelDataTypeS[type].c_str());
                //float fixedSpacing[3];
                //for (auto x = 0; x < 3; ++x) {
                //    fixedSpacing[x] = element.gridSpacing->at(x) / (element.dimensions->at(x) - 1) + element.gridSpacing->at(x);
                //}
                // scaling properties of the volume
                ospSet3iv(vol.back(), "dimensions", element.dimensions->data());
                ospSet3fv(vol.back(), "gridOrigin", element.gridOrigin->data());
                ospSet3fv(vol.back(), "gridSpacing", element.gridSpacing->data());
                ospSet2f(vol.back(), "voxelRange", element.valueRange->first, element.valueRange->second);

                ospSet1b(vol.back(), "singleShade", element.useMIP);
                ospSet1b(vol.back(), "gradientShadingEnables", element.useGradient);
                ospSet1b(vol.back(), "preIntegration", element.usePreIntegration);
                ospSet1b(vol.back(), "adaptiveSampling", element.useAdaptiveSampling);
                ospSet1f(vol.back(), "adaptiveScalar", element.adaptiveFactor);
                ospSet1f(vol.back(), "adaptiveMaxSamplingRate", element.adaptiveMaxRate);
                ospSet1f(vol.back(), "samplingRate", element.samplingRate);

                // add data
                voxels = ospNewData(element.voxelCount, static_cast<OSPDataType>(voxelDataTypeOSP[type]),
                    element.voxels, OSP_DATA_SHARED_BUFFER);
                ospCommit(voxels);
                ospSetData(vol.back(), "voxelData", voxels);

                // ClippingBox

                if (element.clippingBoxActive) {
                    ospSet3fv(vol.back(), "volumeClippingBoxLower", element.clippingBoxLower->data());
                    ospSet3fv(vol.back(), "volumeClippingBoxUpper", element.clippingBoxUpper->data());
                } else {
                    ospSetVec3f(vol.back(), "volumeClippingBoxLower", {0.0f, 0.0f, 0.0f});
                    ospSetVec3f(vol.back(), "volumeClippingBoxUpper", {0.0f, 0.0f, 0.0f});
                }

                OSPTransferFunction tf = ospNewTransferFunction("piecewise_linear");

                OSPData tf_rgb = ospNewData(element.tfRGB->size() / 3, OSP_FLOAT3, element.tfRGB->data());
                OSPData tf_opa = ospNewData(element.tfA->size(), OSP_FLOAT, element.tfA->data());
                ospSetData(tf, "colors", tf_rgb);
                ospSetData(tf, "opacities", tf_opa);
                ospSet2f(tf, "valueRange", element.valueRange->first, element.valueRange->second);

                ospCommit(tf);

                ospSetObject(vol.back(), "transferFunction", tf);
                ospCommit(vol.back());
                ospRelease(tf);
            
            switch (element.volRepType) {
            case volumeRepresentationType::VOLUMEREP:
                ospAddVolume(world, vol.back());
                break;

            case volumeRepresentationType::ISOSURFACE:
                // isosurface
                geo.push_back(ospNewGeometry("isosurfaces"));
                isovalues = ospNewData(1, OSP_FLOAT, element.isoValue->data());
                ospCommit(isovalues);
                ospSetData(geo.back(), "isovalues", isovalues);
                ospSetObject(geo.back(), "volume", vol.back());

                if (material != NULL) {
                    ospSetMaterial(geo.back(), material);
                }

                ospCommit(geo.back());

                ospAddGeometry(world, geo.back()); // Show isosurface

                break;

            case volumeRepresentationType::SLICE:
                geo.push_back(ospNewGeometry("slices"));
                planes = ospNewData(1, OSP_FLOAT4, element.sliceData->data());
                ospCommit(planes);
                ospSetData(geo.back(), "planes", planes);
                ospSetObject(geo.back(), "volume", vol.back());

                if (material != NULL) {
                    ospSetMaterial(geo.back(), material);
                }

                ospCommit(geo.back());

                ospAddGeometry(world, geo.back()); // Show slice

                break;
            }

            if (voxels != NULL) ospRelease(voxels);
            if (planes != NULL) ospRelease(planes);
            if (isovalues != NULL) ospRelease(isovalues);

            break;
        }

    } // for element loop

    if (this->rd_type.Param<megamol::core::param::EnumParam>()->Value() == MPI_RAYCAST && ghostRegions.size() > 0 &&
        regions.size() > 0) {
        for (auto const& el : regions) {
            ghostRegions.push_back(worldBounds);
        }
        auto ghostRegionData = ospNewData(2 * ghostRegions.size(), OSP_FLOAT3, ghostRegions.data());
        auto regionData = ospNewData(2 * regions.size(), OSP_FLOAT3, ghostRegions.data());
        ospCommit(ghostRegionData);
        ospCommit(regionData);
        ospSetData(world, "ghostRegions", ghostRegionData);
        ospSetData(world, "regions", ghostRegionData);
    }

    return returnValue;
}

void AbstractOSPRayRenderer::releaseOSPRayStuff() {}
