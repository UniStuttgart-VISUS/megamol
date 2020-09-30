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
#include "ospcommon/box.h"
#include "ospray/ospray.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/sys/Path.h"
#include "mmcore/utility/sys/SystemInformation.h"


#include <stdio.h>


namespace megamol {
namespace ospray {

void ospErrorCallback(OSPError err, const char* details) {
    megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRay Error %u: %s", err, details);
}

AbstractOSPRayRenderer::AbstractOSPRayRenderer(void)
    : core::view::Renderer3DModule_2()
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
    // Use depth buffer component
    useDB("useDBcomponent", "activates depth composition with OpenGL content")
    , deviceTypeSlot("device", "Set the type of the OSPRay device")
    , numThreads("numThreads", "Number of threads used for rendering") {

    // ospray lights
    lightsToRender = NULL;
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
    const float* db, int& width, int& height, megamol::core::view::CallRender3D_2& cr) {

    auto fbo = cr.FrameBufferObject();
    // if (fbo != NULL) {

    //    if (fbo->IsValid()) {
    //        if ((fbo->GetWidth() != width) || (fbo->GetHeight() != height)) {
    //            fbo->Release();
    //        }
    //    }
    //    if (!fbo->IsValid()) {
    //        fbo->Create(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
    //            vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE, GL_DEPTH_COMPONENT);
    //    }
    //    if (fbo->IsValid() && !fbo->IsEnabled()) {
    //        fbo->Enable();
    //    }

    //    fbo->BindColourTexture();
    //    glClear(GL_COLOR_BUFFER_BIT);
    //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);
    //    glBindTexture(GL_TEXTURE_2D, 0);

    //    fbo->BindDepthTexture();
    //    glClear(GL_DEPTH_BUFFER_BIT);
    //    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, db);
    //    glBindTexture(GL_TEXTURE_2D, 0);

    //    if (fbo->IsValid()) {
    //        fbo->Disable();
    //        // fbo->DrawColourTexture();
    //        // fbo->DrawDepthTexture();
    //    }
    //} else {
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
    //  }
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
            } else {
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


void AbstractOSPRayRenderer::fillLightArray(glm::vec4& eyeDir) {

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
                ospSet3f(light, "direction", eyeDir.x, eyeDir.y, eyeDir.z);
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
        // ospSet1i(renderer, "backgroundEnabled", 0);

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


void AbstractOSPRayRenderer::setupOSPRayCamera(OSPCamera& ospcam, megamol::core::view::Camera_2& mmcam) {


    // calculate image parts for e.g. screenshooter
    std::vector<float> imgStart(2, 0);
    std::vector<float> imgEnd(2, 0);
    imgStart[0] = mmcam.image_tile().left() / static_cast<float>(mmcam.resolution_gate().width());
    imgStart[1] = mmcam.image_tile().bottom() / static_cast<float>(mmcam.resolution_gate().height());

    imgEnd[0] =
        (mmcam.image_tile().left() + mmcam.image_tile().width()) / static_cast<float>(mmcam.resolution_gate().width());
    imgEnd[1] = (mmcam.image_tile().bottom() + mmcam.image_tile().height()) /
                static_cast<float>(mmcam.resolution_gate().height());

    // setup ospcam
    ospSet2fv(ospcam, "imageStart", imgStart.data());
    ospSet2fv(ospcam, "imageEnd", imgEnd.data());
    ospSetf(ospcam, "aspect", mmcam.resolution_gate_aspect());


    ospSet3f(ospcam, "pos", mmcam.eye_position().x(), mmcam.eye_position().y(), mmcam.eye_position().z());
    ospSet3f(ospcam, "dir", mmcam.view_vector().x(), mmcam.view_vector().y(), mmcam.view_vector().z());
    ospSet3f(ospcam, "up", mmcam.up_vector().x(), mmcam.up_vector().y(), mmcam.up_vector().z());
    ospSet1f(ospcam, "fovy", mmcam.aperture_angle());

    // ospSet1i(ospcam, "architectural", 1);
    ospSet1f(ospcam, "nearClip", mmcam.near_clipping_plane());
    ospSet1f(ospcam, "farClip", mmcam.far_clipping_plane());
    // ospSet1f(ospcam, "apertureRadius", );
    // ospSet1f(ospcam, "focalDistance", cr->GetCameraParameters()->FocalDistance());
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
        if (this->materials[entry.first] != nullptr) {
            ospRelease(this->materials[entry.first]);
            this->materials.erase(entry.first);
        }
        if (element.materialContainer != NULL) {
            switch (element.materialContainer->materialType) {
            case OBJMATERIAL:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "OBJMaterial");
                ospSet3fv(this->materials[entry.first], "Kd", element.materialContainer->Kd.data());
                ospSet3fv(this->materials[entry.first], "Ks", element.materialContainer->Ks.data());
                ospSet1f(this->materials[entry.first], "Ns", element.materialContainer->Ns);
                ospSet1f(this->materials[entry.first], "d", element.materialContainer->d);
                ospSet3fv(this->materials[entry.first], "Tf", element.materialContainer->Tf.data());
                break;
            case LUMINOUS:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Luminous");
                ospSet3fv(this->materials[entry.first], "color", element.materialContainer->lumColor.data());
                ospSet1f(this->materials[entry.first], "intensity", element.materialContainer->lumIntensity);
                ospSet1f(this->materials[entry.first], "transparency", element.materialContainer->lumTransparency);
                break;
            case GLASS:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Glass");
                ospSet1f(this->materials[entry.first], "etaInside", element.materialContainer->glassEtaInside);
                ospSet1f(this->materials[entry.first], "etaOutside", element.materialContainer->glassEtaOutside);
                ospSet3fv(this->materials[entry.first], "attenuationColorInside",
                    element.materialContainer->glassAttenuationColorInside.data());
                ospSet3fv(this->materials[entry.first], "attenuationColorOutside",
                    element.materialContainer->glassAttenuationColorOutside.data());
                ospSet1f(this->materials[entry.first], "attenuationDistance",
                    element.materialContainer->glassAttenuationDistance);
                break;
            case MATTE:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Matte");
                ospSet3fv(
                    this->materials[entry.first], "reflectance", element.materialContainer->matteReflectance.data());
                break;
            case METAL:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Metal");
                ospSet3fv(
                    this->materials[entry.first], "reflectance", element.materialContainer->metalReflectance.data());
                ospSet3fv(this->materials[entry.first], "eta", element.materialContainer->metalEta.data());
                ospSet3fv(this->materials[entry.first], "k", element.materialContainer->metalK.data());
                ospSet1f(this->materials[entry.first], "roughness", element.materialContainer->metalRoughness);
                break;
            case METALLICPAINT:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "MetallicPaint");
                ospSet3fv(
                    this->materials[entry.first], "shadeColor", element.materialContainer->metallicShadeColor.data());
                ospSet3fv(this->materials[entry.first], "glitterColor",
                    element.materialContainer->metallicGlitterColor.data());
                ospSet1f(
                    this->materials[entry.first], "glitterSpread", element.materialContainer->metallicGlitterSpread);
                ospSet1f(this->materials[entry.first], "eta", element.materialContainer->metallicEta);
                break;
            case PLASTIC:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Plastic");
                ospSet3fv(this->materials[entry.first], "pigmentColor",
                    element.materialContainer->plasticPigmentColor.data());
                ospSet1f(this->materials[entry.first], "eta", element.materialContainer->plasticEta);
                ospSet1f(this->materials[entry.first], "roughness", element.materialContainer->plasticRoughness);
                ospSet1f(this->materials[entry.first], "thickness", element.materialContainer->plasticThickness);
                break;
            case THINGLASS:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "ThinGlass");
                ospSet3fv(this->materials[entry.first], "transmission",
                    element.materialContainer->thinglassTransmission.data());
                ospSet1f(this->materials[entry.first], "eta", element.materialContainer->thinglassEta);
                ospSet1f(this->materials[entry.first], "thickness", element.materialContainer->thinglassThickness);
                break;
            case VELVET:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Velvet");
                ospSet3fv(
                    this->materials[entry.first], "reflectance", element.materialContainer->velvetReflectance.data());
                ospSet3fv(this->materials[entry.first], "horizonScatteringColor",
                    element.materialContainer->velvetHorizonScatteringColor.data());
                ospSet1f(
                    this->materials[entry.first], "backScattering", element.materialContainer->velvetBackScattering);
                ospSet1f(this->materials[entry.first], "horizonScatteringFallOff",
                    element.materialContainer->velvetHorizonScatteringFallOff);
                break;
            }
            ospCommit(this->materials[entry.first]);
        }

        if (this->materials[entry.first] != NULL) {
            if (element.type == structureTypeEnum::GEOMETRY) {
                ospSetMaterial(
                    std::get<OSPGeometry>(this->baseStructures[entry.first].back()), this->materials[entry.first]);
                ospCommit(std::get<OSPGeometry>(this->baseStructures[entry.first].back()));
            }
        }
    }
}

void AbstractOSPRayRenderer::changeTransformation() {

    for (auto& entry : this->baseStructures) {
        if (this->structureMap[entry.first].transformationContainer == nullptr) continue;
        auto trafo = this->structureMap[entry.first].transformationContainer;
        osp::affine3f xfm;
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

        if (this->structureMap[entry.first].dataChanged) {
            if (instancedModels[entry.first] != nullptr) {
                ospRelease(instancedModels[entry.first]);
                instancedModels.erase(entry.first);
            }
            instancedModels[entry.first] = ospNewModel();
            if (this->structureMap[entry.first].type == structureTypeEnum::GEOMETRY) {
                for (int i = 0; i < entry.second.size(); i++) {
                    ospAddGeometry(instancedModels[entry.first], std::get<OSPGeometry>(entry.second[i]));
                }
            } else {
                for (int i = 0; i < entry.second.size(); i++) {
                    ospAddVolume(instancedModels[entry.first], std::get<OSPVolume>(entry.second[i]));
                }
            }
        }
        ospCommit(instancedModels[entry.first]);
        if (this->instances[entry.first] != nullptr) {
            ospRemoveGeometry(world, this->instances[entry.first]);
            //ospRelease(this->instances[entry.first]);
            this->instances.erase(entry.first);
        }
        this->instances[entry.first] = ospNewInstance(instancedModels[entry.first], xfm);
        if (this->materials[entry.first] != nullptr) {
            ospSetMaterial(this->instances[entry.first], this->materials[entry.first]);
        }
        ospCommit(this->instances[entry.first]);
        ospAddGeometry(world, this->instances[entry.first]);
        ospCommit(world);
    }
}


bool AbstractOSPRayRenderer::fillWorld() {

    bool returnValue = true;
    bool applyTransformation = false;

    ospcommon::box3f worldBounds;
    std::vector<ospcommon::box3f> ghostRegions;
    std::vector<ospcommon::box3f> regions;

    for (auto& entry : this->structureMap) {

        numCreateGeo = 1;
        auto const& element = entry.second;

        // check if structure should be released first
        if (element.dataChanged) {
            for (auto& stru : this->baseStructures[entry.first]) {
                if (element.type == structureTypeEnum::GEOMETRY) {
                    ospRemoveGeometry(this->world, std::get<OSPGeometry>(stru));
                    //ospRelease(std::get<OSPGeometry>(stru));
                } else if (element.type == structureTypeEnum::VOLUME) {
                    ospRemoveVolume(this->world, std::get<OSPVolume>(stru));
                   //ospRelease(std::get<OSPVolume>(stru));
                }
            }
            this->baseStructures.erase(entry.first);
        } else {
            continue;
        }


        // custom material settings
        if (this->materials[entry.first] != nullptr) {
            ospRelease(this->materials[entry.first]);
            this->materials.erase(entry.first);
        }
        if (element.materialContainer != NULL &&
            this->rd_type.Param<megamol::core::param::EnumParam>()->Value() != MPI_RAYCAST) {
            switch (element.materialContainer->materialType) {
            case OBJMATERIAL:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "OBJMaterial");
                ospSet3fv(this->materials[entry.first], "Kd", element.materialContainer->Kd.data());
                ospSet3fv(this->materials[entry.first], "Ks", element.materialContainer->Ks.data());
                ospSet1f(this->materials[entry.first], "Ns", element.materialContainer->Ns);
                ospSet1f(this->materials[entry.first], "d", element.materialContainer->d);
                ospSet3fv(this->materials[entry.first], "Tf", element.materialContainer->Tf.data());
                break;
            case LUMINOUS:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Luminous");
                ospSet3fv(this->materials[entry.first], "color", element.materialContainer->lumColor.data());
                ospSet1f(this->materials[entry.first], "intensity", element.materialContainer->lumIntensity);
                ospSet1f(this->materials[entry.first], "transparency", element.materialContainer->lumTransparency);
                break;
            case GLASS:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Glass");
                ospSet1f(this->materials[entry.first], "etaInside", element.materialContainer->glassEtaInside);
                ospSet1f(this->materials[entry.first], "etaOutside", element.materialContainer->glassEtaOutside);
                ospSet3fv(this->materials[entry.first], "attenuationColorInside",
                    element.materialContainer->glassAttenuationColorInside.data());
                ospSet3fv(this->materials[entry.first], "attenuationColorOutside",
                    element.materialContainer->glassAttenuationColorOutside.data());
                ospSet1f(this->materials[entry.first], "attenuationDistance",
                    element.materialContainer->glassAttenuationDistance);
                break;
            case MATTE:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Matte");
                ospSet3fv(
                    this->materials[entry.first], "reflectance", element.materialContainer->matteReflectance.data());
                break;
            case METAL:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Metal");
                ospSet3fv(
                    this->materials[entry.first], "reflectance", element.materialContainer->metalReflectance.data());
                ospSet3fv(this->materials[entry.first], "eta", element.materialContainer->metalEta.data());
                ospSet3fv(this->materials[entry.first], "k", element.materialContainer->metalK.data());
                ospSet1f(this->materials[entry.first], "roughness", element.materialContainer->metalRoughness);
                break;
            case METALLICPAINT:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "MetallicPaint");
                ospSet3fv(
                    this->materials[entry.first], "shadeColor", element.materialContainer->metallicShadeColor.data());
                ospSet3fv(this->materials[entry.first], "glitterColor",
                    element.materialContainer->metallicGlitterColor.data());
                ospSet1f(
                    this->materials[entry.first], "glitterSpread", element.materialContainer->metallicGlitterSpread);
                ospSet1f(this->materials[entry.first], "eta", element.materialContainer->metallicEta);
                break;
            case PLASTIC:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Plastic");
                ospSet3fv(this->materials[entry.first], "pigmentColor",
                    element.materialContainer->plasticPigmentColor.data());
                ospSet1f(this->materials[entry.first], "eta", element.materialContainer->plasticEta);
                ospSet1f(this->materials[entry.first], "roughness", element.materialContainer->plasticRoughness);
                ospSet1f(this->materials[entry.first], "thickness", element.materialContainer->plasticThickness);
                break;
            case THINGLASS:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "ThinGlass");
                ospSet3fv(this->materials[entry.first], "transmission",
                    element.materialContainer->thinglassTransmission.data());
                ospSet1f(this->materials[entry.first], "eta", element.materialContainer->thinglassEta);
                ospSet1f(this->materials[entry.first], "thickness", element.materialContainer->thinglassThickness);
                break;
            case VELVET:
                this->materials[entry.first] = ospNewMaterial2(this->rd_type_string.c_str(), "Velvet");
                ospSet3fv(
                    this->materials[entry.first], "reflectance", element.materialContainer->velvetReflectance.data());
                ospSet3fv(this->materials[entry.first], "horizonScatteringColor",
                    element.materialContainer->velvetHorizonScatteringColor.data());
                ospSet1f(
                    this->materials[entry.first], "backScattering", element.materialContainer->velvetBackScattering);
                ospSet1f(this->materials[entry.first], "horizonScatteringFallOff",
                    element.materialContainer->velvetHorizonScatteringFallOff);
                break;
            }
            ospCommit(this->materials[entry.first]);
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
                    baseStructures[entry.first].push_back(reinterpret_cast<OSPGeometry>(structure));
                } else if (element.ospStructures.second == structureTypeEnum::VOLUME) {
                    baseStructures[entry.first].push_back(reinterpret_cast<OSPVolume>(structure));
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRAY_API_STRUCTURE: Something went wrong.");
                }
            }
            // General geometry execution
            for (unsigned int i = 0; i < element.ospStructures.first.size(); i++) {
                auto idx = baseStructures[entry.first].size() - 1 - i;
                if (this->materials[entry.first] != NULL && baseStructures[entry.first].size() > 0) {
                    ospSetMaterial(
                        std::get<OSPGeometry>(baseStructures[entry.first][idx]), this->materials[entry.first]);
                }

                if (baseStructures[entry.first].size() > 0) {
                    ospCommit(std::get<OSPGeometry>(baseStructures[entry.first][idx]));
                    ospAddGeometry(world, std::get<OSPGeometry>(baseStructures[entry.first][idx]));
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
                    baseStructures[entry.first].push_back(ospNewGeometry("spheres"));

                    long long int vertexFloatsToRead = element.partCount * element.vertexLength / numCreateGeo;
                    vertexFloatsToRead -= vertexFloatsToRead % element.vertexLength;
                    if (vertexData != NULL) ospRelease(vertexData);
                    vertexData = ospNewData(vertexFloatsToRead, OSP_FLOAT,
                        &element.vertexData->operator[](i* vertexFloatsToRead), OSP_DATA_SHARED_BUFFER);

                    ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "bytes_per_sphere",
                        element.vertexLength * sizeof(float));

                    if (element.vertexLength > 3) {
                        // ospRemoveParam(geo.back(), "radius");
                        ospSet1f(std::get<OSPGeometry>(baseStructures[entry.first].back()), "offset_radius",
                            3 * sizeof(float));
                        // TODO: HACK
                        ospSet1f(std::get<OSPGeometry>(baseStructures[entry.first].back()), "radius", 1);
                    } else {
                        ospSet1f(
                            std::get<OSPGeometry>(baseStructures[entry.first].back()), "radius", element.globalRadius);
                    }
                    ospCommit(vertexData);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "spheres", vertexData);

                    if (element.colorLength == 4) {
                        long long int colorFloatsToRead = element.partCount * element.colorLength / numCreateGeo;
                        colorFloatsToRead -= colorFloatsToRead % element.colorLength;
                        if (colorData != NULL) ospRelease(colorData);
                        colorData = ospNewData(colorFloatsToRead, OSP_FLOAT,
                            &element.colorData->operator[](i* colorFloatsToRead), OSP_DATA_SHARED_BUFFER);
                        ospCommit(colorData);
                        ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color", colorData);
                        // ospSet1i(geo.back(), "color_components", 4);
                        ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color_format", OSP_FLOAT4);
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
                    baseStructures[entry.first].push_back(ospNewGeometry("spheres"));


                    long long int floatsToRead =
                        element.partCount * element.vertexStride / (numCreateGeo * sizeof(float));
                    floatsToRead -= floatsToRead % (element.vertexStride / sizeof(float));

                    if (vertexData != NULL) ospRelease(vertexData);
                    vertexData = ospNewData(floatsToRead, OSP_FLOAT,
                        &static_cast<const float*>(element.raw)[i * floatsToRead], OSP_DATA_SHARED_BUFFER);
                    ospCommit(vertexData);
                    ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "bytes_per_sphere",
                        element.vertexStride);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "spheres", vertexData);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color", NULL);

                    if (element.vertexLength > 3) {
                        ospSet1f(std::get<OSPGeometry>(baseStructures[entry.first].back()), "offset_radius",
                            3 * sizeof(float));
                    } else {
                        ospSet1f(
                            std::get<OSPGeometry>(baseStructures[entry.first].back()), "radius", element.globalRadius);
                    }
                    if (element.mmpldColor ==
                            core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB ||
                        element.mmpldColor ==
                            core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGBA) {

                        ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color_offset",
                            element.vertexLength *
                                sizeof(float)); // TODO: This won't work if there are radii in the array
                        ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color_stride",
                            element.colorStride);
                        ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color", vertexData);
                        if (element.mmpldColor ==
                            core::moldyn::SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB) {
                            // ospSet1i(geo.back(), "color_components", 3);
                            ospSet1i(
                                std::get<OSPGeometry>(baseStructures[entry.first].back()), "color_format", OSP_FLOAT3);
                        } else {
                            // ospSet1i(geo.back(), "color_components", 4);
                            ospSet1i(
                                std::get<OSPGeometry>(baseStructures[entry.first].back()), "color_format", OSP_FLOAT4);
                        }
                    }
                }
                break;
            case geometryTypeEnum::QUADS:
            case geometryTypeEnum::TRIANGLES:
                if (element.mesh == NULL) {
                    // returnValue = false;
                    break;
                }
                {
                    std::vector<mesh::ImageDataAccessCollection::Image> tex_vec;
                    if (element.mesh_textures != nullptr) {
                        assert(element.mesh->accessMesh().size() == element.mesh_textures->accessImages().size());
                        tex_vec = element.mesh_textures->accessImages();
                    }
                    this->numCreateGeo = element.mesh->accessMesh().size();

                    uint32_t mesh_index = 0;
                    for (auto& mesh : element.mesh->accessMesh()) {

                        if (element.geometryType == TRIANGLES) {
                            this->baseStructures[entry.first].push_back(ospNewGeometry("triangles"));
                        } else if (element.geometryType == QUADS) {
                            this->baseStructures[entry.first].push_back(ospNewGeometry("quads"));
                        }

                        for (auto& attrib : mesh.attributes) {

                            if (attrib.semantic == mesh::MeshDataAccessCollection::POSITION) {
                                auto count = attrib.byte_size /
                                             (mesh::MeshDataAccessCollection::getByteSize(attrib.component_type) *
                                                 attrib.component_cnt);
                                auto ospType = OSP_FLOAT3;
                                if (attrib.stride == 4 * sizeof(float)) ospType = OSP_FLOAT3A;
                                vertexData = ospNewData(count, ospType, attrib.data, OSP_DATA_SHARED_BUFFER);
                                ospCommit(vertexData);
                                ospSetData(
                                    std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex", vertexData);
                            }

                             // check normal pointer
                             if (attrib.semantic == mesh::MeshDataAccessCollection::NORMAL) {
                                auto count =
                                    attrib.byte_size / attrib.stride;
                                auto ospType = OSP_FLOAT3;
                                if (attrib.stride == 4 * sizeof(float)) ospType = OSP_FLOAT3A;
                                normalData = ospNewData(count, ospType, attrib.data);
                                ospCommit(normalData);
                                ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex.normal",
                                    normalData);
                            }

                            // check colorpointer and convert to rgba
                            if (attrib.semantic == mesh::MeshDataAccessCollection::COLOR) {
                                if (attrib.component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT)
                                    colorData = ospNewData(
                                        attrib.byte_size /
                                            (mesh::MeshDataAccessCollection::getByteSize(attrib.component_type) *
                                                attrib.component_cnt),
                                        OSP_FLOAT4, attrib.data);
                                else
                                    colorData =
                                        ospNewData(attrib.byte_size, OSP_UCHAR, attrib.data, OSP_DATA_SHARED_BUFFER);
                                ospCommit(colorData);
                                ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex.color",
                                    colorData);
                            }

                            // check texture array
                            if (attrib.semantic == mesh::MeshDataAccessCollection::TEXCOORD) {
                                texData = ospNewData(attrib.byte_size / (mesh::MeshDataAccessCollection::getByteSize(
                                                                             attrib.component_type) *
                                                                            attrib.component_cnt),
                                    OSP_FLOAT2, attrib.data, OSP_DATA_SHARED_BUFFER);
                                ospCommit(texData);
                                ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex.texcoord",
                                    texData);
                            }
                        }
                        // check index pointer
                        if (mesh.indices.data != nullptr) {
                            auto count =
                                mesh.indices.byte_size / mesh::MeshDataAccessCollection::getByteSize(mesh.indices.type);
                            indexData = ospNewData(count, OSP_UINT, mesh.indices.data, OSP_DATA_SHARED_BUFFER);
                            ospCommit(indexData);
                            ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "index", indexData);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRay cannot render meshes without index array");
                            returnValue = false;
                        }
                        if (element.mesh_textures != nullptr) {
                            OSPTextureFormat osp_tex_format = OSP_TEXTURE_FORMAT_INVALID;
                            switch (tex_vec[mesh_index].format) {
                            case mesh::ImageDataAccessCollection::TextureFormat::RGBA8:
                                osp_tex_format = OSP_TEXTURE_RGBA8;
                                break;
                            case mesh::ImageDataAccessCollection::TextureFormat::RGB32F:
                                osp_tex_format = OSP_TEXTURE_RGB32F;
                                break;
                            case mesh::ImageDataAccessCollection::TextureFormat::RGB8:
                                osp_tex_format = OSP_TEXTURE_RGB8;
                                break;
                            case mesh::ImageDataAccessCollection::TextureFormat::RGBA32F:
                                osp_tex_format = OSP_TEXTURE_RGBA32F;
                                break;
                            default:
                                osp_tex_format = OSP_TEXTURE_RGB8;
                                break;
                            }

                            auto ospTexture = ospNewTexture2D({tex_vec[mesh_index].width, tex_vec[mesh_index].height},
                                osp_tex_format, tex_vec[mesh_index].data, OSP_DATA_SHARED_BUFFER);
                            auto ospMat = ospNewMaterial2(this->rd_type_string.c_str(), "OBJMaterial");
                            ospCommit(ospTexture);
                            ospSetObject(ospMat, "map_Kd", ospTexture);
                            // ospSetObject(ospMat, "map_Ks", ospTexture);
                            // ospSetObject(ospMat, "map_d", ospTexture);
                            ospCommit(ospMat);
                            ospSetMaterial(std::get<OSPGeometry>(baseStructures[entry.first].back()), ospMat);
                        }
                        mesh_index++;
                    }
                }
                break;
            case geometryTypeEnum::STREAMLINES:
                if (element.vertexData == nullptr && element.mesh == nullptr) {
                    // returnValue = false;
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[AbstractOSPRayRenderer]Streamline geometry detected but no data found.");
                    break;
                }
                if (element.mesh != nullptr) {
                    this->numCreateGeo = element.mesh->accessMesh().size();
                    for (auto& mesh : element.mesh->accessMesh()) {

                        baseStructures[entry.first].push_back(ospNewGeometry("streamlines"));

                        for (auto& attrib : mesh.attributes) {

                            if (attrib.semantic == mesh::MeshDataAccessCollection::POSITION) {
                                const auto count = attrib.byte_size / attrib.stride;
                                assert(attrib.stride == 4 * sizeof(float));
                                vertexData = ospNewData(count, OSP_FLOAT3A, attrib.data, OSP_DATA_SHARED_BUFFER);
                                ospCommit(vertexData);
                                ospSetData(
                                    std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex", vertexData);
                            }

                            // check colorpointer and convert to rgba
                            if (attrib.semantic == mesh::MeshDataAccessCollection::COLOR) {
                                if (attrib.component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT)
                                    colorData = ospNewData(attrib.byte_size / attrib.stride, OSP_FLOAT4, attrib.data);
                                else
                                    colorData =
                                        ospNewData(attrib.byte_size, OSP_UCHAR, attrib.data, OSP_DATA_SHARED_BUFFER);
                                ospCommit(colorData);
                                ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex.color",
                                    colorData);
                            }
                        }
                        // check index pointer
                        if (mesh.indices.data != nullptr) {
                            const auto count =
                                mesh.indices.byte_size / mesh::MeshDataAccessCollection::getByteSize(mesh.indices.type);
                            indexData = ospNewData(count, OSP_INT, mesh.indices.data, OSP_DATA_SHARED_BUFFER);
                            ospCommit(indexData);
                            ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "index", indexData);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError("OSPRay cannot render meshes without index array");
                            returnValue = false;
                        }

                        ospSet1f(
                            std::get<OSPGeometry>(baseStructures[entry.first].back()), "radius", element.globalRadius);
                        ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "smooth", element.smooth);
                    } // end for geometry
                } else {
                    baseStructures[entry.first].push_back(ospNewGeometry("streamlines"));
                    this->numCreateGeo = 1;
                    osp::vec3fa* data = new osp::vec3fa[element.vertexData->size() / 3];

                    // fill aligned array with vertex data
                    for (unsigned int i = 0; i < element.vertexData->size() / 3; i++) {
                        data[i].x = element.vertexData->data()[3 * i + 0];
                        data[i].y = element.vertexData->data()[3 * i + 1];
                        data[i].z = element.vertexData->data()[3 * i + 2];
                    }


                    vertexData = ospNewData(element.vertexData->size() / 3, OSP_FLOAT3A, data, OSP_DATA_SHARED_BUFFER);
                    ospCommit(vertexData);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex", vertexData);

                    indexData = ospNewData(
                        element.indexData->size(), OSP_UINT, element.indexData->data(), OSP_DATA_SHARED_BUFFER);
                    ospCommit(indexData);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "index", indexData);

                    if (element.colorData->size() > 0) {
                        colorData = ospNewData(element.colorData->size() / element.colorLength, OSP_FLOAT4,
                            element.colorData->data(), OSP_DATA_SHARED_BUFFER);
                        ospCommit(colorData);
                        ospSetData(
                            std::get<OSPGeometry>(baseStructures[entry.first].back()), "vertex.color", colorData);
                    }

                    ospSet1f(std::get<OSPGeometry>(baseStructures[entry.first].back()), "radius", element.globalRadius);
                    ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "smooth", element.smooth);
                }
                break;
            case geometryTypeEnum::CYLINDERS:
                if (element.raw == NULL) {
                    // returnValue = false;
                    break;
                }

                numCreateGeo = element.partCount * element.vertexStride / ispcLimit + 1;

                for (unsigned int i = 0; i < numCreateGeo; i++) {
                    baseStructures[entry.first].push_back(ospNewGeometry("cylinders"));


                    long long int floatsToRead =
                        element.partCount * element.vertexStride / (numCreateGeo * sizeof(float));
                    floatsToRead -= floatsToRead % (element.vertexStride / sizeof(float));

                    if (vertexData != NULL) ospRelease(vertexData);
                    vertexData = ospNewData(floatsToRead, OSP_FLOAT,
                        &static_cast<const float*>(element.raw)[i * floatsToRead], OSP_DATA_SHARED_BUFFER);
                    ospCommit(vertexData);
                    ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "bytes_per_cylinder",
                        2*element.vertexStride);
                    ospSet1i(std::get<OSPGeometry>(baseStructures[entry.first].back()), "offset_v1",
                        element.vertexStride);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "cylinders", vertexData);
                    ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "color", NULL);

                    if (element.vertexLength > 3) {
                        ospSet1f(std::get<OSPGeometry>(baseStructures[entry.first].back()), "offset_radius",
                            3 * sizeof(float));
                    } else {
                        ospSet1f(
                            std::get<OSPGeometry>(baseStructures[entry.first].back()), "radius", element.globalRadius);
                    }
                }
                break;
            }

            // General geometry execution
            for (unsigned int i = 0; i < this->numCreateGeo; i++) {
                if (this->materials[entry.first] != NULL && baseStructures[entry.first].size() > 0) {
                    ospSetMaterial(
                        std::get<OSPGeometry>(baseStructures[entry.first].rbegin()[i]), this->materials[entry.first]);
                }

                if (baseStructures[entry.first].size() > 0) {
                    ospCommit(std::get<OSPGeometry>(baseStructures[entry.first].rbegin()[i]));
                    if (element.transformationContainer == nullptr) {
                        ospAddGeometry(world, std::get<OSPGeometry>(baseStructures[entry.first].rbegin()[i]));
                    } else {
                        applyTransformation = true;
                    }
                }
            }

            //if (vertexData != NULL) ospRelease(vertexData);
            //if (colorData != NULL) ospRelease(colorData);
            //if (normalData != NULL) ospRelease(normalData);
            //if (texData != NULL) ospRelease(texData);
            //if (indexData != NULL) ospRelease(indexData);
            //if (xData != NULL) ospRelease(xData);
            //if (yData != NULL) ospRelease(yData);
            //if (zData != NULL) ospRelease(zData);
            //if (bboxData != NULL) ospRelease(bboxData);
            //if (aovol != NULL) ospRelease(aovol);
            //if (voxels != NULL) ospRelease(voxels);

            break;

        case structureTypeEnum::VOLUME:

            if (element.voxels == NULL) {
                // returnValue = false;
                break;
            }

            baseStructures[entry.first].push_back(ospNewVolume("shared_structured_volume"));

            auto type = static_cast<uint8_t>(element.voxelDType);

            ospSetString(
                std::get<OSPVolume>(baseStructures[entry.first].back()), "voxelType", voxelDataTypeS[type].c_str());
            // float fixedSpacing[3];
            // for (auto x = 0; x < 3; ++x) {
            //    fixedSpacing[x] = element.gridSpacing->at(x) / (element.dimensions->at(x) - 1) +
            //    element.gridSpacing->at(x);
            //}
            // scaling properties of the volume
            ospSet3iv(
                std::get<OSPVolume>(baseStructures[entry.first].back()), "dimensions", element.dimensions->data());
            ospSet3fv(
                std::get<OSPVolume>(baseStructures[entry.first].back()), "gridOrigin", element.gridOrigin->data());
            ospSet3fv(
                std::get<OSPVolume>(baseStructures[entry.first].back()), "gridSpacing", element.gridSpacing->data());
            ospSet2f(std::get<OSPVolume>(baseStructures[entry.first].back()), "voxelRange", element.valueRange->first,
                element.valueRange->second);

            ospSet1b(std::get<OSPVolume>(baseStructures[entry.first].back()), "singleShade", element.useMIP);
            ospSet1b(
                std::get<OSPVolume>(baseStructures[entry.first].back()), "gradientShadingEnables", element.useGradient);
            ospSet1b(
                std::get<OSPVolume>(baseStructures[entry.first].back()), "preIntegration", element.usePreIntegration);
            ospSet1b(std::get<OSPVolume>(baseStructures[entry.first].back()), "adaptiveSampling",
                element.useAdaptiveSampling);
            ospSet1f(std::get<OSPVolume>(baseStructures[entry.first].back()), "adaptiveScalar", element.adaptiveFactor);
            ospSet1f(std::get<OSPVolume>(baseStructures[entry.first].back()), "adaptiveMaxSamplingRate",
                element.adaptiveMaxRate);
            ospSet1f(std::get<OSPVolume>(baseStructures[entry.first].back()), "samplingRate", element.samplingRate);

            // add data
            voxels = ospNewData(element.voxelCount, static_cast<OSPDataType>(voxelDataTypeOSP[type]), element.voxels,
                OSP_DATA_SHARED_BUFFER);
            ospCommit(voxels);
            ospSetData(std::get<OSPVolume>(baseStructures[entry.first].back()), "voxelData", voxels);

            // ClippingBox

            if (element.clippingBoxActive) {
                ospSet3fv(std::get<OSPVolume>(baseStructures[entry.first].back()), "volumeClippingBoxLower",
                    element.clippingBoxLower->data());
                ospSet3fv(std::get<OSPVolume>(baseStructures[entry.first].back()), "volumeClippingBoxUpper",
                    element.clippingBoxUpper->data());
            } else {
                ospSetVec3f(std::get<OSPVolume>(baseStructures[entry.first].back()), "volumeClippingBoxLower",
                    {0.0f, 0.0f, 0.0f});
                ospSetVec3f(std::get<OSPVolume>(baseStructures[entry.first].back()), "volumeClippingBoxUpper",
                    {0.0f, 0.0f, 0.0f});
            }

            OSPTransferFunction tf = ospNewTransferFunction("piecewise_linear");

            OSPData tf_rgb = ospNewData(element.tfRGB->size() / 3, OSP_FLOAT3, element.tfRGB->data());
            OSPData tf_opa = ospNewData(element.tfA->size(), OSP_FLOAT, element.tfA->data());
            ospSetData(tf, "colors", tf_rgb);
            ospSetData(tf, "opacities", tf_opa);
            ospSet2f(tf, "valueRange", element.valueRange->first, element.valueRange->second);

            ospCommit(tf);

            ospSetObject(std::get<OSPVolume>(baseStructures[entry.first].back()), "transferFunction", tf);
            ospCommit(std::get<OSPVolume>(baseStructures[entry.first].back()));
            //ospRelease(tf);

            switch (element.volRepType) {
            case volumeRepresentationType::VOLUMEREP:
                if (element.transformationContainer == nullptr) {
                    ospAddVolume(world, std::get<OSPVolume>(baseStructures[entry.first].back()));
                } else {
                    applyTransformation = true;
                }
                break;

            case volumeRepresentationType::ISOSURFACE:
                // isosurface
                baseStructures[entry.first].push_back(ospNewGeometry("isosurfaces"));
                isovalues = ospNewData(1, OSP_FLOAT, element.isoValue->data());
                ospCommit(isovalues);
                ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "isovalues", isovalues);
                ospSetObject(std::get<OSPGeometry>(baseStructures[entry.first].back()), "volume",
                    std::get<OSPVolume>(baseStructures[entry.first].front()));

                if (this->materials[entry.first] != NULL) {
                    ospSetMaterial(
                        std::get<OSPGeometry>(baseStructures[entry.first].back()), this->materials[entry.first]);
                }

                ospCommit(std::get<OSPGeometry>(baseStructures[entry.first].back()));

                if (element.transformationContainer == nullptr) {
                    ospAddGeometry(world, std::get<OSPGeometry>(baseStructures[entry.first].back())); // Show isosurface
                } else {
                    applyTransformation = true;
                }
                break;

            case volumeRepresentationType::SLICE:
                baseStructures[entry.first].push_back(ospNewGeometry("slices"));
                planes = ospNewData(1, OSP_FLOAT4, element.sliceData->data());
                ospCommit(planes);
                ospSetData(std::get<OSPGeometry>(baseStructures[entry.first].back()), "planes", planes);
                ospSetObject(std::get<OSPGeometry>(baseStructures[entry.first].back()), "volume",
                    std::get<OSPVolume>(baseStructures[entry.first].front()));

                if (this->materials[entry.first] != NULL) {
                    ospSetMaterial(
                        std::get<OSPGeometry>(baseStructures[entry.first].back()), this->materials[entry.first]);
                }

                ospCommit(std::get<OSPGeometry>(baseStructures[entry.first].back()));

                if (element.transformationContainer == nullptr) {
                    ospAddGeometry(world, std::get<OSPGeometry>(baseStructures[entry.first].back())); // Show slice
                } else {
                    applyTransformation = true;
                }

                break;
            }

            //if (voxels != NULL) ospRelease(voxels);
            //if (planes != NULL) ospRelease(planes);
            //if (isovalues != NULL) ospRelease(isovalues);

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

    if (applyTransformation) this->changeTransformation();


    return returnValue;
}

void AbstractOSPRayRenderer::releaseOSPRayStuff() {}

} // end namespace ospray
} // end namespace megamol
