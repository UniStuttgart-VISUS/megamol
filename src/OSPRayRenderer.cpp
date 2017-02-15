/*
* OSPRayRenderer.cpp
* Copyright (C) 2009-2015 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "vislib/sys/Path.h"
#include "OSPRayRenderer.h"
#include "ospray/ospray.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"

#include <stdio.h>

using namespace megamol;

ospray::OSPRayRenderer::OSPRayRenderer(void) :
    core::view::Renderer3DModule(),
    extraSamles("General::extraSamples", "Extra sampling when camera is not moved"),
    // General light parameters
    lightColor("Light::General::LightColor", "Sets the color of the Light"),
    shadows("Light::General::Shadows", "Enables/Disables computation of hard shadows"),
    lightIntensity("Light::General::LightIntensity", "Intensity of the Light"),
    lightType("Light::Type::LightType", "Type of the light"),
    // Distant light parameters
    dl_direction("Light::DistantLight::LightDirection", "Direction of the Light"),
    dl_angularDiameter("Light::DistantLight::AngularDiameter", "If greater than zero results in soft shadows"),
    dl_eye_direction("Light::DistantLight::EyeDirection", "Sets the light direction as view direction"),
    // point light parameters
    pl_position("Light::PointLight::Position", ""),
    pl_radius("Light::PointLight::Radius", ""),
    // spot light parameters
    sl_position("Light::SpotLight::Position", ""),
    sl_direction("Light::SpotLight::Direction", ""),
    sl_openingAngle("Light::SpotLight::openingAngle", ""),
    sl_penumbraAngle("Light::SpotLight::penumbraAngle", ""),
    sl_radius("Light::SpotLight::Radius", ""),
    // quad light parameters
    ql_position("Light::QuadLight::Position", ""),
    ql_edgeOne("Light::QuadLight::Edge1", ""),
    ql_edgeTwo("Light::QuadLight::Edge2", ""),
    // hdri light parameteres
    hdri_up("Light::HDRILight::up", ""),
    hdri_direction("Light::HDRILight::Direction", ""),
    hdri_evnfile("Light::HDRILight::EvironmentFile", ""),
    // general renderer parameters
    rd_epsilon("Renderer::Epsilon", "Ray epsilon to avoid self-intersections"),
    rd_spp("Renderer::SamplesPerPixel", "Samples per pixel"),
    rd_maxRecursion("Renderer::maxRecursion", "Maximum ray recursion depth"),
    rd_type("Renderer::Type", "Select between SciVis and PathTracer"),
    // scivis renderer parameters
    AOtransparencyEnabled("Renderer::SciVis::AOtransparencyEnabled", "Enables or disables AO transparency"),
    AOsamples("Renderer::SciVis::AOsamples", "Number of rays per sample to compute ambient occlusion"),
    AOdistance("Renderer::SciVis::AOdistance", "Maximum distance to consider for ambient occlusion"),
    // pathtracer renderer parameters
    rd_ptBackground("Renderer::PathTracer::BackgroundTexture", "Texture image used as background, replacing visible lights in infinity") {

    light = NULL;
    lightArray = NULL;
    device = NULL;
    framebufferIsDirty = true;

    core::param::EnumParam *lt = new core::param::EnumParam(NONE);
    lt->SetTypePair(NONE, "None");
    lt->SetTypePair(DISTANTLIGHT, "DistantLight");
    lt->SetTypePair(POINTLIGHT, "PointLight");
    lt->SetTypePair(SPOTLIGHT, "SpotLight");
    lt->SetTypePair(QUADLIGHT, "QuadLight");
    lt->SetTypePair(AMBIENTLIGHT, "AmbientLight");
    lt->SetTypePair(HDRILIGHT, "HDRILight");

    core::param::EnumParam *rdt = new core::param::EnumParam(SCIVIS);
    rdt->SetTypePair(SCIVIS, "SciVis");
    rdt->SetTypePair(PATHTRACER, "PathTracer");

    // Ambient parameters
    this->AOtransparencyEnabled << new core::param::BoolParam(false);
    this->AOsamples << new core::param::IntParam(1);
    this->AOdistance << new core::param::FloatParam(1e20f);
    this->extraSamles << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->AOtransparencyEnabled);
    this->MakeSlotAvailable(&this->AOsamples);
    this->MakeSlotAvailable(&this->AOdistance);
    this->MakeSlotAvailable(&this->extraSamles);

    // general light
    this->shadows << new core::param::BoolParam(0);
    this->lightColor << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->lightType << lt;
    this->lightIntensity << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lightIntensity);
    this->MakeSlotAvailable(&this->lightColor);
    this->MakeSlotAvailable(&this->shadows);
    this->MakeSlotAvailable(&this->lightType);

    // distant light
    this->dl_angularDiameter << new core::param::FloatParam(0.0f);
    this->dl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, -1.0f, 0.0f));
    this->dl_eye_direction << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->dl_direction);
    this->MakeSlotAvailable(&this->dl_angularDiameter);
    this->MakeSlotAvailable(&this->dl_eye_direction);

    // point light
    this->pl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->pl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->pl_position);
    this->MakeSlotAvailable(&this->pl_radius);

    // spot light
    this->sl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->sl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->sl_openingAngle << new core::param::FloatParam(0.0f);
    this->sl_penumbraAngle << new core::param::FloatParam(0.0f);
    this->sl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->sl_position);
    this->MakeSlotAvailable(&this->sl_direction);
    this->MakeSlotAvailable(&this->sl_openingAngle);
    this->MakeSlotAvailable(&this->sl_penumbraAngle);
    this->MakeSlotAvailable(&this->sl_radius);

    // quad light
    this->ql_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->ql_edgeOne << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->ql_edgeTwo << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->ql_position);
    this->MakeSlotAvailable(&this->ql_edgeOne);
    this->MakeSlotAvailable(&this->ql_edgeTwo);

    // HDRI light
    this->hdri_up << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->hdri_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->hdri_evnfile << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->hdri_up);
    this->MakeSlotAvailable(&this->hdri_direction);
    this->MakeSlotAvailable(&this->hdri_evnfile);

    // General Renderer
    this->rd_epsilon << new core::param::FloatParam(1e-4f);
    this->rd_spp << new core::param::IntParam(1);
    this->rd_maxRecursion << new core::param::IntParam(10);
    this->rd_type << rdt;
    this->MakeSlotAvailable(&this->rd_epsilon);
    this->MakeSlotAvailable(&this->rd_spp);
    this->MakeSlotAvailable(&this->rd_maxRecursion);
    this->MakeSlotAvailable(&this->rd_type);

    //PathTracer
    this->rd_ptBackground << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->rd_ptBackground);

}

void ospray::OSPRayRenderer::renderTexture2D(vislib::graphics::gl::GLSLShader &shader, const uint32_t * fb, int &width, int &height) {

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, fb);

    glUniform1i(shader.ParameterLocation("tex"), 0);

    glBindVertexArray(this->vaScreen);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void ospray::OSPRayRenderer::setupTextureScreen() {

    // setup vertexarray
    float screenVertices[] = { 0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f, 1.0f,1.0f };

    glGenVertexArrays(1, &this->vaScreen);
    glGenBuffers(1, &this->vbo);

    glBindVertexArray(this->vaScreen);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glEnableVertexAttribArray(0);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * 2, screenVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);


    // setup texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &this->tex);
    glBindTexture(GL_TEXTURE_2D, this->tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void ospray::OSPRayRenderer::releaseTextureScreen() {
    glDeleteTextures(1, &this->tex);
    glDeleteBuffers(1, &this->vbo);
    glDeleteVertexArrays(1, &vaScreen);
}

void ospray::OSPRayRenderer::initOSPRay(OSPDevice &dvce) {

    if (device == NULL) {
        dvce = ospCreateDevice("default");
        ospDeviceCommit(dvce);
    }
}


void ospray::OSPRayRenderer::setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, OSPGeometry &geometry, const char * geometry_name, const char * renderer_name) {

    // create and setup renderer
    renderer = ospNewRenderer(renderer_name);
    camera = ospNewCamera("perspective");
    world = ospNewModel();
    ospSetObject(renderer, "model", world);
    ospSetObject(renderer, "camera", camera);
    geometry = ospNewGeometry(geometry_name);
    ospAddGeometry(world, geometry);

}

void ospray::OSPRayRenderer::setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, OSPVolume &volume, const char * volume_name, const char * renderer_name) {

    // create and setup renderer
    renderer = ospNewRenderer(renderer_name);
    camera = ospNewCamera("perspective");
    world = ospNewModel();
    ospSetObject(renderer, "model", world);
    ospSetObject(renderer, "camera", camera);
    volume = ospNewVolume(volume_name);
    //ospAddVolume(world, volume);  // should be changeable

}

void ospray::OSPRayRenderer::setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, const char * volume_name, const char * renderer_name) {
    // create and setup renderer
    renderer = ospNewRenderer(renderer_name);
    camera = ospNewCamera("perspective");
    world = ospNewModel();
    ospSetObject(renderer, "model", world);
    ospSetObject(renderer, "camera", camera);
}

void ospray::OSPRayRenderer::colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray) {
  
    float gray_max = *std::max_element(grayArray.begin(), grayArray.end());
    float gray_min = *std::min_element(grayArray.begin(), grayArray.end());

    for (auto &gray : grayArray) {
        float scaled_gray;
        if ((gray_max - gray_min) <= 1e-4f) {
            scaled_gray = 0;
        } else {
            scaled_gray = (gray - gray_min) / (gray_max - gray_min);
        }
        if (transferTable == NULL && tableSize == 0) {
            for (int i = 0; i < 3; i++) {
                rgbaArray.push_back((0.3 + scaled_gray) / 1.3);
            }
            rgbaArray.push_back(1.0f);
        } else {
            float exact_tf = (tableSize - 1) * scaled_gray;
            int floor = std::floor(exact_tf);
            float tail = exact_tf - (float)floor;
            floor *= 4;
            for (int i = 0; i < 4; i++) {
                float colorFloor = transferTable[floor + i];
                float colorCeil = transferTable[floor + i + 4];
                float finalColor = colorFloor + (colorCeil - colorFloor)*(tail);
                rgbaArray.push_back(finalColor);
            }
        }
    }
}


OSPTexture2D ospray::OSPRayRenderer::TextureFromFile(vislib::TString fileName) {

    fileName = vislib::sys::Path::Resolve(fileName);

    vislib::TString ext = vislib::TString("");
    size_t pos = fileName.FindLast('.');
    if (pos != std::string::npos)
        ext = fileName.Substring(pos + 1);

    FILE *file = fopen(vislib::StringA(fileName).PeekBuffer(), "rb");
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
            } ungetc(peekchar, file);

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
            } ungetc(peekchar, file);

            // read maxval
            int maxVal = -1;
            rc = fscanf(file, "%i", &maxVal);
            peekchar = getc(file);

            unsigned char* data;
            data = new unsigned char[width * height * 3];
            rc = fread(data, width*height * 3, 1, file);
            // flip in y, because OSPRay's textures have the origin at the lower left corner
            unsigned char *texels = (unsigned char *)data;
            for (int y = 0; y < height / 2; y++)
                for (int x = 0; x < width * 3; x++)
                    std::swap(texels[y*width * 3 + x], texels[(height - 1 - y)*width * 3 + x]);

            OSPTexture2D ret_tex = ospNewTexture2D({ width, height }, OSP_TEXTURE_RGB8, texels);
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
            char format[2] = { 0 };
            if (fscanf(file, "%c%c\n", &format[0], &format[1]) != 2)
                throw std::runtime_error("could not fscanf");

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
            float *texels = (float *)data;
            for (int y = 0; y < height / 2; ++y) {
                for (int x = 0; x < width * numChannels; ++x) {
                    // Scale the pixels by the scale factor
                    texels[y * width * numChannels + x] = texels[y * width * numChannels + x] * scaleFactor;
                    texels[(height - 1 - y) * width * numChannels + x] = texels[(height - 1 - y) * width * numChannels + x] * scaleFactor;
                    std::swap(texels[y * width * numChannels + x], texels[(height - 1 - y) * width * numChannels + x]);
                }
            }
            OSPTextureFormat type = OSP_TEXTURE_R8;

            if (numChannels == 1) type = OSP_TEXTURE_R32F;
            if (numChannels == 3) type = OSP_TEXTURE_RGB32F;
            if (numChannels == 4) type = OSP_TEXTURE_RGBA32F;

            OSPTexture2D ret_tex = ospNewTexture2D({ width, height }, type, texels);
            return ret_tex;
        } catch (std::runtime_error e) {
            std::cerr << e.what() << std::endl;
        }
    } else {
        std::cerr << "File type not supported. Only PPM and PFM file formats allowed." << std::endl;
    }
}

bool ospray::OSPRayRenderer::AbstractIsDirty() {
        if (this->AOsamples.IsDirty() ||
            this->AOtransparencyEnabled.IsDirty() ||
            this->AOdistance.IsDirty() ||
            this->extraSamles.IsDirty() ||
            this->lightIntensity.IsDirty() ||
            this->lightType.IsDirty() ||
            this->lightColor.IsDirty() ||
            this->shadows.IsDirty() ||
            this->dl_angularDiameter.IsDirty() ||
            this->dl_direction.IsDirty() ||
            this->dl_eye_direction.IsDirty() ||
            this->pl_position.IsDirty() ||
            this->pl_radius.IsDirty() ||
            this->sl_position.IsDirty() ||
            this->sl_direction.IsDirty() ||
            this->sl_openingAngle.IsDirty() ||
            this->sl_penumbraAngle.IsDirty() ||
            this->sl_radius.IsDirty() ||
            this->ql_position.IsDirty() ||
            this->ql_edgeOne.IsDirty() ||
            this->ql_edgeTwo.IsDirty() ||
            this->hdri_up.IsDirty() ||
            this->hdri_direction.IsDirty() ||
            this->hdri_evnfile.IsDirty() ||
            this->rd_epsilon.IsDirty() ||
            this->rd_spp.IsDirty() ||
            this->rd_maxRecursion.IsDirty() ||
            this->rd_ptBackground.IsDirty() ||
            this->framebufferIsDirty)
        {
            return true;
        } else {
            return false;
        }
}

void ospray::OSPRayRenderer::AbstractResetDirty() {
    this->AOsamples.ResetDirty();
    this->AOtransparencyEnabled.ResetDirty();
    this->AOdistance.ResetDirty();
    this->extraSamles.ResetDirty();
    this->lightIntensity.ResetDirty();
    this->lightType.ResetDirty();
    this->lightColor.ResetDirty();
    this->shadows.ResetDirty();
    this->dl_angularDiameter.ResetDirty();
    this->dl_direction.ResetDirty();
    this->dl_eye_direction.ResetDirty();
    this->pl_position.ResetDirty();
    this->pl_radius.ResetDirty();
    this->sl_position.ResetDirty();
    this->sl_direction.ResetDirty();
    this->sl_openingAngle.ResetDirty();
    this->sl_penumbraAngle.ResetDirty();
    this->sl_radius.ResetDirty();
    this->ql_position.ResetDirty();
    this->ql_edgeOne.ResetDirty();
    this->ql_edgeTwo.ResetDirty();
    this->hdri_up.ResetDirty();
    this->hdri_direction.ResetDirty();
    this->hdri_evnfile.ResetDirty();
    this->rd_epsilon.ResetDirty();
    this->rd_spp.ResetDirty();
    this->rd_maxRecursion.ResetDirty();
    this->rd_ptBackground.ResetDirty();
    this->framebufferIsDirty = false;
}

void ospray::OSPRayRenderer::OSPRayLights(OSPRenderer &renderer, core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    // create custom ospray light
    switch (this->lightType.Param<core::param::EnumParam>()->Value()) {
    case NONE:
        light = NULL;
        break;
    case DISTANTLIGHT:
        light = ospNewLight(renderer, "distant");
        if (this->dl_eye_direction.Param<core::param::BoolParam>()->Value() == true) {
            // take the light direction from the View3D
            //GLfloat lightdir[4];
            //glGetLightfv(GL_LIGHT0, GL_POSITION, lightdir);
            //ospSetVec3f(light, "direction", { lightdir[0], lightdir[1], lightdir[2] });
            ospSet3fv(light, "direction", cr->GetCameraParameters()->EyeDirection().PeekComponents());
        } else {
            ospSet3fv(light, "direction", this->dl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        }
        ospSet1f(light, "angularDiameter", this->dl_angularDiameter.Param<core::param::FloatParam>()->Value());
        break;
    case POINTLIGHT:
        light = ospNewLight(renderer, "point");
        ospSet3fv(light, "position", this->pl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        ospSet1f(light, "radius", this->pl_radius.Param<core::param::FloatParam>()->Value());
        break;
    case SPOTLIGHT:
        light = ospNewLight(renderer, "spot");
        ospSet3fv(light, "position", this->sl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        ospSet3fv(light, "direction", this->sl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        ospSet1f(light, "openingAngle", this->sl_openingAngle.Param<core::param::FloatParam>()->Value());
        ospSet1f(light, "penumbraAngle", this->sl_penumbraAngle.Param<core::param::FloatParam>()->Value());
        ospSet1f(light, "radius", this->sl_radius.Param<core::param::FloatParam>()->Value());
        break;
    case QUADLIGHT:
        light = ospNewLight(renderer, "quad");
        ospSet3fv(light, "position", this->ql_position.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        ospSet3fv(light, "edge1", this->ql_edgeOne.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        ospSet3fv(light, "edge2", this->ql_edgeTwo.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        break;
    case HDRILIGHT:
        light = ospNewLight(renderer, "hdri");
        ospSet3fv(light, "up", this->hdri_up.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        ospSet3fv(light, "dir", this->hdri_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents());
        if (this->hdri_evnfile.Param<core::param::FilePathParam>()->Value() != vislib::TString("")) {
            OSPTexture2D hdri_tex = this->TextureFromFile(this->hdri_evnfile.Param<core::param::FilePathParam>()->Value());
            ospSetObject(renderer, "backplate", hdri_tex);
        }
        break;
    case AMBIENTLIGHT:
        light = ospNewLight(renderer, "ambient");
        break;
    }

    if (light == NULL) {
        ospSetData(renderer, "lights", NULL);
    } else {
        ospSet1f(light, "intensity", this->lightIntensity.Param<core::param::FloatParam>()->Value());
        vislib::math::Vector<float, 3> lc = this->lightColor.Param<core::param::Vector3fParam>()->Value();
        ospSetVec3f(light, "color", { lc.X(), lc.Y(), lc.Z() });
        ospCommit(light);
        lightArray = ospNewData(1, OSP_OBJECT, &light, 0);
        ospSetData(renderer, "lights", lightArray);
    }
}

void ospray::OSPRayRenderer::RendererSettings(OSPRenderer &renderer) {
    // general renderer settings
    ospSet1f(renderer, "epsilon", this->rd_epsilon.Param<core::param::FloatParam>()->Value());
    ospSet1i(renderer, "spp", this->rd_spp.Param<core::param::IntParam>()->Value());
    ospSet1i(renderer, "maxDepth", this->rd_maxRecursion.Param<core::param::IntParam>()->Value());

    switch (this->rd_type.Param<core::param::EnumParam>()->Value()) {
    case SCIVIS:
        // scivis renderer settings
        ospSet1f(renderer, "aoTransparencyEnabled", this->AOtransparencyEnabled.Param<core::param::BoolParam>()->Value());
        ospSet1i(renderer, "aoSamples", this->AOsamples.Param<core::param::IntParam>()->Value());
        ospSet1i(renderer, "shadowsEnabled", this->shadows.Param<core::param::BoolParam>()->Value());
        ospSet1f(renderer, "aoOcclusionDistance", this->AOdistance.Param<core::param::FloatParam>()->Value());
        GLfloat bgcolor[4];
        glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
        ospSet3fv(renderer, "bgColor", bgcolor);
        /* Not implemented
        ospSet1i(renderer, "oneSidedLighting", 0);
        ospSet1i(renderer, "backgroundEnabled", 0);
        */
        break;
    case PATHTRACER:
        if (this->rd_ptBackground.Param<core::param::FilePathParam>()->Value() != vislib::TString("")) {
            OSPTexture2D bkgnd_tex = this->TextureFromFile(this->rd_ptBackground.Param<core::param::FilePathParam>()->Value());
            ospSetObject(renderer, "backplate", bkgnd_tex);
        }
        break;
    }

}

void ospray::OSPRayRenderer::setupOSPRayCamera(OSPCamera& camera, core::view::CallRender3D* cr) {


    // calculate image parts for e.g. screenshooter
    std::vector<float> imgStart(2, 0);
    std::vector<float> imgEnd(2, 0);
    imgStart[0] = cr->GetCameraParameters()->TileRect().GetLeft() / (float)cr->GetCameraParameters()->VirtualViewSize().GetWidth();
    imgStart[1] = cr->GetCameraParameters()->TileRect().GetBottom() / (float)cr->GetCameraParameters()->VirtualViewSize().GetHeight();

    imgEnd[0] = cr->GetCameraParameters()->TileRect().GetRight() / (float)cr->GetCameraParameters()->VirtualViewSize().GetWidth();
    imgEnd[1] = cr->GetCameraParameters()->TileRect().GetTop() / (float)cr->GetCameraParameters()->VirtualViewSize().GetHeight();

    // setup camera
    ospSet2fv(camera, "image_start", imgStart.data());
    ospSet2fv(camera, "image_end", imgEnd.data());
    ospSetf(camera, "aspect", cr->GetCameraParameters()->TileRect().AspectRatio());
    ospSet3fv(camera, "pos", cr->GetCameraParameters()->EyePosition().PeekCoordinates());
    ospSet3fv(camera, "dir", cr->GetCameraParameters()->EyeDirection().PeekComponents());
    ospSet3fv(camera, "up", cr->GetCameraParameters()->EyeUpVector().PeekComponents());
    ospSet1f(camera, "fovy", cr->GetCameraParameters()->ApertureAngle());

    //ospSet1i(camera, "architectural", 1);
    //ospSet1f(camera, "nearClip", cr->GetCameraParameters()->NearClip());
    //ospSet1f(camera, "farClip", cr->GetCameraParameters()->FarClip());
    //ospSet1f(camera, "apertureRadius", cr->GetCameraParameters()->ApertureAngle);
    //ospSet1f(camera, "focalDistance", cr->GetCameraParameters()->FocalDistance());

}

OSPFrameBuffer ospray::OSPRayRenderer::newFrameBuffer(osp::vec2i& imgSize, const OSPFrameBufferFormat format, const uint32_t frameBufferChannels) {
    OSPFrameBuffer frmbuff = ospNewFrameBuffer(imgSize, format, frameBufferChannels);
    this->framebufferIsDirty = true;
    return frmbuff;
}


ospray::OSPRayRenderer::~OSPRayRenderer(void) {
    if (light != NULL) ospRelease(light);
    if (lightArray != NULL) ospRelease(lightArray);
}

// helper function to write the rendered image as PPM file
void ospray::OSPRayRenderer::writePPM(const char *fileName, const osp::vec2i &size, const uint32_t *pixel) {
    //std::ofstream file;
    //file << "P6\n" << size.x << " " << size.y << "\n255\n";
    FILE *file = fopen(fileName, "wb");
    fprintf(file, "P6\n%i %i\n255\n", size.x, size.y);
    unsigned char *out = (unsigned char *)alloca(3 * size.x);
    for (int y = 0; y < size.y; y++) {
        const unsigned char *in = (const unsigned char *)&pixel[(size.y - 1 - y)*size.x];
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