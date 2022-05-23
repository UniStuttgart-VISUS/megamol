/*
 * AmbientOcclusionCalculator.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "AmbientOcclusionCalculator.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/AbstractPolynomImpl.h"
#include "vislib/math/ShallowVector.h"

#include <array>
#include <fstream>
#include <omp.h>

using namespace megamol;
using namespace megamol::molecularmaps;

/*
 * AmbientOcclusionCalculator::AmbientOcclusionCalculator
 */
AmbientOcclusionCalculator::AmbientOcclusionCalculator(void)
        : aoSampleMax(0)
        , colourSSBOHandle(0)
        , dirTexture(0)
        , lvlTexture(0)
        , mdc(nullptr)
        , normalSSBOHandle(0)
        , resultVector(std::vector<float>(0))
        , shaderChanged(false)
        , vertex_normals(nullptr)
        , vertexSSBOHandle(0)
        , vertices(nullptr)
        , volTexture(0)
        , volumeInitialized(false) {}

/*
 * AmbientOcclusionCalculator::~AmbientOcclusionCalculator
 */
AmbientOcclusionCalculator::~AmbientOcclusionCalculator(void) {
    this->aoComputeShader.Release();
    if (this->volTexture != 0) {
        glDeleteTextures(1, &this->volTexture);
    }
    if (this->dirTexture != 0) {
        glDeleteTextures(1, &this->dirTexture);
    }
    if (this->lvlTexture != 0) {
        glDeleteTextures(1, &this->lvlTexture);
    }

    if (this->colourSSBOHandle != 0) {
        glDeleteBuffers(1, &this->colourSSBOHandle);
    }
    if (this->normalSSBOHandle != 0) {
        glDeleteBuffers(1, &this->normalSSBOHandle);
    }
    if (this->vertexSSBOHandle != 0) {
        glDeleteBuffers(1, &this->vertexSSBOHandle);
    }
}

/*
 * AmbientOcclusionCalculator::calcDirections
 */
void AmbientOcclusionCalculator::calcDirections(void) {
    int sample = this->settings.numSampleDirections;
    std::vector<float> directions(sample * 4, 0.0f);
    float goldenAngle = static_cast<float>(M_PI * (3.0 - sqrt(5.0)));
    float phi = 0.0f;
    float theta = 0.0f;
    float stepZ = 1.0f / static_cast<float>(sample * 1.25);
    float z = 1.0f;
    float weight = 0.0f;

    for (int i = 0; i < sample; i++) {
        directions[4 * i + 2] = z;
        directions[4 * i + 3] = z;
        weight += z;
        theta = acos(directions[4 * i + 2]);
        directions[4 * i + 1] = sin(theta * cos(phi));
        directions[4 * i + 0] = sin(theta * sin(phi));
        phi += goldenAngle;
        z -= stepZ;
    }
    for (int i = 0; i < sample; i++) {
        directions[4 * i + 3] /= weight;
    }

    if (this->dirTexture != 0) {
        glDeleteTextures(1, &this->dirTexture);
        this->dirTexture = 0;
    }
    glGenTextures(1, &this->dirTexture);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_1D, this->dirTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, sample, 0, GL_RGBA, GL_FLOAT, directions.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
}

/*
 * AmbientOcclusionCalculator::calcLevels
 */
void AmbientOcclusionCalculator::calcLevels(float aoWidthX, float aoWidthY, float aoWidthZ, float diag) {
    int sample = this->settings.numSampleDirections;
    float distMax = this->settings.maxDist;
    distMax *= diag;
    float widthFactor = this->settings.angleFactor;
    float sampDist = this->settings.minDist;
    float falloff = this->settings.falloffFactor;
    int sampleMax = 100;
    std::vector<float> levels(sampleMax * 4, 0.0f);

    float sampleArea = static_cast<float>(2.0 * M_PI / static_cast<double>(sample));
    float sampleWidth = sqrt(sampleArea) * widthFactor;
    float minVoxelWidth = std::fminf(aoWidthX, std::fminf(aoWidthY, aoWidthZ));
    float dist = aoWidthX * sampDist;
    float step = minVoxelWidth / 2.0f;
    int counter = 0;

    for (counter = 0; counter < sampleMax && dist + step < distMax; counter++) {
        dist += step;
        float levelWidth = sampleWidth * dist;
        if (levelWidth < minVoxelWidth) {
            levelWidth = minVoxelWidth;
        }
        levels[counter * 4 + 0] = std::log2f(levelWidth / minVoxelWidth);
        levels[counter * 4 + 1] = dist;
        levels[counter * 4 + 2] = std::powf((1.0f - dist / distMax), falloff);
        levels[counter * 4 + 3] = levelWidth / minVoxelWidth;
        if (distMax - dist < levelWidth / 2.0f) {
            levels[counter * 4 + 3] = (distMax - (dist - levelWidth / 2.0f)) / minVoxelWidth;
        }
        step = levelWidth / 2.0f;
        dist += step * (1.0f + (levelWidth != minVoxelWidth) * sampleWidth);
    }
    /*for (int j = 0; j < counter; j++) {
            levels[j * 4 + 3] = levels[j * 4 + 3];
    }*/
    this->aoSampleMax = counter;

    levels.resize(counter * 4);

    if (this->lvlTexture != 0) {
        glDeleteTextures(1, &this->lvlTexture);
        this->lvlTexture = 0;
    }
    glGenTextures(1, &this->lvlTexture);

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_1D, this->lvlTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, counter, 0, GL_RGBA, GL_FLOAT, levels.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
}

/*
 * AmbientOcclusionCalculator::calculateVertexShadows
 */
const std::vector<float>* AmbientOcclusionCalculator::calculateVertexShadows(
    AmbientOcclusionCalculator::AOSettings settings) {
    if (this->vertices == nullptr)
        return nullptr;
    if (this->vertex_normals == nullptr)
        return nullptr;
    if (this->mdc == nullptr)
        return nullptr;

    // allocate enough space for the result
    if (this->settings.isDirty(settings) || this->resultVector.size() == 0 || this->shaderChanged) {
        size_t size = (this->vertices->size() / 3) * 4;
        this->resultVector.resize(size, 0.0f);
        this->resultVector.shrink_to_fit();

        this->resizeVolume(settings);
        this->settings = settings;

        this->createEmptyVolume(settings);
        this->createVolumeCPU(settings);

        this->uploadVertexData();
        this->calcDirections();

        auto bb = mdc->AccessBoundingBoxes().ObjectSpaceClipBox();
        float minOSx = bb.Left();
        float minOSy = bb.Bottom();
        float minOSz = bb.Back();
        float rangeOSx = bb.Width();
        float rangeOSy = bb.Height();
        float rangeOSz = bb.Depth();
        float aoWidthX = rangeOSx / static_cast<float>(this->settings.volSizeX);
        float aoWidthY = rangeOSy / static_cast<float>(this->settings.volSizeY);
        float aoWidthZ = rangeOSz / static_cast<float>(this->settings.volSizeZ);
        rangeOSx /= (1.0f - 2.0f / static_cast<float>(this->settings.volSizeX));
        rangeOSy /= (1.0f - 2.0f / static_cast<float>(this->settings.volSizeY));
        rangeOSz /= (1.0f - 2.0f / static_cast<float>(this->settings.volSizeZ));
        minOSx -= rangeOSx / static_cast<float>(this->settings.volSizeX);
        minOSy -= rangeOSy / static_cast<float>(this->settings.volSizeY);
        minOSz -= rangeOSz / static_cast<float>(this->settings.volSizeZ);

        float diag = (bb.GetRightTopFront() - bb.GetLeftBottomBack()).Length();
        this->calcLevels(aoWidthX, aoWidthY, aoWidthZ, diag);

        int n = static_cast<int>(this->vertices->size() / 3);

        this->aoComputeShader.Enable();
        this->aoComputeShader.SetParameter("aoSampFact", this->settings.evalFactor);
        this->aoComputeShader.SetParameter("vertexCount", n);
        this->aoComputeShader.SetParameter("sampleNum", this->settings.numSampleDirections);
        this->aoComputeShader.SetParameter("sampleMax", this->aoSampleMax);
        this->aoComputeShader.SetParameter("posOrigin", minOSx, minOSy, minOSz);
        this->aoComputeShader.SetParameter("posExtents", rangeOSx, rangeOSy, rangeOSz);

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_1D, this->dirTexture);
        this->aoComputeShader.SetParameter("directionTex", 5);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_1D, this->lvlTexture);
        this->aoComputeShader.SetParameter("levelTex", 6);

        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_3D, this->volTexture);
        this->aoComputeShader.SetParameter("aoVol", 7);

        glDispatchCompute((n / 512) + 1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        this->aoComputeShader.Disable();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, 0);

        this->readColourData();
        this->shaderChanged = false;
    }
    return &this->resultVector;
}

/*
 * AmbientOcclusionCalculator::clearStoredShadowData
 */
void AmbientOcclusionCalculator::clearStoredShadowData(void) {
    this->resultVector.clear();
}

/*
 * AmbientOcclusionCalculator::createEmptyVolume
 */
void AmbientOcclusionCalculator::createEmptyVolume(AmbientOcclusionCalculator::AOSettings settings) {
    if (this->volTexture == 0)
        return;
    std::vector<float> vol(settings.volSizeX * settings.volSizeY * settings.volSizeZ, 0.0f);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_3D, this->volTexture);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, settings.volSizeX, settings.volSizeY, settings.volSizeZ, GL_RED,
        GL_FLOAT, vol.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    std::array<float, 4> borderColour = {0.0f, 0.0f, 0.0f, 0.0f};
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColour.data());
    glBindTexture(GL_TEXTURE_3D, 0);
}

/*
 * AmbientOcclusionCalculator::createVolumeCPU
 */
void AmbientOcclusionCalculator::createVolumeCPU(AmbientOcclusionCalculator::AOSettings settings) {
    unsigned int sx = settings.volSizeX;
    unsigned int sy = settings.volSizeY;
    unsigned int sz = settings.volSizeZ;

    std::vector<std::vector<float>> vol(omp_get_max_threads(), std::vector<float>(sx * sy * sz, 0.0f));

    float minOSx = this->mdc->AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = this->mdc->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = this->mdc->AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = this->mdc->AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = this->mdc->AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = this->mdc->AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    float voxelVol =
        (rangeOSx / static_cast<float>(sx)) * (rangeOSy / static_cast<float>(sy)) * (rangeOSz / static_cast<float>(sz));

    int j;

#pragma omp parallel for
    for (j = 0; j < static_cast<int>(mdc->AtomCount()); j++) {
        vislib::math::ShallowVector<const float, 3> ppos(&mdc->AtomPositions()[j * 3]);

        int x = static_cast<int>(((ppos[0] - minOSx) / rangeOSx) * static_cast<float>(sx));
        if (x < 0)
            x = 0;
        else if (x >= static_cast<int>(sx))
            x = sx - 1;

        int y = static_cast<int>(((ppos[1] - minOSy) / rangeOSy) * static_cast<float>(sy));
        if (y < 0)
            y = 0;
        else if (y >= static_cast<int>(sy))
            y = sy - 1;

        int z = static_cast<int>(((ppos[2] - minOSz) / rangeOSz) * static_cast<float>(sz));
        if (z < 0)
            z = 0;
        else if (z >= static_cast<int>(sz))
            z = sz - 1;

        float sphereRadius = mdc->AtomTypes()[mdc->AtomTypeIndices()[j]].Radius();
        float sphereVolume = 4.0f / 3.0f * static_cast<float>(M_PI) * sphereRadius * sphereRadius * sphereRadius;
        vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (sphereVolume / voxelVol) * settings.genFac;
    }

#pragma omp parallel for
    for (j = 0; j < static_cast<int>(sx * sy * sz); j++) {
        for (int i = 1; i < omp_get_max_threads(); i++) {
            vol[0][j] += vol[i][j];
        }
        if (vol[0][j] > 1.0f) {
            vol[0][j] = 1.0f;
        }
    }

#ifdef DEBUG_WRITE
    std::ofstream file("aovolume.raw", std::ios::binary);
    file.write((char*)vol[0].data(), sizeof(float) * vol[0].size());
    file.close();
#endif /* DEBUG_WRITE */

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_3D, this->volTexture);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, sx, sy, sz, GL_RED, GL_FLOAT, vol[0].data());
    glGenerateMipmap(GL_TEXTURE_3D);
    glTexParameteri(GL_TEXTURE_3D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
    glBindTexture(GL_TEXTURE_3D, 0);
}

/*
 * AmbientOcclusionCalculator::getVertexShadows
 */
const std::vector<float>* AmbientOcclusionCalculator::getVertexShadows(void) {
    if (this->resultVector.size() > 0) {
        return &this->resultVector;
    }
    return nullptr;
}

/*
 * AmbientOcclusionCalculator::initialize
 */
bool AmbientOcclusionCalculator::initilialize(core::CoreInstance* instance, const std::vector<float>* vertices,
    const std::vector<float>* vertex_normals, protein_calls::MolecularDataCall* mdc) {

    this->vertices = vertices;
    this->vertex_normals = vertex_normals;
    this->mdc = mdc;

    // generate volume texture
    if (this->volTexture != 0) {
        glDeleteTextures(1, &this->volTexture);
        this->volTexture = 0;
    }
    glGenTextures(1, &this->volTexture);

    if (!this->loadShaders(instance))
        return false;

    // create SSBOs
    if (this->colourSSBOHandle != 0) {
        glDeleteBuffers(1, &this->colourSSBOHandle);
        this->colourSSBOHandle = 0;
    }
    glGenBuffers(1, &this->colourSSBOHandle);

    if (this->normalSSBOHandle != 0) {
        glDeleteBuffers(1, &this->normalSSBOHandle);
        this->normalSSBOHandle = 0;
    }
    glGenBuffers(1, &this->normalSSBOHandle);

    if (this->vertexSSBOHandle != 0) {
        glDeleteBuffers(1, &this->vertexSSBOHandle);
        this->vertexSSBOHandle = 0;
    }
    glGenBuffers(1, &this->vertexSSBOHandle);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->vertexSSBOHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->normalSSBOHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, this->colourSSBOHandle);

    return true;
}

/*
 * AmbientOcclusionCalculator::loadShaders
 */
bool AmbientOcclusionCalculator::loadShaders(core::CoreInstance* instance) {
    // load compute shader
    instance->ShaderSourceFactory().LoadBTF("aocompute", true);
    vislib::graphics::gl::ShaderSource compute;
    if (!instance->ShaderSourceFactory().MakeShaderSource("aocompute::compute", compute)) {
        return false;
    }

    try {
        if (!this->aoComputeShader.Compile(compute.Code(), compute.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile aocompute shader: Unknown error\n");
            return false;
        }
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile aocompute shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile aocompute shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile aocompute shader: Unknown exception\n");
        return false;
    }
    this->shaderChanged = true;
    return true;
}

/*
 * AmbientOcclusionCalculator::readColourData
 */
void AmbientOcclusionCalculator::readColourData(void) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->colourSSBOHandle);
    float* p = static_cast<float*>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, this->resultVector.size() * sizeof(float), GL_MAP_READ_BIT));
    std::memcpy(this->resultVector.data(), p, this->resultVector.size() * sizeof(float));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

/*
 * AmbientOcclusionCalculator::resizeVolume
 */
void AmbientOcclusionCalculator::resizeVolume(AmbientOcclusionCalculator::AOSettings settings) {
    if (this->volTexture == 0)
        return;
    std::vector<float> vol(settings.volSizeX * settings.volSizeY * settings.volSizeZ, 0.0f);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_3D, this->volTexture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, settings.volSizeX, settings.volSizeY, settings.volSizeZ, 0, GL_RED,
        GL_FLOAT, vol.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    std::array<float, 4> borderColour = {0.0f, 0.0f, 0.0f, 0.0f};
    glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColour.data());
    glBindTexture(GL_TEXTURE_3D, 0);
}

/*
 * AmbientOcclusionCalculator::uploadVertexData
 */
void AmbientOcclusionCalculator::uploadVertexData(void) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, this->vertexSSBOHandle);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER, this->vertices->size() * sizeof(float), this->vertices->data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, this->normalSSBOHandle);
    glBufferData(GL_SHADER_STORAGE_BUFFER, this->vertex_normals->size() * sizeof(float), this->vertex_normals->data(),
        GL_DYNAMIC_DRAW);
    size_t n = (this->vertices->size() / 3) * 4;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, this->colourSSBOHandle);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
}
