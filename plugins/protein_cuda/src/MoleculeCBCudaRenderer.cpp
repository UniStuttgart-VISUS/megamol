/*
 * MoleculeCBCudaRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "MoleculeCBCudaRenderer.h"

#define _USE_MATH_DEFINES 1
#define SFB_DEMO

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "vislib/math/Matrix.h"
#include "vislib/sys/sysfunctions.h"
#include <iostream>
#include <math.h>

#include "helper_cuda.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"
#include <cuda_gl_interop.h>

extern "C" void cudaInit(int argc, char** argv);
extern "C" void cudaGLInit(int argc, char** argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;
using namespace megamol::core::utility::log;

/*
 * MoleculeCBCudaRenderer::MoleculeCBCudaRenderer
 */
MoleculeCBCudaRenderer::MoleculeCBCudaRenderer(void)
        : Renderer3DModuleGL()
        , molDataCallerSlot("getData", "Connects the protein SES rendering with PDB data source")
        , probeRadiusParam("probeRadius", "Probe Radius")
        , opacityParam("opacity", "Atom opacity")
        , stepsParam("steps", "Drawing steps")
        , probeRadius(1.4f)
        , atomNeighborCount(64)
        , cudaInitalized(false)
        , probeNeighborCount(32)
        , sphereVAO_(0)
        , torusVAO_(0)
        , sphericalTriangleVAO_(0)
        , texHeight(0)
        , texWidth(0)
        , width(0)
        , height(0)
        , setCUDAGLDevice(true)
        , sphereShader_(nullptr)
        , torusShader_(nullptr)
        , sphericalTriangleShader_(nullptr)
        , singTex_(nullptr) {
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);

    // setup probe radius parameter
    this->probeRadiusParam.SetParameter(new param::FloatParam(this->probeRadius, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->probeRadiusParam);

    // setup atom opacity parameter
    this->opacityParam.SetParameter(new param::FloatParam(0.4f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->opacityParam);

    // DEBUG
    this->stepsParam.SetParameter(new param::IntParam(5, 0, 100));
    this->MakeSlotAvailable(&this->stepsParam);
}


/*
 * MoleculeCBCudaRenderer::~MoleculeCBCudaRenderer
 */
MoleculeCBCudaRenderer::~MoleculeCBCudaRenderer(void) {
    this->Release();
}


/*
 * protein_cuda::MoleculeCBCudaRenderer::release
 */
void protein_cuda::MoleculeCBCudaRenderer::release(void) {

    if (this->cudaInitalized) {
        freeArray(m_dPos);
        freeArray(m_dSortedPos);
        freeArray(m_dSortedProbePos);
        freeArray(m_dNeighborCount);
        freeArray(m_dNeighbors);
        freeArray(m_dSmallCircles);
        freeArray(m_dSmallCircleVisible);
        freeArray(m_dSmallCircleVisibleScan);
        freeArray(m_dArcs);
        freeArray(m_dArcIdxK);
        freeArray(m_dArcCount);
        freeArray(m_dArcCountScan);
        freeArray(m_dGridParticleHash);
        freeArray(m_dGridParticleIndex);
        freeArray(m_dGridProbeHash);
        freeArray(m_dGridProbeIndex);
        freeArray(m_dCellStart);
        freeArray(m_dCellEnd);
    }
}


/*
 * MoleculeCBCudaRenderer::create
 */
bool MoleculeCBCudaRenderer::create(void) {
    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

        sphereShader_ = core::utility::make_shared_glowl_shader("sphere", shdr_options,
            std::filesystem::path("protein_cuda/molecule_cb/mcbc_sphere.vert.glsl"),
            std::filesystem::path("protein_cuda/molecule_cb/mcbc_sphere.frag.glsl"));

        torusShader_ = core::utility::make_shared_glowl_shader("torus", shdr_options,
            std::filesystem::path("protein_cuda/molecule_cb/mcbc_torus.vert.glsl"),
            std::filesystem::path("protein_cuda/molecule_cb/mcbc_torus.frag.glsl"));

        sphericalTriangleShader_ = core::utility::make_shared_glowl_shader("sphericalTriangle", shdr_options,
            std::filesystem::path("protein_cuda/molecule_cb/mcbc_sphericaltriangle.vert.glsl"),
            std::filesystem::path("protein_cuda/molecule_cb/mcbc_sphericaltriangle.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[MoleculeCBCudaRenderer] %s", ex.what());
        return false;
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[MoleculeCBCudaRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[MoleculeCBCudaRenderer] Unable to compile shader: Unknown exception.");
        return false;
    }

    return true;
}

/*
 * MoleculeCBCudaRenderer::GetExtents
 */
bool MoleculeCBCudaRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent))
        return false;

    call.AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    call.SetTimeFramesCount(mol->FrameCount());

    return true;
}


/*
 * MoleculeCBCudaRenderer::Render
 */
bool MoleculeCBCudaRenderer::Render(mmstd_gl::CallRender3DGL& call) {
    // get camera information
    this->cameraInfo = call.GetCamera();
    this->width = call.GetViewResolution().x;
    this->height = call.GetViewResolution().y;

    float callTime = call.Time();

    // get pointer to call
    MolecularDataCall* mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    // if something went wrong --> return
    if (!mol)
        return false;

    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>(callTime));
    // execute the call
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    // update parameters
    this->UpdateParameters(mol);

    // try to initialize CUDA
    if (!this->cudaInitalized) {
        cudaInitalized = this->initCuda(mol, 16, &call);
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "%s: CUDA initialization: %i", this->ClassName(), cudaInitalized);
    }

    // ==================== Scale & Translate ====================
    glPushMatrix();

    // ==================== Start actual rendering ====================
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);

    this->ContourBuildupCuda(mol);

    glPopMatrix();
    return true;
}

/*
 * CUDA version of contour buildup algorithm
 */
void MoleculeCBCudaRenderer::ContourBuildupCuda(MolecularDataCall* mol) {
    // do nothing if cuda was not initialized first
    if (!this->cudaInitalized)
        return;

    // write atom positions to VBO
    this->writeAtomPositionsVBO(mol);

    // update constants
    this->params.probeRadius = this->probeRadius;
    setParameters(&this->params);

    // map OpenGL buffer object for writing from CUDA
    float* atomPosPtr;
    cudaGLMapBufferObject((void**)&atomPosPtr, buffers_[static_cast<int>(Buffers::ATOM_POS)]->getName());

    // calculate grid hash
    calcHash(m_dGridParticleHash, m_dGridParticleIndex,
        //m_dPos,
        atomPosPtr, this->numAtoms);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, this->numAtoms);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPos, m_dGridParticleHash, m_dGridParticleIndex,
        //m_dPos,
        atomPosPtr, this->numAtoms, this->numGridCells);

    // unmap buffer object
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::ATOM_POS)]->getName());

    // find neighbors of all atoms and compute small circles
    findNeighborsCB(m_dNeighborCount, m_dNeighbors, m_dSmallCircles, m_dSortedPos, m_dCellStart, m_dCellEnd,
        this->numAtoms, this->atomNeighborCount, this->numGridCells);

    // find and remove unnecessary small circles
    removeCoveredSmallCirclesCB(m_dSmallCircles, m_dSmallCircleVisible, m_dNeighborCount, m_dNeighbors, m_dSortedPos,
        numAtoms, params.maxNumNeighbors);

    cudaMemset(m_dArcCount, 0, this->numAtoms * this->atomNeighborCount * sizeof(uint));

    // compute all arcs for all small circles
    computeArcsCB(m_dSmallCircles, m_dSmallCircleVisible, m_dNeighborCount, m_dNeighbors, m_dSortedPos, m_dArcs,
        m_dArcCount, numAtoms, params.maxNumNeighbors);

    // ---------- vertex buffer object generation (for rendering) ----------
    // count total number of small circles
    scanParticles(m_dSmallCircleVisible, m_dSmallCircleVisibleScan, this->numAtoms * this->atomNeighborCount);
    // get total number of small circles
    uint numSC = 0;
    uint lastSC = 0;
    checkCudaErrors(
        cudaMemcpy((void*)&numSC, (void*)(m_dSmallCircleVisibleScan + (this->numAtoms * this->atomNeighborCount) - 1),
            sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy((void*)&lastSC, (void*)(m_dSmallCircleVisible + (this->numAtoms * this->atomNeighborCount) - 1),
            sizeof(uint), cudaMemcpyDeviceToHost));
    numSC += lastSC;

    // count total number of arcs
    scanParticles(m_dArcCount, m_dArcCountScan, this->numAtoms * this->atomNeighborCount);
    // get total number of probes
    uint numProbes = 0;
    uint lastProbeCnt = 0;
    checkCudaErrors(
        cudaMemcpy((void*)&numProbes, (void*)(m_dArcCountScan + (this->numAtoms * this->atomNeighborCount) - 1),
            sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)&lastProbeCnt,
        (void*)(m_dArcCount + (this->numAtoms * this->atomNeighborCount) - 1), sizeof(uint), cudaMemcpyDeviceToHost));
    numProbes += lastProbeCnt;

    // resize torus buffer objects
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_POS)]->getName());
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_VS)]->getName());
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->getName());
    buffers_[static_cast<int>(Buffers::TORUS_POS)]->rebuffer(nullptr, numSC * 4 * sizeof(float));
    buffers_[static_cast<int>(Buffers::TORUS_VS)]->rebuffer(nullptr, numSC * 4 * sizeof(float));
    buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->rebuffer(nullptr, numSC * 4 * sizeof(float));
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_POS)]->getName());
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_VS)]->getName());
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->getName());

    // resize probe buffer object
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::PROBE_POS)]->getName());
    buffers_[static_cast<int>(Buffers::PROBE_POS)]->rebuffer(nullptr, numProbes * 4 * sizeof(float));
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::PROBE_POS)]->getName());

    // resize spherical triangle buffer objects
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->getName());
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->getName());
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->getName());
    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->rebuffer(nullptr, numProbes * 4 * sizeof(float));
    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->rebuffer(nullptr, numProbes * 4 * sizeof(float));
    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->rebuffer(nullptr, numProbes * 4 * sizeof(float));
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->getName());
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->getName());
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->getName());

    // map probe buffer object for writing from CUDA
    float* probePosPtr;
    cudaGLMapBufferObject((void**)&probePosPtr, buffers_[static_cast<int>(Buffers::PROBE_POS)]->getName());

    // map spherical triangle buffer objects for writing from CUDA
    float *sphereTriaVec1Ptr, *sphereTriaVec2Ptr, *sphereTriaVec3Ptr;
    cudaGLMapBufferObject(
        (void**)&sphereTriaVec1Ptr, buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->getName());
    cudaGLMapBufferObject(
        (void**)&sphereTriaVec2Ptr, buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->getName());
    cudaGLMapBufferObject(
        (void**)&sphereTriaVec3Ptr, buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->getName());

    // map torus buffer objects for writing from CUDA
    float *torusPosPtr, *torusVSPtr, *torusAxisPtr;
    cudaGLMapBufferObject((void**)&torusPosPtr, buffers_[static_cast<int>(Buffers::TORUS_POS)]->getName());
    cudaGLMapBufferObject((void**)&torusVSPtr, buffers_[static_cast<int>(Buffers::TORUS_VS)]->getName());
    cudaGLMapBufferObject((void**)&torusAxisPtr, buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->getName());

    // compute vertex buffer objects for probe positions
    writeProbePositionsCB(probePosPtr, sphereTriaVec1Ptr, sphereTriaVec2Ptr, sphereTriaVec3Ptr, torusPosPtr, torusVSPtr,
        torusAxisPtr, m_dNeighborCount, m_dNeighbors, m_dSortedPos, m_dArcs, m_dArcCount, m_dArcCountScan,
        m_dSmallCircleVisible, m_dSmallCircleVisibleScan, m_dSmallCircles, this->numAtoms, this->atomNeighborCount);

    // unmap torus buffer objects
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::TORUS_POS)]->getName());
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::TORUS_VS)]->getName());
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->getName());

    // unmap spherical triangle buffer objects
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->getName());
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->getName());
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->getName());

    // ---------- singularity handling ----------
    // calculate grid hash
    calcHash(m_dGridProbeHash, m_dGridProbeIndex, probePosPtr, numProbes);

    // sort probes based on hash
    sortParticles(m_dGridProbeHash, m_dGridProbeIndex, numProbes);

    // reorder particle arrays into sorted order and find start and end of each cell
    reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedProbePos, m_dGridProbeHash, m_dGridProbeIndex,
        probePosPtr, numProbes, this->numGridCells);

    // unmap probe buffer object
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::PROBE_POS)]->getName());

    // resize texture coordinate buffer object
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::TEX_COORD)]->getName());
    cudaGLUnregisterBufferObject(buffers_[static_cast<int>(Buffers::SING_TEX)]->getName());
    buffers_[static_cast<int>(Buffers::TEX_COORD)]->rebuffer(nullptr, numProbes * 3 * sizeof(float));
    buffers_[static_cast<int>(Buffers::SING_TEX)]->rebuffer(
        nullptr, numProbes * this->probeNeighborCount * 3 * sizeof(float));
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TEX_COORD)]->getName());
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SING_TEX)]->getName());

    // map texture coordinate buffer object for writing from CUDA
    float* texCoordPtr;
    cudaGLMapBufferObject((void**)&texCoordPtr, buffers_[static_cast<int>(Buffers::TEX_COORD)]->getName());
    // map singularity texture buffer object for writing from CUDA
    float* singTexPtr;
    cudaGLMapBufferObject((void**)&singTexPtr, buffers_[static_cast<int>(Buffers::SING_TEX)]->getName());

    // find all intersecting probes for each probe and write them to a texture
    writeSingularityTextureCB(texCoordPtr, singTexPtr, m_dSortedProbePos, m_dGridProbeIndex, m_dCellStart, m_dCellEnd,
        numProbes, this->probeNeighborCount, this->numGridCells);

    //copyArrayFromDevice( m_hPos, m_dSortedProbePos, 0, sizeof(float)*4);
    //std::cout << "probe: " << m_hPos[0] << ", " << m_hPos[1] << ", " << m_hPos[2] << " r = " << m_hPos[3] << std::endl;
    //copyArrayFromDevice( m_hPos, singTexPtr, 0, sizeof(float)*3*this->probeNeighborCount);
    //for( unsigned int i = 0; i < this->probeNeighborCount; i++ ) {
    //    std::cout << "neighbor probe " << i << ": " << m_hPos[i*3] << " " << m_hPos[i*3+1] << " " << m_hPos[i*3+2] << std::endl;
    //}

    // unmap texture coordinate buffer object
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::TEX_COORD)]->getName());
    // unmap singularity texture buffer object
    cudaGLUnmapBufferObject(buffers_[static_cast<int>(Buffers::SING_TEX)]->getName());

    // copy PBO to texture
    buffers_[static_cast<int>(Buffers::SING_TEX)]->bind();
    glEnable(GL_TEXTURE_2D);
    singTex_->bindTexture();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (numProbes / params.texSize + 1) * this->probeNeighborCount,
        numProbes % params.texSize, GL_RGB, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    ///////////////////////////////////////////////////////////////////////////
    // RENDER THE SOLVENT EXCLUDED SURFACE
    ///////////////////////////////////////////////////////////////////////////
    glm::mat4 view = this->cameraInfo.getViewMatrix();
    glm::mat4 proj = this->cameraInfo.getProjectionMatrix();
    glm::mat4 mvp = proj * view;
    glm::mat4 mvpinv = glm::inverse(mvp);
    glm::mat4 mvptrans = glm::transpose(mvp);
    glm::mat4 invview = glm::inverse(view);

    // do actual rendering
    glm::vec4 viewportStuff(0.0f, 0.0f, this->width, this->height);
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    auto cp = this->cameraInfo.getPose();

    // enable and set up sphere shader
    sphereShader_->use();
    sphereShader_->setUniform("viewAttr", viewportStuff);
    sphereShader_->setUniform("camIn", cp.direction);
    sphereShader_->setUniform("camRight", cp.right);
    sphereShader_->setUniform("camUp", cp.up);
    sphereShader_->setUniform("mvp", mvp);
    sphereShader_->setUniform("invview", invview);
    sphereShader_->setUniform("mvpinv", mvpinv);
    sphereShader_->setUniform("mvptrans", mvptrans);

    // render atoms VBO
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glBindVertexArray(sphereVAO_);
    glDrawArrays(GL_POINTS, 0, mol->AtomCount());
    glBindVertexArray(0);

    glDisable(GL_BLEND);

    // disable sphere shader
    glUseProgram(0);

    // get clear color (i.e. background color) for fogging
    float* clearColor = new float[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, clearColor);
    vislib::math::Vector<float, 3> fogCol(clearColor[0], clearColor[1], clearColor[2]);
    delete[] clearColor;

    auto cv = this->cameraInfo.get<core::view::Camera::PerspectiveParameters>();

    /////////////////////////////////////////////////
    // ray cast the spherical triangles on the GPU //
    /////////////////////////////////////////////////
    // bind texture
    singTex_->bindTexture();
    glBindVertexArray(sphericalTriangleVAO_);
    // enable spherical triangle shader
    sphericalTriangleShader_->use();
    // set shader variables
    sphericalTriangleShader_->setUniform("viewAttr", viewportStuff);
    sphericalTriangleShader_->setUniform("camIn", cp.direction);
    sphericalTriangleShader_->setUniform("camRight", cp.right);
    sphericalTriangleShader_->setUniform("camUp", cp.up);
    sphericalTriangleShader_->setUniform("zValues", 0.5f, cv.near_plane, cv.far_plane);
    sphericalTriangleShader_->setUniform("fogCol", fogCol.GetX(), fogCol.GetY(), fogCol.GetZ());
    sphericalTriangleShader_->setUniform("alpha", 1.0f);
    sphericalTriangleShader_->setUniform("mvp", mvp);
    sphericalTriangleShader_->setUniform("invview", invview);
    sphericalTriangleShader_->setUniform("mvpinv", mvpinv);
    sphericalTriangleShader_->setUniform("mvptrans", mvptrans);

    glDrawArrays(GL_POINTS, 0, numProbes);

    glBindVertexArray(0);
    glUseProgram(0);
    // unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);

    //////////////////////////////////
    // ray cast the tori on the GPU //
    //////////////////////////////////

    torusShader_->use();
    glBindVertexArray(torusVAO_);

    torusShader_->setUniform("viewAttr", viewportStuff);
    torusShader_->setUniform("camIn", cp.direction);
    torusShader_->setUniform("camRight", cp.right);
    torusShader_->setUniform("camUp", cp.up);
    torusShader_->setUniform("zValues", 0.5f, cv.near_plane, cv.far_plane);
    torusShader_->setUniform("fogCol", fogCol.GetX(), fogCol.GetY(), fogCol.GetZ());
    torusShader_->setUniform("alpha", 1.0f);
    torusShader_->setUniform("mvp", mvp);
    torusShader_->setUniform("invview", invview);
    torusShader_->setUniform("mvpinv", mvpinv);
    torusShader_->setUniform("mvptrans", mvptrans);

    glDrawArrays(GL_POINTS, 0, numSC);

    glBindVertexArray(0);
    glUseProgram(0);
}

/*
 * update parameters
 */
void MoleculeCBCudaRenderer::UpdateParameters(const MolecularDataCall* mol) {
    // color table param
    if (this->probeRadiusParam.IsDirty()) {
        this->probeRadius = this->probeRadiusParam.Param<param::FloatParam>()->Value();
        this->probeRadiusParam.ResetDirty();
    }
}


/*
 * MoleculeCBCudaRenderer::deinitialise
 */
void MoleculeCBCudaRenderer::deinitialise(void) {}

/*
 * Initialize CUDA
 */
bool MoleculeCBCudaRenderer::initCuda(MolecularDataCall* mol, uint gridDim, mmstd_gl::CallRender3DGL* cr3d) {
    // set number of atoms
    this->numAtoms = mol->AtomCount();

    if (setCUDAGLDevice) {
        cudaGLSetGLDevice(cudaUtilGetMaxGflopsDeviceId());
        printf("cudaGLSetGLDevice: %s\n", cudaGetErrorString(cudaGetLastError()));
        setCUDAGLDevice = false;
    }

    // set grid dimensions
    this->gridSize.x = this->gridSize.y = this->gridSize.z = gridDim;
    this->numGridCells = this->gridSize.x * this->gridSize.y * this->gridSize.z;
    float3 worldSize = make_float3(mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(), mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth());
    this->gridSortBits = 18; // increase this for larger grids

    // set parameters
    this->params.gridSize = this->gridSize;
    this->params.numCells = this->numGridCells;
    this->params.numBodies = this->numAtoms;
    //this->params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    this->params.worldOrigin = make_float3(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetX(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetY(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().GetZ());
    this->params.cellSize =
        make_float3(worldSize.x / this->gridSize.x, worldSize.y / this->gridSize.y, worldSize.z / this->gridSize.z);
    this->params.probeRadius = this->probeRadius;
    this->params.maxNumNeighbors = this->atomNeighborCount;

    // allocate host storage
    hPos_.clear();
    hPos_.resize(this->numAtoms, glm::vec4(0.0f));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * this->numAtoms;
    // array for atom positions
    allocateArray((void**)&m_dPos, memSize);
    //cudaMalloc(  (void**)&m_dPos, memSize);
    //cudaError e;
    //e = cudaGetLastError();

    //cuMemGetInfo
    //uint free, total;
    //cuMemGetInfo( &free, &total);
    //megamol::core::utility::log::Log::DefaultLog.WriteMsg( megamol::core::utility::log::Log::LEVEL_ERROR,
    //    "Free GPU Memory: %i / %i (MB)", free / ( 1024 * 1024), total / ( 1024 * 1024));
    // array for sorted atom positions
    allocateArray((void**)&m_dSortedPos, memSize);
    // array for sorted atom positions
    allocateArray((void**)&m_dSortedProbePos, memSize * this->atomNeighborCount);
    // array for the counted number of atoms
    allocateArray((void**)&m_dNeighborCount, this->numAtoms * sizeof(uint));
    // array for the neighbor atoms
    allocateArray((void**)&m_dNeighbors, this->numAtoms * this->atomNeighborCount * sizeof(uint));
    // array for the small circles
    allocateArray((void**)&m_dSmallCircles, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float));
    // array for the small circle visibility
    allocateArray((void**)&m_dSmallCircleVisible, this->numAtoms * this->atomNeighborCount * sizeof(uint));
    // array for the small circle visibility prefix sum
    allocateArray((void**)&m_dSmallCircleVisibleScan, this->numAtoms * this->atomNeighborCount * sizeof(uint));
    // array for the arcs
    allocateArray(
        (void**)&m_dArcs, this->numAtoms * this->atomNeighborCount * this->atomNeighborCount * 4 * sizeof(float));
    // array for the arcs
    allocateArray(
        (void**)&m_dArcIdxK, this->numAtoms * this->atomNeighborCount * this->atomNeighborCount * sizeof(uint));
    // array for the arc count
    allocateArray((void**)&m_dArcCount, this->numAtoms * this->atomNeighborCount * sizeof(uint));
    // array for the arc count scan (prefix sum)
    allocateArray((void**)&m_dArcCountScan, this->numAtoms * this->atomNeighborCount * sizeof(uint));

    allocateArray((void**)&m_dGridParticleHash, this->numAtoms * sizeof(uint));
    allocateArray((void**)&m_dGridParticleIndex, this->numAtoms * sizeof(uint));

    allocateArray((void**)&m_dGridProbeHash, this->numAtoms * this->atomNeighborCount * sizeof(uint));
    allocateArray((void**)&m_dGridProbeIndex, this->numAtoms * this->atomNeighborCount * sizeof(uint));

    allocateArray((void**)&m_dCellStart, this->numGridCells * sizeof(uint));
    allocateArray((void**)&m_dCellEnd, this->numGridCells * sizeof(uint));

    // clear all buffers
    for (auto& e : buffers_) {
        e = nullptr;
    }

    // re-create all buffers
    buffers_[static_cast<int>(Buffers::PROBE_POS)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::PROBE_POS)]->getName());

    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->getName());

    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->getName());

    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->getName());

    buffers_[static_cast<int>(Buffers::TORUS_POS)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_POS)]->getName());

    buffers_[static_cast<int>(Buffers::TORUS_VS)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_VS)]->getName());

    buffers_[static_cast<int>(Buffers::TORUS_AXIS)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * this->atomNeighborCount * 4 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->getName());

    // get maximum texture size
    GLint texSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texSize);
    texHeight = vislib::math::Min<int>(this->numAtoms * 3, texSize);
    texWidth = this->probeNeighborCount * ((this->numAtoms * 3) / texSize + 1);
    this->params.texSize = texSize;

    // create singularity texture
    std::vector<std::pair<GLenum, GLint>> int_parameters = {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
        {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE}, {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
        {GL_TEXTURE_MAG_FILTER, GL_NEAREST}};
    std::vector<std::pair<GLenum, GLfloat>> float_parameters;
    glowl::TextureLayout tx_layout{GL_RGB32F, static_cast<int>(texWidth), static_cast<int>(texHeight), 1, GL_RGB,
        GL_FLOAT, 1, int_parameters, float_parameters};
    singTex_ = std::make_unique<glowl::Texture2D>("molecule_cbc_singTex", tx_layout, nullptr);

    // create PBO
    buffers_[static_cast<int>(Buffers::SING_TEX)] = std::make_unique<glowl::BufferObject>(
        GL_PIXEL_UNPACK_BUFFER, nullptr, this->texWidth * texHeight * 3 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::SING_TEX)]->getName());

    // create texture coordinate buffer object
    buffers_[static_cast<int>(Buffers::TEX_COORD)] = std::make_unique<glowl::BufferObject>(
        GL_ARRAY_BUFFER, nullptr, this->numAtoms * 3 * 3 * sizeof(float), GL_DYNAMIC_DRAW);
    cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::TEX_COORD)]->getName());

    // set parameters
    setParameters(&this->params);

    // create VAOs
    if (sphericalTriangleVAO_ == 0) {
        glGenVertexArrays(1, &sphericalTriangleVAO_);
    }
    glBindVertexArray(sphericalTriangleVAO_);

    buffers_[static_cast<int>(Buffers::PROBE_POS)]->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_1)]->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_2)]->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::SPHERE_TRIA_VEC_3)]->bind();
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::TEX_COORD)]->bind();
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    if (torusVAO_ == 0) {
        glGenVertexArrays(1, &torusVAO_);
    }
    glBindVertexArray(torusVAO_);

    buffers_[static_cast<int>(Buffers::TORUS_POS)]->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::TORUS_AXIS)]->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    buffers_[static_cast<int>(Buffers::TORUS_VS)]->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    return true;
}

/*
 * Write atom positions and radii to an array for processing in CUDA
 */
void MoleculeCBCudaRenderer::writeAtomPositions(const MolecularDataCall* mol) {
    // write atoms to array
    int p = 0;
    for (unsigned int cnt = 0; cnt < mol->AtomCount(); ++cnt) {
        // write pos and rad to array
        hPos_[cnt] = glm::vec4(
            glm::make_vec3(&mol->AtomPositions()[cnt * 3]), mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius());
    }

    // setArray( POSITION, m_hPos, 0, this->numAtoms);
    //copyArrayToDevice(this->m_dPos, this->m_hPos, 0, this->numAtoms * 4 * sizeof(float));
    copyArrayToDevice(this->m_dPos, hPos_.data(), 0, hPos_.size() * sizeof(glm::vec4));
}

/*
 * Write atom positions and radii to a VBO for processing in CUDA
 */
void MoleculeCBCudaRenderer::writeAtomPositionsVBO(MolecularDataCall* mol) {
    // write atoms to array
    for (unsigned int cnt = 0; cnt < mol->AtomCount(); cnt++) {
        // write pos and rad to array
        hPos_[cnt] = glm::vec4(
            glm::make_vec3(&mol->AtomPositions()[cnt * 3]), mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius());
    }

    if (buffers_[static_cast<int>(Buffers::ATOM_POS)] == nullptr) {
        buffers_[static_cast<int>(Buffers::ATOM_POS)] = std::make_unique<glowl::BufferObject>(
            GL_ARRAY_BUFFER, hPos_.data(), mol->AtomCount() * 4 * sizeof(float), GL_DYNAMIC_DRAW);
        cudaGLRegisterBufferObject(buffers_[static_cast<int>(Buffers::ATOM_POS)]->getName());
    } else {
        buffers_[static_cast<int>(Buffers::ATOM_POS)]->rebuffer(hPos_.data(), mol->AtomCount() * 4 * sizeof(float));
    }

    if (sphereVAO_ == 0) {
        glGenVertexArrays(1, &sphereVAO_);
        glBindVertexArray(sphereVAO_);

        buffers_[static_cast<int>(Buffers::ATOM_POS)]->bind();
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableVertexAttribArray(0);
    }
}
