/*
 * ParticleWorker.cpp
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "ParticleWorker.h"

#include <cfloat>
#include <climits>

#include "OpenGL_Context.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "vislib/StringTokeniser.h"

using namespace megamol::core;
using namespace megamol::datatools_gl::misc;


/*
 * ParticleWorker::ParticleWorker
 */
ParticleWorker::ParticleWorker(void)
        : inParticlesDataSlot("inPartData", "Input for particle data")
        , outParticlesDataSlot("outPartData", "Output of particle data")
        , glClusterInfos(0)
// glParticleList(0),
// glPrefixIn(0), glPrefixOut(0)
{

    this->inParticlesDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inParticlesDataSlot);

    this->outParticlesDataSlot.SetCallback("MultiParticleDataCall", "GetData", &ParticleWorker::getDataCallback);
    this->outParticlesDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &ParticleWorker::getExtentCallback);
    this->MakeSlotAvailable(&this->outParticlesDataSlot);
}


/*
 * ParticleWorker::~ParticleWorker
 */
ParticleWorker::~ParticleWorker(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * ParticleWorker::create
 */
bool ParticleWorker::create(void) {

    using namespace megamol::core::utility::log;

    ASSERT(IsAvailable());

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isVersionGEQ(4, 3))
        return false;

    if (!this->GetCoreInstance())
        return false;

    auto const shader_options =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        this->shaderOnClusterComputation = core::utility::make_glowl_shader(
            "shaderOnClusterComputation", shader_options, "datatools_gl/particleWorker_work_on_clusters.comp.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("ParticleWorker: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


/*
 * ParticleWorker::release
 */
void ParticleWorker::release(void) {
    glDeleteBuffersARB(static_cast<GLsizei>(glVB.Count()), glVB.PeekElements());
    glDeleteVertexArrays(static_cast<GLsizei>(glVAO.Count()), glVAO.PeekElements());
    this->shaderOnClusterComputation.reset();
}

/*
 * ParticleWorker::getData
 */
bool ParticleWorker::getDataCallback(Call& call) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* outMpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (outMpdc == NULL)
        return false;

    MultiParticleDataCall* inMpdc = this->inParticlesDataSlot.CallAs<MultiParticleDataCall>();

    if (!(*inMpdc)(0))
        return false;

    unsigned int count = inMpdc->GetParticleListCount();

    outMpdc->SetParticleListCount(count);
    outMpdc->SetDataHash(inMpdc->DataHash());
    outMpdc->SetFrameCount(inMpdc->FrameCount());
    outMpdc->SetFrameID(inMpdc->FrameID());
    outMpdc->SetUnlocker(new VAOUnlocker(), false);

    if (count == 0) {
        return false;
    }

    glBindVertexArray(0);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER, 0);

    // currently only 1 particle list is supported
    ASSERT(count <= 1);

    // varying listcounts are bad
    if (count != glVAO.Count()) {
        glDeleteBuffersARB(static_cast<GLsizei>(glVB.Count()), glVB.PeekElements());
        for (unsigned int i = 0; i < count; ++i) {
            GLuint buffer;
            glGenBuffersARB(1, &buffer);
            glVB.Append(buffer);
        }

        glDeleteBuffersARB(static_cast<GLsizei>(glCB.Count()), glCB.PeekElements());
        for (unsigned int i = 0; i < count; ++i) {
            GLuint buffer;
            glGenBuffersARB(1, &buffer);
            glCB.Append(buffer);
        }

        glDeleteVertexArrays(static_cast<GLsizei>(glVAO.Count()), glVAO.PeekElements());
        for (unsigned int i = 0; i < count; ++i) {
            GLuint buffer;
            glGenVertexArrays(1, &buffer);
            glVAO.Append(buffer);
        }

        if (glClusterInfos) {
            glDeleteBuffersARB(1, &glClusterInfos);
            glClusterInfos = 0;
        }
    }

    if (!glClusterInfos) {
        glGenBuffersARB(1, &glClusterInfos);
    }

    unsigned int particleCount = 0;
    unsigned int particleStride = 0;
    float particleRadius = 0.0f;
    using geocalls::SimpleSphericalParticles;
    SimpleSphericalParticles::ClusterInfos* clusterInfos;
    count = (inMpdc->GetParticleListCount() < 1) ? inMpdc->GetParticleListCount() : 1;
    for (unsigned int i = 0; i < count; ++i) {
        MultiParticleDataCall::Particles& partsIn = inMpdc->AccessParticles(i);
        MultiParticleDataCall::Particles& partsOut = outMpdc->AccessParticles(i);

        // colour
        unsigned int colorBytes = 0;
        switch (partsIn.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            colorBytes = 3 * sizeof(unsigned char);
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
            colorBytes = 4 * sizeof(unsigned char);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
            colorBytes = 3 * sizeof(float);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
            colorBytes = 4 * sizeof(float);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
            // unsupported
            break;
        default:
            break;
        }
        //
        unsigned int elementBytes = 0;
        switch (partsIn.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            continue;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            elementBytes = 3 * sizeof(float);
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            elementBytes = 4 * sizeof(float);
            break;
        default:
            continue;
        }

        partsOut.disableNullChecksForVAOs();
        partsOut.SetIsVAO(true);
        partsOut.SetCount(partsIn.GetCount());
        partsOut.SetGlobalRadius(partsIn.GetGlobalRadius());
        const unsigned char* color = partsIn.GetGlobalColour();
        partsOut.SetGlobalColour(color[0], color[1], color[2], color[3]);
        partsOut.SetGlobalType(partsIn.GetGlobalType());

        GLuint& vao(glVAO[i]);
        GLuint& vb(glVB[i]);
        GLuint& cb(glCB[i]);
        clusterInfos = partsIn.GetClusterInfos();

        // support the rest yourself...
        if (partsIn.GetVertexDataStride() == 3 * sizeof(float) && partsIn.GetColourDataStride() == 4 * sizeof(float)) {
            // highly specific, because we know:
            partsOut.SetVertexData(partsIn.GetVertexDataType(), NULL, partsIn.GetVertexDataStride());
            partsOut.SetColourData(
                MultiParticleDataCall::Particles::COLDATA_FLOAT_I, NULL, partsIn.GetColourDataStride());

            partsOut.SetVAOs(glVAO[i], glVB[i], glCB[i]);
            glBindVertexArray(vao);
            glBindBufferARB(GL_ARRAY_BUFFER, vb);
            glBufferDataARB(GL_ARRAY_BUFFER,
                static_cast<GLsizeiptrARB>(partsOut.GetVertexDataStride() * partsOut.GetCount()),
                partsIn.GetVertexData(), GL_DYNAMIC_DRAW);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GL_FLOAT, partsOut.GetVertexDataStride(), partsOut.GetVertexData());
            glBindBufferARB(GL_ARRAY_BUFFER, cb);
            glBufferDataARB(GL_ARRAY_BUFFER,
                static_cast<GLsizeiptrARB>(partsOut.GetColourDataStride() * partsOut.GetCount()),
                partsIn.GetColourData(), GL_DYNAMIC_DRAW);
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(4, GL_FLOAT, partsOut.GetColourDataStride(), partsOut.GetColourData());
            glBindVertexArray(0);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);

            glBindBufferARB(GL_ARRAY_BUFFER, glClusterInfos);
            glBufferDataARB(GL_ARRAY_BUFFER, clusterInfos->sizeofPlainData, clusterInfos->plainData, GL_DYNAMIC_DRAW);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            shaderOnClusterComputation->use();
            glUniform1ui(this->shaderOnClusterComputation->getUniformLocation("count"), clusterInfos->numClusters);
            glUniform1ui(this->shaderOnClusterComputation->getUniformLocation("pos_stride"),
                partsOut.GetVertexDataStride() / sizeof(float));
            glUniform1ui(this->shaderOnClusterComputation->getUniformLocation("col_stride"),
                partsOut.GetColourDataStride() / sizeof(float));
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, glClusterInfos);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, glVB[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, glCB[0]);
            glDispatchCompute((clusterInfos->numClusters / 1024) + 1, 1, 1);
            glUseProgram(0);

            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        } else if (partsIn.GetVertexDataStride() == 3 * sizeof(float) + 4 * sizeof(float)) {
            // highly specific, because we know:
            partsOut.SetVertexData(partsIn.GetVertexDataType(), NULL, partsIn.GetVertexDataStride());
            partsOut.SetColourData(MultiParticleDataCall::Particles::COLDATA_FLOAT_I, (void*)(NULL + 3 * sizeof(float)),
                partsIn.GetVertexDataStride());

            partsOut.SetVAOs(glVAO[i], glVB[i], glVB[i]);
            glBindVertexArray(vao);
            glBindBufferARB(GL_ARRAY_BUFFER, vb);
            glBufferDataARB(GL_ARRAY_BUFFER,
                static_cast<GLsizeiptrARB>(partsOut.GetVertexDataStride() * partsOut.GetCount()),
                partsIn.GetVertexData(), GL_DYNAMIC_DRAW);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GL_FLOAT, partsOut.GetVertexDataStride(), partsOut.GetVertexData());
            // glEnableClientState(GL_COLOR_ARRAY);
            // glColorPointer(4, GL_FLOAT, partsOut.GetColourDataStride(), partsOut.GetColourData());
            glBindVertexArray(0);
            glBindBufferARB(GL_ARRAY_BUFFER, 0);

            glBindBufferARB(GL_ARRAY_BUFFER, glClusterInfos);
            glBufferDataARB(GL_ARRAY_BUFFER, clusterInfos->sizeofPlainData, clusterInfos->plainData, GL_DYNAMIC_DRAW);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            shaderOnClusterComputation->use();
            glUniform1ui(this->shaderOnClusterComputation->getUniformLocation("count"), clusterInfos->numClusters);
            glUniform1ui(this->shaderOnClusterComputation->getUniformLocation("pos_stride"),
                partsOut.GetVertexDataStride() / sizeof(float));
            glUniform1ui(this->shaderOnClusterComputation->getUniformLocation("col_stride"),
                partsOut.GetVertexDataStride() / sizeof(float));
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, glClusterInfos);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, glVB[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, glVB[0]);
            glDispatchCompute((clusterInfos->numClusters / 1024) + 1, 1, 1);
            glUseProgram(0);

            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        }

        particleCount += static_cast<unsigned int>(partsOut.GetCount());
        particleStride = partsOut.GetVertexDataStride();
        particleRadius = partsOut.GetGlobalRadius();
    }


    /*

    */
    /*
    if(!glParticleList)
    {
        glGenBuffersARB(1, &glParticleList);
        glBindBufferARB (GL_ARRAY_BUFFER, glParticleList);
        glBufferDataARB (GL_ARRAY_BUFFER, particleCount * sizeof(unsigned int) * 2, NULL, GL_STREAM_DRAW);
    }

    const unsigned int prefixTestCount = 6;
    float prefixIn[prefixTestCount];
    prefixIn[0] = 3.0f;
    prefixIn[1] = 1.0f;
    prefixIn[2] = 4.0f;
    prefixIn[3] = 2.0f;
    prefixIn[4] = 1.0f;
    prefixIn[5] = 1.0f;
    float prefixOut[prefixTestCount];
    prefixOut[0] = -1.0f;
    prefixOut[1] = -1.0f;
    prefixOut[2] = -1.0f;
    prefixOut[3] = -1.0f;
    prefixOut[4] = -1.0f;
    prefixOut[5] = -1.0f;

    if(!glPrefixIn)
    {
        glGenBuffersARB(1, &glPrefixIn);
        glBindBufferARB (GL_ARRAY_BUFFER, glPrefixIn);
        glGenBuffersARB(1, &glPrefixOut);
        glBindBufferARB (GL_ARRAY_BUFFER, glPrefixOut);
    }

    glBindBufferARB (GL_ARRAY_BUFFER, glPrefixIn);
    glBufferDataARB (GL_ARRAY_BUFFER, prefixTestCount * sizeof(float), prefixIn, GL_DYNAMIC_DRAW);
    glBindBufferARB (GL_ARRAY_BUFFER, glPrefixOut);
    glBufferDataARB (GL_ARRAY_BUFFER, prefixTestCount * sizeof(float), prefixOut, GL_DYNAMIC_DRAW);
    glBindBufferARB (GL_ARRAY_BUFFER, 0);

    shaderComputePrefixSum.Enable();
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("count"), 4);
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("exclusive"), 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, glPrefixIn);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, glPrefixOut);
        shaderComputePrefixSum.Dispatch((((4/2))/1024)+1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    shaderComputePrefixSum.Disable();

    glBindBuffer(GL_ARRAY_BUFFER, glPrefixOut);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, prefixTestCount * sizeof(float), &prefixOut);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    shaderComputePrefixSum.Enable();
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("count"), 2);
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("exclusive"), 0);
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("memoryOffset"), 4);
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("memoryOffsetStartAdd"), 3);
        glUniform1ui(this->shaderComputePrefixSum.ParameterLocation("addMemoryOffsetStartAdd"), 1);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, glPrefixIn);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, glPrefixOut);
        shaderComputePrefixSum.Dispatch((((2/2))/1024)+1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    shaderComputePrefixSum.Disable();

    glBindBuffer(GL_ARRAY_BUFFER, glPrefixOut);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, prefixTestCount * sizeof(float), &prefixOut);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /*
    shaderComputeInitParticleList.Enable();
        glUniform1ui(this->shaderComputeInitParticleList.ParameterLocation("particleListStride"), 2 * sizeof(unsigned
    int)); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, glParticleList);
        shaderComputeInitParticleList.Dispatch((particleCount/1024)+1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    shaderComputeInitParticleList.Disable();

    shaderComputeGrid.Enable();
        glUniform1ui(this->shaderComputeGrid.ParameterLocation("count"), particleCount);
        glUniform1ui(this->shaderComputeGrid.ParameterLocation("stride"), particleStride);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, glVB[0]);
        shaderComputeGrid.Dispatch((particleCount/1024)+1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    shaderComputeGrid.Disable();

    shaderComputeGriddify.Enable();
        glUniform1ui(this->shaderComputeGriddify.ParameterLocation("count"), particleCount);
        glUniform1ui(this->shaderComputeGriddify.ParameterLocation("stride"), particleStride);
        glUniform1f(this->shaderComputeGriddify.ParameterLocation("radius"), particleRadius);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, glVB[0]);
        shaderComputeGriddify.Dispatch((particleCount/1024)+1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    shaderComputeGriddify.Disable();



    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    */

    // glBindBufferARB (GL_ARRAY_BUFFER, glVB[0]);

    inMpdc->Unlock();

    return true;
    /*
    DirectionalParticleDataCall *outDpdc = dynamic_cast<DirectionalParticleDataCall*>(&call);
    if ((outMpdc == NULL) && (outDpdc == NULL)) return false;

    bool doStuff = false;
    if (this->globalColorMapComputationSlot.IsDirty()) {
        this->globalColorMapComputationSlot.ResetDirty();
        doStuff = true;
    }
    if (this->includeHiddenInColorMapSlot.IsDirty()) {
        this->includeHiddenInColorMapSlot.ResetDirty();
        doStuff = true;
    }

    if (outMpdc != NULL) {
        MultiParticleDataCall *inMpdc = this->inParticlesDataSlot.CallAs<MultiParticleDataCall>();
        if (inMpdc == NULL) return false;

        if (this->includeAllSlot.IsDirty()) {
            this->includeAllSlot.ResetDirty();
            vislib::StringA str, str2;
            unsigned int cnt = inMpdc->GetParticleListCount();
            for (unsigned int i = 0; i < cnt; i++) {
                str2.Format("%u%s", inMpdc->AccessParticles(i).GetGlobalType(), (i == cnt - 1) ? "" : ", ");
                str.Append(str2);
            }
            this->includedListsSlot.Param<megamol::core::param::StringParam>()->SetValue(str);
        }
        vislib::Array<unsigned int> included = this->getSelectedLists();
        if (this->includedListsSlot.IsDirty()) {
            this->includedListsSlot.ResetDirty();
            doStuff = true;
        }

        // make a deep copy (also of content pointers)
        //*inDpdc = *outDpdc;
        // call DataCallback, updating content
        if (!(*inMpdc)(0)) {
            return false;
        }
        // copy diffs back (maybe)
        //*outDpdc = *inDpdc;

        if (inMpdc->DataHash() != this->datahashParticlesIn) {
            doStuff = true;
            this->datahashParticlesIn = inMpdc->DataHash();
        }
        // not sure
        if (outMpdc->FrameID() != this->frameID) {
            doStuff = true;
        }

        if (!doStuff) {
            return true;
        }

        unsigned int cnt = inMpdc->GetParticleListCount();
        unsigned int outCnt = 0;
        if (included.Count() == 0) {
            outCnt = cnt;
        } else {
            for (unsigned int i = 0; i < cnt; i++) {
                if (included.Contains(inMpdc->AccessParticles(i).GetGlobalType())) {
                    outCnt++;
                }
            }
        }
        outMpdc->SetParticleListCount(outCnt);
        outCnt = 0;
        float globalMin = FLT_MAX;
        float globalMax = -FLT_MAX;
        for (unsigned int i = 0; i < cnt; i++) {
            if (this->includeHiddenInColorMapSlot.Param<megamol::core::param::BoolParam>()->Value()
                || included.Count() == 0
                || (included.Count() > 0 && included.Contains(inMpdc->AccessParticles(i).GetGlobalType()))) {
                if (inMpdc->AccessParticles(i).GetMinColourIndexValue() < globalMin) globalMin =
    inMpdc->AccessParticles(i).GetMinColourIndexValue(); if (inMpdc->AccessParticles(i).GetMaxColourIndexValue() >
    globalMax) globalMax = inMpdc->AccessParticles(i).GetMaxColourIndexValue();
            }
            if (included.Count() > 0 && !included.Contains(inMpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            outMpdc->AccessParticles(outCnt).SetCount(inMpdc->AccessParticles(i).GetCount());
            outMpdc->AccessParticles(outCnt).SetColourData(inMpdc->AccessParticles(i).GetColourDataType(),
                inMpdc->AccessParticles(i).GetColourData(), inMpdc->AccessParticles(i).GetColourDataStride());
            outMpdc->AccessParticles(outCnt).SetVertexData(inMpdc->AccessParticles(i).GetVertexDataType(),
                inMpdc->AccessParticles(i).GetVertexData(), inMpdc->AccessParticles(i).GetVertexDataStride());
            // TODO BUG HAZARD this is most probably wrong, as different list subsets have a different dynamic range :(
            // probably still loop over all...
            //outMpdc->AccessParticles(outCnt).SetColourMapIndexValues(inMpdc->AccessParticles(i).GetMinColourIndexValue(),
            //    inMpdc->AccessParticles(i).GetMaxColourIndexValue());

            const unsigned char *col = inMpdc->AccessParticles(i).GetGlobalColour();
            outMpdc->AccessParticles(outCnt).SetGlobalColour(col[0], col[1], col[2], col[3]);
            outMpdc->AccessParticles(outCnt).SetGlobalRadius(inMpdc->AccessParticles(i).GetGlobalRadius());
            outMpdc->AccessParticles(outCnt).SetGlobalType(inMpdc->AccessParticles(i).GetGlobalType());
            outCnt++;
        }
        outCnt = 0;
        for (unsigned int i = 0; i < cnt; i++) {
            if (included.Count() > 0 && !included.Contains(inMpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            if (this->globalColorMapComputationSlot.Param<megamol::core::param::BoolParam>()->Value()) {
                outMpdc->AccessParticles(outCnt).SetColourMapIndexValues(globalMin, globalMax);
            } else {
                outMpdc->AccessParticles(outCnt).SetColourMapIndexValues(inMpdc->AccessParticles(i).GetMinColourIndexValue(),
                    inMpdc->AccessParticles(i).GetMaxColourIndexValue());
            }
            outCnt++;
        }
        this->datahashParticlesOut++;
        outMpdc->SetDataHash(this->datahashParticlesOut);

    } else if (outDpdc != NULL) {
        DirectionalParticleDataCall *inDpdc = this->inParticlesDataSlot.CallAs<DirectionalParticleDataCall>();
        if (inDpdc == NULL) return false;

        if (this->includeAllSlot.IsDirty()) {
            this->includeAllSlot.ResetDirty();
            vislib::StringA str, str2;
            unsigned int cnt = inDpdc->GetParticleListCount();
            for (unsigned int i = 0; i < cnt; i++) {
                str2.Format("%u%s", inDpdc->AccessParticles(i).GetGlobalType(), (i == cnt - 1) ? "" : ", ");
                str.Append(str2);
            }
            this->includedListsSlot.Param<megamol::core::param::StringParam>()->SetValue(str);
        }
        vislib::Array<unsigned int> included = this->getSelectedLists();
        if (this->includedListsSlot.IsDirty()) {
            this->includedListsSlot.ResetDirty();
            doStuff = true;
        }

        // make a deep copy (also of content pointers)
        //*inDpdc = *outDpdc;
        // call DataCallback, updating content
        if (!(*inDpdc)(0)) {
            return false;
        }
        // copy diffs back (maybe)
        //*outDpdc = *inDpdc;

        if (inDpdc->DataHash() != this->datahashParticlesIn) {
            doStuff = true;
            this->datahashParticlesIn = inDpdc->DataHash();
        }
        // not sure
        if (outDpdc->FrameID() != this->frameID) {
            doStuff = true;
        }

        if (!doStuff) {
            return true;
        }

        unsigned int cnt = inDpdc->GetParticleListCount();
        unsigned int outCnt = 0;
        if (included.Count() == 0) {
            outCnt = cnt;
        } else {
            for (unsigned int i = 0; i < cnt; i++) {
                if (included.Contains(inDpdc->AccessParticles(i).GetGlobalType())) {
                    outCnt++;
                }
            }
        }
        outDpdc->SetParticleListCount(outCnt);
        outCnt = 0;
        float globalMin = FLT_MAX;
        float globalMax = -FLT_MAX;
        for (unsigned int i = 0; i < cnt; i++) {
            if (this->includeHiddenInColorMapSlot.Param<megamol::core::param::BoolParam>()->Value()
                || included.Count() == 0
                || (included.Count() > 0 && included.Contains(inDpdc->AccessParticles(i).GetGlobalType()))) {
                if (inDpdc->AccessParticles(i).GetMinColourIndexValue() < globalMin) globalMin =
    inDpdc->AccessParticles(i).GetMinColourIndexValue(); if (inDpdc->AccessParticles(i).GetMaxColourIndexValue() >
    globalMax) globalMax = inDpdc->AccessParticles(i).GetMaxColourIndexValue();
            }
            if (included.Count() > 0 && !included.Contains(inDpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            outDpdc->AccessParticles(outCnt).SetCount(inDpdc->AccessParticles(i).GetCount());
            outDpdc->AccessParticles(outCnt).SetColourData(inDpdc->AccessParticles(i).GetColourDataType(),
                inDpdc->AccessParticles(i).GetColourData(), inDpdc->AccessParticles(i).GetColourDataStride());
            outDpdc->AccessParticles(outCnt).SetVertexData(inDpdc->AccessParticles(i).GetVertexDataType(),
                inDpdc->AccessParticles(i).GetVertexData(), inDpdc->AccessParticles(i).GetVertexDataStride());
            // TODO BUG HAZARD this is most probably wrong, as different list subsets have a different dynamic range :(
            // probably still loop over all...
            //outDpdc->AccessParticles(outCnt).SetColourMapIndexValues(inDpdc->AccessParticles(i).GetMinColourIndexValue(),
            //    inDpdc->AccessParticles(i).GetMaxColourIndexValue());
            const unsigned char *col = inDpdc->AccessParticles(i).GetGlobalColour();
            outDpdc->AccessParticles(outCnt).SetGlobalColour(col[0], col[1], col[2], col[3]);
            outDpdc->AccessParticles(outCnt).SetGlobalRadius(inDpdc->AccessParticles(i).GetGlobalRadius());
            outDpdc->AccessParticles(outCnt).SetGlobalType(inDpdc->AccessParticles(i).GetGlobalType());
            outDpdc->AccessParticles(outCnt).SetDirData(inDpdc->AccessParticles(i).GetDirDataType(),
                inDpdc->AccessParticles(i).GetDirData(), inDpdc->AccessParticles(i).GetDirDataStride());
            outCnt++;
        }
        outCnt = 0;
        for (unsigned int i = 0; i < cnt; i++) {
            if (included.Count() > 0 && !included.Contains(inDpdc->AccessParticles(i).GetGlobalType())) {
                continue;
            }
            if (this->globalColorMapComputationSlot.Param<megamol::core::param::BoolParam>()->Value()) {
                outDpdc->AccessParticles(outCnt).SetColourMapIndexValues(globalMin, globalMax);
            } else {
                outDpdc->AccessParticles(outCnt).SetColourMapIndexValues(inDpdc->AccessParticles(i).GetMinColourIndexValue(),
                    inDpdc->AccessParticles(i).GetMaxColourIndexValue());
            }
            outCnt++;
        }
        this->datahashParticlesOut++;
        outDpdc->SetDataHash(this->datahashParticlesOut);
    }
    return true;
    */
}


/*
 * io::DirPartColModulate::getExtent
 */
bool ParticleWorker::getExtentCallback(Call& call) {
    using megamol::geocalls::MultiParticleDataCall;
    MultiParticleDataCall* outMpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (outMpdc == NULL)
        return false;

    if (outMpdc != NULL) {
        MultiParticleDataCall* inMpdc = this->inParticlesDataSlot.CallAs<MultiParticleDataCall>();
        if (inMpdc == NULL)
            return false;
        // this is the devil. don't do it.
        //*inMpdc = *outMpdc;
        // this is better but still not good.
        static_cast<AbstractGetData3DCall*>(inMpdc)->operator=(*outMpdc);
        if ((*inMpdc)(1)) {
            //*outMpdc = *inMpdc;
            static_cast<AbstractGetData3DCall*>(outMpdc)->operator=(*inMpdc);
            outMpdc->SetDataHash(inMpdc->DataHash());
            return true;
        }
    }
    return false;
}
