//
// CUDAStreamlines.cu
//
//  Created on: Nov 6, 2013
//      Author: scharnkn
//

#include "CUDAStreamlines.h"
#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "ogl_error_check.h"
#include "cuda_error_check.h"
#include "CUDAGrid.cuh"
#include "HostArr.h"
#include "vislib/sys/Log.h"

#include <cuda_runtime.h>
#define WGL_NV_gpu_affinity
#include <cuda_gl_interop.h>
#include "helper_math.h"

using namespace megamol;
using namespace megamol::protein_cuda;

// Offset in output VBO for positions
const int CUDAStreamlines::vboOffsPos = 0;

// Offset in output VBO for color
const int CUDAStreamlines::vboOffsCol = 3;

// VBO vertex data stride
const int CUDAStreamlines::vboStride = 7;

// TODO
// + Implement backward and/or bi-directional integration of streamlines

/**
 * @return Returns the thread index based on the current CUDA grid dimensions
 */
inline __device__ uint getThreadIdx() {
    return __umul24(__umul24(blockIdx.y, gridDim.x) + blockIdx.x, blockDim.x) +
            threadIdx.x;
}


/**
 * Performs one iteration step using RK4 integration.
 *
 * @param lineStrip_D The line strip buffer
 * qparam vecField_D The vector field
 * @param nStreamlines The number of streamlines to be integrated
 * @param nSegments The number of segments in one line
 * @param step The step size
 * @param the offset for the line strip buffer to get current position
 */
__global__ void CUDAStreamlines_IntegrateRK4Step(
        float *lineStrip_D,
        float *vecField_D,
        int nStreamlines,
        int nSegments,
        float step,
        int offset,
        int vboPosOffs,
        int vboStride,
        float dir) {

    const uint idx = ::getThreadIdx();
    if (idx >= nStreamlines) return;

    const uint lineBuffSize = (nSegments+1)*vboStride;

    float3 x0, x1, x2, x3;
    float3 v0, v1, v2, v3;

    // Get current position
    x0 = make_float3(
            lineStrip_D[idx*lineBuffSize+offset*vboStride+vboPosOffs+0],
            lineStrip_D[idx*lineBuffSize+offset*vboStride+vboPosOffs+1],
            lineStrip_D[idx*lineBuffSize+offset*vboStride+vboPosOffs+2]);

    if (!::IsValidGridpos(x0)) {
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+0] = x0.x;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+1] = x0.y;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+2] = x0.z;
        return; // This should never happen!
    }

    // Find new position using fourth order Runge-Kutta method
    // Sample vector field at current position
    v0  = ::SampleFieldAtPosTrilin_D<float3, false>(x0, (float3*)vecField_D)*dir;
    v0 = normalize(v0);
    v0 *= step;


    x1 = x0 + 0.5*v0;
    if (::IsValidGridpos(x1)) {
        v1 = ::SampleFieldAtPosTrilin_D<float3, false>(x1, (float3*)vecField_D)*dir;
        v1 = normalize(v1);
        v1 *= step;
    } else {
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+0] = x0.x;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+1] = x0.y;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+2] = x0.z;
        return;
    }

    x2 = x0 + 0.5*v1;
    if (::IsValidGridpos(x2)) {
        v2 = ::SampleFieldAtPosTrilin_D<float3, false>(x2, (float3*)vecField_D)*dir;
        v2 = normalize(v2);
        v2 *= step;
    } else {
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+0] = x0.x;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+1] = x0.y;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+2] = x0.z;
        return;
    }

    x3 = x0 + v2;
    if (::IsValidGridpos(x3)) {
        v3 = ::SampleFieldAtPosTrilin_D<float3, false>(x3, (float3*)vecField_D)*dir;
        v3 = normalize(v3);
        v3 *= step;
    } else {
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+0] = x0.x;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+1] = x0.y;
        lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+2] = x0.z;
        return;
    }

    x0 += (1.0/6.0)*(v0+2.0*v1+2.0*v2+v3);

    // Write new position to global device memory
    lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+0] = x0.x;
    lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+1] = x0.y;
    lineStrip_D[idx*lineBuffSize+(offset+int(dir))*vboStride+vboPosOffs+2] = x0.z;

}


/*
 * CUDAStreamlines_SampleScalarFieldToAlpha_D
 */
__global__ void CUDAStreamlines_SampleScalarFieldToAlpha_D(
        float *lineStrip_D,
        float *field_D,
        int nStreamlines,
        int nSegments,
        int vboPosOffs,
        int vboColOffs,
        int vboStride) {

    const uint idx = ::getThreadIdx();
    const int vertexCnt = nStreamlines*(nSegments+1);
    if (idx >= vertexCnt) return;

    float3 pos = make_float3(
            lineStrip_D[idx*vboStride+vboPosOffs+0],
            lineStrip_D[idx*vboStride+vboPosOffs+1],
            lineStrip_D[idx*vboStride+vboPosOffs+2]);

    float sample = ::SampleFieldAtPosTrilin_D<float, false>(pos, field_D);
    lineStrip_D[idx*vboStride+vboColOffs+3] = sample;
}


/*
 * CUDAStreamlines_SampleVecFieldToRGB_D
 */
__global__ void CUDAStreamlines_SampleVecFieldToRGB_D(
        float *lineStrip_D,
        float *field_D,
        int nStreamlines,
        int nSegments,
        int vboPosOffs,
        int vboColOffs,
        int vboStride) {

    const uint idx = ::getThreadIdx();
    const int vertexCnt = nStreamlines*(nSegments+1);
    if (idx >= vertexCnt) return;

    float3 pos = make_float3(
            lineStrip_D[idx*vboStride+vboPosOffs+0],
            lineStrip_D[idx*vboStride+vboPosOffs+1],
            lineStrip_D[idx*vboStride+vboPosOffs+2]);

    float3 sample = ::SampleFieldAtPosTrilin_D<float3, false>(pos, (float3*)field_D);
    lineStrip_D[idx*vboStride+vboColOffs+0] = sample.x;
    lineStrip_D[idx*vboStride+vboColOffs+1] = sample.y;
    lineStrip_D[idx*vboStride+vboColOffs+2] = sample.z;

//    lineStrip_D[idx*vboStride+vboColOffs+0] = 1.0;
//    lineStrip_D[idx*vboStride+vboColOffs+1] = 0.0;
//    lineStrip_D[idx*vboStride+vboColOffs+2] = 0.0;
}


/*
 * CUDAStreamlines_SetUniformRGBColor_D
 */
__global__ void CUDAStreamlines_SetUniformRGBColor_D(
        float *lineStrip_D,
        int nStreamlines,
        int nSegments,
        int vboColOffs,
        int vboStride,
        float3 col) {

    const uint idx = ::getThreadIdx();
    const int vertexCnt = nStreamlines*(nSegments+1);
    if (idx >= vertexCnt) return;

    lineStrip_D[idx*vboStride+vboColOffs+0] = col.x;
    lineStrip_D[idx*vboStride+vboColOffs+1] = col.y;
    lineStrip_D[idx*vboStride+vboColOffs+2] = col.z;
}


/*
 * CUDAStreamlines::CUDAStreamlines
 */
CUDAStreamlines::CUDAStreamlines() : cudaToken(NULL), lineStripVBO(0) {
    // EMPTY
};


/*
 * CUDAStreamlines:~CUDAStreamlines
 */
CUDAStreamlines::~CUDAStreamlines() {
    if (this->lineStripVBO != 0) {
        this->destroyVBO();
    }
    this->vecField_D.Release();
}


/*
 * CUDAStreamlines::InitStreamlines
 */
bool CUDAStreamlines::InitStreamlines(int nSegments, int nStreamlines, Direction dir) {
    if ((dir == CUDAStreamlines::FORWARD)||(dir == CUDAStreamlines::BACKWARD)) {
        this->nSegments = nSegments;
    } else if (dir == CUDAStreamlines::BIDIRECTIONAL) {
        this->nSegments = nSegments*2;
    }

    this->nStreamlines = nStreamlines;
    this->dir = dir;
    if (!this->initVBO()) {
        return false;
    }
    return ::CheckForGLError();
}


/*
 * CUDAStreamlines::IntegrateRK4
 */
bool CUDAStreamlines::IntegrateRK4(const float *seedPoints, float step,
        float *vecField, int3 vecFieldDim, float3 vecFieldOrg, float3 vecFieldDelta) {

    // Get pointer to line strip
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *lineStrip_D;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&lineStrip_D), // The mapped pointer
            &vboSize,                  // The size of the accessible data
            this->cudaToken))) {                 // The mapped resource
        return false;
    }

    // Init constant grid parameters of the vector field
    if (!initGridParams(vecFieldDim, vecFieldOrg, vecFieldDelta)) {
        return false;
    }

    size_t latticeSize = vecFieldDim.x*vecFieldDim.y*vecFieldDim.z;

    // Copy vector field to device memory
    if (!CudaSafeCall(this->vecField_D.Validate(latticeSize*3))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->vecField_D.Peek(), vecField,
            sizeof(float)*3*latticeSize, cudaMemcpyHostToDevice))) {
        return false;
    }

    ::CheckForCudaErrorSync();

//    for (int i = 0; i < this->nStreamlines; ++i) {
//        printf("seed point #%i %f %f %f\n", i,
//                seedPoints[3*i+0],
//                seedPoints[3*i+1],
//                seedPoints[3*i+2]);
//    }

    // Init streamlines with starting position
    if ((this->dir == CUDAStreamlines::BACKWARD)||(this->dir == CUDAStreamlines::FORWARD)) {
        size_t lineBuffSize = (this->nSegments+1)*CUDAStreamlines::vboStride;
        for (int cnt = 0; cnt < this->nStreamlines; ++cnt) {
            if (!CudaSafeCall(cudaMemcpy(
                    lineStrip_D+lineBuffSize*cnt,
                    //seedPoints+cnt*sizeof(float)*3,
                    seedPoints+cnt*3,
                    sizeof(float)*3,
                    cudaMemcpyHostToDevice))) {
                return false;
            }
        }
    } else if (this->dir == CUDAStreamlines::BIDIRECTIONAL) {
        size_t lineBuffSize = (this->nSegments+1)*CUDAStreamlines::vboStride;
        for (int cnt = 0; cnt < this->nStreamlines; ++cnt) {
            if (!CudaSafeCall(cudaMemcpy(
                    CUDAStreamlines::vboStride*(this->nSegments/2) + lineStrip_D+lineBuffSize*cnt,
                    //seedPoints+cnt*sizeof(float)*3,
                    seedPoints+cnt*3,
                    sizeof(float)*3,
                    cudaMemcpyHostToDevice))) {
                return false;
            }
        }
    }

    ::CheckForCudaErrorSync();

    if (this->dir == CUDAStreamlines::FORWARD) {

        // RK4 integration
        for (int it = 0; it < this->nSegments; ++it) {
            // Call cuda kernel for one integration step
            CUDAStreamlines_IntegrateRK4Step <<< Grid(this->nStreamlines, 256), 256 >>> (
                    lineStrip_D,
                    this->vecField_D.Peek(),
                    this->nStreamlines,
                    this->nSegments,
                    step,
                    it, // Offset for line strip buffer to get current position
                    CUDAStreamlines::vboOffsPos,
                    CUDAStreamlines::vboStride,
                    1.0 // Direction
            );
            if (!::CheckForCudaErrorSync()) {
                return false;
            }
        }
    } else if (this->dir == CUDAStreamlines::BACKWARD) {
        // RK4 integration
        for (int it = 0; it < this->nSegments; ++it) {
            // Call cuda kernel for one integration step
            CUDAStreamlines_IntegrateRK4Step <<< Grid(this->nStreamlines, 256), 256 >>> (
                    lineStrip_D,
                    this->vecField_D.Peek(),
                    this->nStreamlines,
                    this->nSegments,
                    step,
                    it, // Offset for line strip buffer to get current position
                    CUDAStreamlines::vboOffsPos,
                    CUDAStreamlines::vboStride,
                    -1.0 // Direction
            );
            if (!::CheckForCudaErrorSync()) {
                return false;
            }
        }
    } else {
        // RK4 forward integration
        for (int it = this->nSegments/2; it < this->nSegments; ++it) {
        //for (int it = this->nSegments; it < 1; ++it) {
            // Call cuda kernel for one integration step
            CUDAStreamlines_IntegrateRK4Step <<< Grid(this->nStreamlines, 256), 256 >>> (
                    lineStrip_D,
                    this->vecField_D.Peek(),
                    this->nStreamlines,
                    this->nSegments,
                    step,
                    it, // Offset for line strip buffer to get current position
                    CUDAStreamlines::vboOffsPos,
                    CUDAStreamlines::vboStride,
                    1.0 // Direction
            );
            if (!::CheckForCudaErrorSync()) {
                return false;
            }
        }

        // RK4 backward integration
        for (int it = this->nSegments/2; it > 0; --it) {
            // Call cuda kernel for one integration step
            CUDAStreamlines_IntegrateRK4Step <<< Grid(this->nStreamlines, 256), 256 >>> (
                    lineStrip_D,
                    this->vecField_D.Peek(),
                    this->nStreamlines,
                    this->nSegments,
                    step,
                    it, // Offset for line strip buffer to get current position
                    CUDAStreamlines::vboOffsPos,
                    CUDAStreamlines::vboStride,
                    -1.0 // Direction
            );
            if (!::CheckForCudaErrorSync()) {
                return false;
            }
        }

    }

//    HostArr<float> lineStrip;
//    lineStrip.Validate(this->nStreamlines*(this->nSegments+1)*CUDAStreamlines::vboStride);
//    if (!CudaSafeCall(cudaMemcpy(lineStrip.Peek(), lineStrip_D,
//            lineStrip.GetCount()*sizeof(float), cudaMemcpyDeviceToHost))) {
//        return false;
//    }
//    for (int s = 0; s < this->nStreamlines; ++s) {
//        printf("streamline %i\n", s);
//        for (int i = 0; i <= this->nSegments; ++i) {
//            printf("%i: pos: %f %f %f\n", i,
//                    lineStrip.Peek()[s*((this->nSegments+1)*CUDAStreamlines::vboStride)+CUDAStreamlines::vboStride*i+0],
//                    lineStrip.Peek()[s*((this->nSegments+1)*CUDAStreamlines::vboStride)+CUDAStreamlines::vboStride*i+1],
//                    lineStrip.Peek()[s*((this->nSegments+1)*CUDAStreamlines::vboStride)+CUDAStreamlines::vboStride*i+2]);
//        }
//        printf("---------------------------\n");
//    }

    // Unmap the device pointer
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->cudaToken))) {
        return false;
    }

    return CheckForCudaErrorSync();
}


/*
 * CUDAStreamlines::RenderLineStrip
 */
bool CUDAStreamlines::RenderLineStrip() {

    glBindBufferARB(GL_ARRAY_BUFFER, this->lineStripVBO);

    glEnableClientState(GL_VERTEX_ARRAY);
    ::CheckForGLError();

    // Draw stream lines using the line strip buffer
    for (int cnt = 0; cnt < this->nStreamlines; ++cnt) {
        int offs = cnt*CUDAStreamlines::vboStride*(this->nSegments+1)*sizeof(float);
        glVertexPointer(3, GL_FLOAT,
                CUDAStreamlines::vboStride*sizeof(float),
                (const GLvoid*)((long int)(offs))); // last param is offset, not ptr

        glDrawArrays(GL_LINE_STRIP_ADJACENCY, 0, this->nSegments+1);
        ::CheckForGLError(); // OpenGL error check
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    return ::CheckForGLError();
}


/*
 * CUDAStreamlines::RenderLineStripWithColor
 */
bool CUDAStreamlines::RenderLineStripWithColor() {

    glBindBufferARB(GL_ARRAY_BUFFER, this->lineStripVBO);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    ::CheckForGLError();

    // Draw stream lines using the line strip buffer
    for (int cnt = 0; cnt < this->nStreamlines; ++cnt) {
        int offs = cnt*CUDAStreamlines::vboStride*(this->nSegments+1)*sizeof(float);
        glVertexPointer(3, GL_FLOAT,
                CUDAStreamlines::vboStride*sizeof(float),
                (const GLvoid*)((long int)(offs))); // last param is offset, not ptr

        glColorPointer(4, GL_FLOAT,
                CUDAStreamlines::vboStride*sizeof(float),
                (const GLvoid*)((long int)(offs + CUDAStreamlines::vboOffsCol*sizeof(float))));

        glDrawArrays(GL_LINE_STRIP_ADJACENCY, 0, this->nSegments+1);
        ::CheckForGLError(); // OpenGL error check
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    return ::CheckForGLError();
}


/*
 * CUDAStreamlines::SampleScalarFieldToAlpha
 */
bool CUDAStreamlines::SampleScalarFieldToAlpha(
        float *field,
        int3 fieldDim,
        float3 fieldOrg,
        float3 fieldDelta) {

    // Get pointer to line strip
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *lineStrip_D;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&lineStrip_D), // The mapped pointer
            &vboSize,                  // The size of the accessible data
            this->cudaToken))) {                 // The mapped resource
        return false;
    }

    // Init constant grid parameters of the vector field
    if (!initGridParams(fieldDim, fieldOrg, fieldDelta)) {
        return false;
    }

    size_t latticeSize = fieldDim.x*fieldDim.y*fieldDim.z;

    // Copy scalar field to device memory
    if (!CudaSafeCall(this->sclField_D.Validate(latticeSize))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->sclField_D.Peek(), field,
            sizeof(float)*latticeSize, cudaMemcpyHostToDevice))) {
        return false;
    }
    ::CheckForCudaErrorSync();

    int vertexCnt = this->nStreamlines*(this->nSegments+1);
    CUDAStreamlines_SampleScalarFieldToAlpha_D <<< Grid(vertexCnt, 256), 256 >>>(
            lineStrip_D,
            this->sclField_D.Peek(),
            this->nStreamlines,
            this->nSegments,
            CUDAStreamlines::vboOffsPos,
            CUDAStreamlines::vboOffsCol,
            CUDAStreamlines::vboStride);

    if (!::CheckForCudaErrorSync()){
        return false;
    }

    // Unmap the device pointer
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->cudaToken))) {
        return false;
    }

    return CheckForCudaErrorSync();
}


bool CUDAStreamlines::SampleVecFieldToRGB(
        float *field,
        int3 fieldDim,
        float3 fieldOrg,
        float3 fieldDelta) {

    // Get pointer to line strip
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *lineStrip_D;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&lineStrip_D), // The mapped pointer
            &vboSize,                  // The size of the accessible data
            this->cudaToken))) {                 // The mapped resource
        return false;
    }

    // Init constant grid parameters of the vector field
    if (!initGridParams(fieldDim, fieldOrg, fieldDelta)) {
        return false;
    }

    size_t latticeSize = fieldDim.x*fieldDim.y*fieldDim.z;

    // Copy vector field to device memory
    if (!CudaSafeCall(this->vecField_D.Validate(latticeSize*3))) {
        return false;
    }
    if (!CudaSafeCall(cudaMemcpy(this->vecField_D.Peek(), field,
            sizeof(float)*latticeSize*3, cudaMemcpyHostToDevice))) {
        return false;
    }
    ::CheckForCudaErrorSync();

    int vertexCnt = this->nStreamlines*(this->nSegments+1);
    CUDAStreamlines_SampleVecFieldToRGB_D <<< Grid(vertexCnt, 256), 256 >>>(
            lineStrip_D,
            this->vecField_D.Peek(),
            this->nStreamlines,
            this->nSegments,
            CUDAStreamlines::vboOffsPos,
            CUDAStreamlines::vboOffsCol,
            CUDAStreamlines::vboStride);

    if (!::CheckForCudaErrorSync()){
        return false;
    }


    // Unmap the device pointer
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->cudaToken))) {
        return false;
    }

    return CheckForCudaErrorSync();

}


/*
 * CUDAStreamlines::SetUniformRGBColor
 */
bool CUDAStreamlines::SetUniformRGBColor(float3 col) {

    // Get pointer to line strip
    if (!CudaSafeCall(cudaGraphicsMapResources(1, &this->cudaToken, 0))) {
        return false;
    }

    // Get mapped pointers to the vertex data buffers
    float *lineStrip_D;
    size_t vboSize;
    if (!CudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&lineStrip_D), // The mapped pointer
            &vboSize,                  // The size of the accessible data
            this->cudaToken))) {                 // The mapped resource
        return false;
    }

    int vertexCnt = this->nStreamlines*(this->nSegments+1);
    CUDAStreamlines_SetUniformRGBColor_D <<< Grid(vertexCnt, 256), 256 >>>(
            lineStrip_D,
            this->nStreamlines,
            this->nSegments,
            CUDAStreamlines::vboOffsCol,
            CUDAStreamlines::vboStride,
            col);

    if (!::CheckForCudaErrorSync()){
        return false;
    }


    // Unmap the device pointer
    if (!CudaSafeCall(cudaGraphicsUnmapResources(1, &this->cudaToken))) {
        return false;
    }

    return CheckForCudaErrorSync();

}


/*
 * CUDAStreamlines::destroyVBO
 */
void CUDAStreamlines::destroyVBO() {
    if (this->lineStripVBO) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->lineStripVBO);
        glDeleteBuffersARB(1, &this->lineStripVBO);
        this->lineStripVBO = 0;
        ::CheckForGLError();
        CudaSafeCall(cudaGraphicsUnregisterResource(this->cudaToken));
    }
}


/*
 * CUDAStreamlines::initVBO
 */
bool CUDAStreamlines::initVBO() {
    // Destroy if necessary
    if (this->lineStripVBO != 0) {
        glBindBufferARB(GL_ARRAY_BUFFER, this->lineStripVBO);
        glDeleteBuffersARB(1, &this->lineStripVBO);
        this->lineStripVBO = 0;
    }

    // Create vertex buffer object for vertex data
    glGenBuffersARB(1, &this->lineStripVBO);
    glBindBufferARB(GL_ARRAY_BUFFER, this->lineStripVBO);
    glBufferDataARB(GL_ARRAY_BUFFER,
            this->nStreamlines*(this->nSegments+1)*sizeof(float)*CUDAStreamlines::vboStride,
            0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    // Register buffer with cuda token
    if (!CudaSafeCall(cudaGraphicsGLRegisterBuffer(
            &this->cudaToken,
            this->lineStripVBO,
            cudaGraphicsMapFlagsNone))) {
        return false;
    }

    return CheckForGLError();
}
