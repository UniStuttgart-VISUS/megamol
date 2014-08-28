/*
 * NGSphereRenderer.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/IncludeAllGL.h"
#include "NGSphereRenderer.h"
#include "MultiParticleDataCall.h"
#include "CoreInstance.h"
#include "view/CallClipPlane.h"
#include "view/CallGetTransferFunction.h"
#include "view/CallRender3D.h"
#include "vislib/assert.h"
#include "vislib/mathfunctions.h"

using namespace megamol::core;


/*
 * moldyn::NGSphereRenderer::NGSphereRenderer
 */
moldyn::NGSphereRenderer::NGSphereRenderer(void) : AbstractSimpleSphereRenderer(),
    sphereShader(), bufSize(50 * 1024 * 1024) {
    // intentionally empty
}


/*
 * moldyn::NGSphereRenderer::~NGSphereRenderer
 */
moldyn::NGSphereRenderer::~NGSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool moldyn::NGSphereRenderer::create(void) {

    vislib::graphics::gl::ShaderSource vert, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::fragment", frag)) {
        return false;
    }

    //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    //    vert.WholeCode().PeekBuffer(),
    //    frag.WholeCode().PeekBuffer());

    try {
        if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n", 
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return false;
    }

    glGenVertexArrays(1, &this->vertArray);
    glBindVertexArray(this->vertArray);
    glGenBuffers(1, &this->vertBuffer);
    glGenBuffers(1, &this->colBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertBuffer);
    glBufferStorage(GL_ARRAY_BUFFER, this->bufSize, NULL, GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, this->colBuffer);
    glBufferStorage(GL_ARRAY_BUFFER, this->bufSize, NULL, GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    mappedVertexMem = glMapNamedBufferRangeEXT(this->vertBuffer, 0, this->bufSize, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_PERSISTENT_BIT);
    mappedColorMem = glMapNamedBufferRangeEXT(this->colBuffer, 0, this->bufSize, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_PERSISTENT_BIT);

    return AbstractSimpleSphereRenderer::create();
}


/*
 * moldyn::SimpleSphereRenderer::release
 */
void moldyn::NGSphereRenderer::release(void) {
    glUnmapNamedBufferEXT(this->vertBuffer);
    glUnmapNamedBufferEXT(this->colBuffer);
    this->sphereShader.Release();
    glDeleteBuffers(1, &this->vertBuffer);
    glDeleteBuffers(1, &this->colBuffer);
    glDeleteVertexArrays(1, &this->vertArray);
    AbstractSimpleSphereRenderer::release();
}


void moldyn::NGSphereRenderer::setPointers(MultiParticleDataCall::Particles &parts, GLuint vertBuf, const void *vertPtr, GLuint colBuf, const void *colPtr) {
    float minC = 0.0f, maxC = 0.0f;
    unsigned int colTabSize = 0;

    // colour
    glBindBuffer(GL_ARRAY_BUFFER, colBuf);
    switch (parts.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE:
            glColor3ubv(parts.GetGlobalColour());
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), colPtr);
            break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
            glEnableVertexAttribArrayARB(colIdxAttribLoc);
            glVertexAttribPointerARB(colIdxAttribLoc, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), colPtr);

            glEnable(GL_TEXTURE_1D);

            view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
            if ((cgtf != NULL) && ((*cgtf)())) {
                glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                colTabSize = cgtf->TextureSize();
            } else {
                glBindTexture(GL_TEXTURE_1D, this->greyTF);
                colTabSize = 2;
            }

            glUniform1i(this->sphereShader.ParameterLocation("colTab"), 0);
            minC = parts.GetMinColourIndexValue();
            maxC = parts.GetMaxColourIndexValue();
            glColor3ub(127, 127, 127);
        } break;
        default:
            glColor3ub(127, 127, 127);
            break;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vertBuf);
    // radius and position
    switch (parts.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            glEnableClientState(GL_VERTEX_ARRAY);
            glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
            glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), vertPtr);
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glEnableClientState(GL_VERTEX_ARRAY);
            glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), vertPtr);
            break;
        default:
            break;
    }
}


/*
 * moldyn::SimpleSphereRenderer::Render
 */
bool moldyn::NGSphereRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    float scaling = 1.0f;
    MultiParticleDataCall *c2 = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (c2 == NULL) return false;

    float clipDat[4];
    float clipCol[4];
    this->getClipData(clipDat, clipCol);
	
	glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    this->sphereShader.Enable();
    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());

    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);

    glScalef(scaling, scaling, scaling);

    colIdxAttribLoc = glGetAttribLocationARB(this->sphereShader, "colIdx");

    for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
        MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);


        unsigned int colBytes = 0, vertBytes = 0;
        switch (parts.GetColourDataType()) {
            case MultiParticleDataCall::Particles::COLDATA_NONE:
                // nothing
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                colBytes = vislib::math::Max(colBytes, 3U);
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                colBytes = vislib::math::Max(colBytes, 4U);
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                colBytes = vislib::math::Max(colBytes, 3 * 4U);
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                colBytes = vislib::math::Max(colBytes, 4 * 4U);
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    colBytes = vislib::math::Max(colBytes, 1 * 4U);
                    // nothing else
                }
                break;
            default:
                // nothing
                break;
        }

        // radius and position
        switch (parts.GetVertexDataType()) {
            case MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vertBytes = vislib::math::Max(vertBytes, 3 * 4U);
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vertBytes = vislib::math::Max(vertBytes, 4 * 4U);
                break;
            default:
                continue;
        }

        unsigned int colStride = parts.GetColourDataStride();
        colStride = colStride < colBytes ? colBytes : colStride;
        unsigned int vertStride = parts.GetVertexDataStride();
        vertStride = vertStride < vertBytes ? vertBytes : vertStride;
        UINT64 numVerts, vertCounter;

        // does all data reside interleaved in the same memory?
        if ((reinterpret_cast<const ptrdiff_t>(parts.GetColourData()) 
                - reinterpret_cast<const ptrdiff_t>(parts.GetVertexData()) <= vertStride
                && vertStride == colStride) || colStride == 0)  {
                    GLuint vb = this->vertBuffer;
                    numVerts = this->bufSize / vertStride;
                    const char *currVert = static_cast<const char *>(parts.GetVertexData());
                    const char *currCol = static_cast<const char *>(parts.GetColourData());
                    vertCounter = 0;
                    while (vertCounter < parts.GetCount()) {
                        currCol = colStride == 0 ? currVert : currCol;
                        const char *whence = currVert < currCol ? currVert : currCol;
                        UINT64 vertsThisTime = vislib::math::Min(parts.GetCount() - vertCounter, numVerts);
                        //mappedVertexMem = glMapNamedBufferRangeEXT(this->vertBuffer, 0, this->bufSize, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
                        //mappedColorMem = glMapNamedBufferRangeEXT(this->colBuffer, 0, this->bufSize, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
                        //void *buf = glMapNamedBufferRangeEXT(this->vertBuffer, 0, vertsThisTime * vertStride, GL_WRITE_ONLY | GL_MAP_FLUSH_EXPLICIT_BIT);
                        //memcpy(buf, whence, vertsThisTime * vertStride);
                        //glUnmapNamedBufferEXT(this->vertBuffer);
                        //glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
                        memcpy(this->mappedVertexMem, whence, vertsThisTime * vertStride);
                        glFlushMappedNamedBufferRangeEXT(this->vertBuffer, 0, vertsThisTime * vertStride);
                        //glUnmapNamedBufferEXT(this->vertBuffer);
                        //glUnmapNamedBufferEXT(this->colBuffer);

                        this->setPointers(parts, vb, reinterpret_cast<const void *>(currVert - whence), vb, reinterpret_cast<const void *>(currCol - whence));
                        glDrawArrays(GL_POINTS, 0, vertsThisTime);
                        vertCounter += vertsThisTime;
                        currVert += vertsThisTime * vertStride;
                        currCol += vertsThisTime * colStride;
                    }
        } else {

        }



        //glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableVertexAttribArrayARB(colIdxAttribLoc);
        glDisable(GL_TEXTURE_1D);
    }

    c2->Unlock();

    this->sphereShader.Disable();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    return true;
}
