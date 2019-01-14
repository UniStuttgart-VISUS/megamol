/*
 * AOAOSphereRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>

#include "AOSphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "vislib/assert.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include "vislib/math/ShallowVector.h"
#include "protein_calls/CallMouseInput.h"


#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace demos {
/*
 * AOSphereRenderer::AOSphereRenderer
 */
AOSphereRenderer::AOSphereRenderer(void) : megamol::core::view::Renderer3DModuleDS(),
        sphereShaderAOMainAxes(), sphereShaderAONormals(),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
        greyTF(0),
		renderFlagSlot("renderOnOff", "Turn rendering on/off"),
        volSizeXSlot("vol::sizex", "The size of the volume in numbers of voxels"),
        volSizeYSlot("vol::sizey", "The size of the volume in numbers of voxels"),
        volSizeZSlot("vol::sizez", "The size of the volume in numbers of voxels"),
        volGenSlot("vol::gen", "The generation method"),
        volAccSlot("ao::acc", "The access method"),
        aoStepLengthSlot("ao::stepLen", "The access step length in voxels"),
        aoGenFacSlot("vol::genFac", "The generation factor (influence factor of a single sphere on a voxel)"),
        aoEvalFacSlot("ao::evalFac", "The evaluation factor (shadowing amount factor multiplied with the ambient occlusion factors)"),
        aoShadModeSlot("ao::shadeMode", "The shading mode"),
        aoClipFlagSlot("ao::clipData", "Clip ao data"),
        volTex(0), volFBO(0)
        , particleVBO( 0), particleCountVBO( 0)
{

    this->getDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->getDataSlot.SetCompatibleCall<protein_calls::MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<megamol::core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<megamol::core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

	this->renderFlagSlot << new core::param::BoolParam( true);
	this->MakeSlotAvailable( &this->renderFlagSlot);

    this->volSizeXSlot << new core::param::IntParam(32, 4);
    this->MakeSlotAvailable(&this->volSizeXSlot);
    this->volSizeXSlot.ForceSetDirty();
    this->volSizeYSlot << new core::param::IntParam(32, 4);
    this->MakeSlotAvailable(&this->volSizeYSlot);
    this->volSizeZSlot << new core::param::IntParam(32, 4);
    this->MakeSlotAvailable(&this->volSizeZSlot);

    core::param::EnumParam *gen = new core::param::EnumParam(0);
    gen->SetTypePair(0, "CPU");
    gen->SetTypePair(1, "GPU (GLSL)");
    this->volGenSlot << gen;
    this->MakeSlotAvailable(&this->volGenSlot);

    core::param::EnumParam *acc = new core::param::EnumParam(0);
    acc->SetTypePair(0, "main axes");
    acc->SetTypePair(1, "normal-based");
    this->volAccSlot << acc;
    this->MakeSlotAvailable(&this->volAccSlot);

    this->aoStepLengthSlot << new core::param::FloatParam(0.5f, 0.0f);
    this->MakeSlotAvailable(&this->aoStepLengthSlot);

    this->aoGenFacSlot << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->aoGenFacSlot);

    this->aoEvalFacSlot << new core::param::FloatParam(1.0f, 0.0f);
    this->MakeSlotAvailable(&this->aoEvalFacSlot);

    this->aoClipFlagSlot << new core::param::BoolParam( false);
    this->MakeSlotAvailable( &this->aoClipFlagSlot);

    core::param::EnumParam *sm = new core::param::EnumParam(0);
    sm->SetTypePair(0, "Lighting + AO");
    sm->SetTypePair(1, "Colour + AO");
    sm->SetTypePair(2, "AO");
    sm->SetTypePair(3, "Multiple RT");
    this->aoShadModeSlot << sm;
    this->MakeSlotAvailable(&this->aoShadModeSlot);

    this->clipDat[0] = 0.f;
    this->clipDat[1] = 0.f;
    this->clipDat[2] = 0.f;
    this->clipDat[3] = 0.f;
    this->clipCol[0] = 0.f;
    this->clipCol[1] = 0.f;
    this->clipCol[2] = 0.f;
}


/*
 * AOSphereRenderer::~AOSphereRenderer
 */
AOSphereRenderer::~AOSphereRenderer(void) {
    this->Release();
}


/*
 * AOSphereRenderer::create
 */
bool AOSphereRenderer::create(void) {
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }
    if (!::areExtsAvailable("GL_ARB_multitexture GL_EXT_framebuffer_object") || !ogl_IsVersionGEQ(2,0)) {
        return false;
    }
    const char* maFragNames[] = {
        "AOSphere::mainaxes::fragLightAO",
        "AOSphere::mainaxes::fragColourAO",
        "AOSphere::mainaxes::fragAO",
        "AOSphere::mainaxes::fragMultipleRT"};
    const char* nFragNames[] = {
        "AOSphere::normals::fragLightAO",
        "AOSphere::normals::fragColourAO",
        "AOSphere::normals::fragAO",
        "AOSphere::normals::fragMultipleRT"};
    vislib::StringA shaderName("unknown");
    vislib::graphics::gl::ShaderSource vert, frag;

    try {

        for (int i = 0; i < 4; i++) {
            shaderName.Format("main axes %d", i);
            if (!instance()->ShaderSourceFactory().MakeShaderSource("AOSphere::mainaxes::vertex", vert)) {
                throw vislib::Exception("Unable to load vertex shader", __FILE__, __LINE__);
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(maFragNames[i], frag)) {
                throw vislib::Exception("Unable to load fragment shader", __FILE__, __LINE__);
            }

            if (!this->sphereShaderAOMainAxes[i].Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
                throw vislib::Exception("Shader creation failed", __FILE__, __LINE__);
            }

            shaderName.Format("normals %d", i);
            if (!instance()->ShaderSourceFactory().MakeShaderSource("AOSphere::normals::vertex", vert)) {
                throw vislib::Exception("Unable to load vertex shader", __FILE__, __LINE__);
            }
            if (!instance()->ShaderSourceFactory().MakeShaderSource(nFragNames[i], frag)) {
                throw vislib::Exception("Unable to load fragment shader", __FILE__, __LINE__);
            }

            if (!this->sphereShaderAONormals[i].Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
                throw vislib::Exception("Shader creation failed", __FILE__, __LINE__);
            }

        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (%s) (@%s): %s\n", shaderName.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (%s): %s\n", shaderName.PeekBuffer(), e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (%s): Unknown exception\n", shaderName.PeekBuffer());
        return false;
    }

    // Load volume texture generation shader
    if ( !instance()->ShaderSourceFactory().MakeShaderSource ( "AOSphere::volume::updateVolumeVertex", vert ) ) {
        vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_ERROR,
            "%: Unable to load vertex shader source for volume texture update shader",
            this->ClassName() );
        return false;
    }
    if ( !instance()->ShaderSourceFactory().MakeShaderSource ( "AOSphere::volume::updateVolumeFragment", frag ) ) {
        vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to load fragment shader source for volume texture update shader",
            this->ClassName() );
        return false;
    }
    try {
        if ( !this->updateVolumeShader.Create ( vert.Code(), vert.Count(), frag.Code(), frag.Count() ) ) {
            throw vislib::Exception ( "Generic creation failure", __FILE__, __LINE__ );
        }
    } catch ( vislib::Exception e ) {
        vislib::sys::Log::DefaultLog.WriteMsg ( vislib::sys::Log::LEVEL_ERROR,
            "%s: Unable to create volume texture update shader: %s\n",
            this->ClassName(), e.GetMsgA() );
        return false;
    }

    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {
        0, 0, 0,  255, 255, 255
    };
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);

    glDisable(GL_TEXTURE_1D);

    ::glEnable(GL_TEXTURE_3D);
    ::glGenTextures(1, &this->volTex);
    //::glBindTexture(GL_TEXTURE_3D, this->volTex);
    //::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

    return true;
}


/*
 * AOSphereRenderer::GetExtents
 */
bool AOSphereRenderer::GetExtents(megamol::core::Call& call) {
    megamol::core::view::AbstractCallRender3D *cr = dynamic_cast<megamol::core::view::AbstractCallRender3D*>(&call);
    if (cr == NULL) return false;

    megamol::core::moldyn::MultiParticleDataCall *c2 = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    protein_calls::MolecularDataCall *mol = this->getDataSlot.CallAs<protein_calls::MolecularDataCall>();
    if ((c2 != NULL) && ((*c2)(1))) {
        cr->SetTimeFramesCount(c2->FrameCount());
        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    } else if ((mol != NULL) && ((*mol)(protein_calls::MolecularDataCall::CallForGetExtent))) {
        cr->SetTimeFramesCount(mol->FrameCount());
        cr->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


/*
 * AOSphereRenderer::release
 */
void AOSphereRenderer::release(void) {
    for (unsigned int i = 0; i < 3; i++) {
        this->sphereShaderAOMainAxes[i].Release();
        this->sphereShaderAONormals[i].Release();
    }
    ::glDeleteTextures(1, &this->greyTF);

    if( glIsBuffer( this->particleVBO) ) {
        glDeleteBuffers( 1, &this->particleVBO);
    }
}


/*
 * AOSphereRenderer::Render
 */
bool AOSphereRenderer::Render(megamol::core::Call& call) {
    megamol::core::view::AbstractCallRender3D *cr = dynamic_cast<megamol::core::view::AbstractCallRender3D*>(&call);
    if (cr == NULL) return false;

    megamol::core::moldyn::MultiParticleDataCall *c2 = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    protein_calls::MolecularDataCall *mol = this->getDataSlot.CallAs<protein_calls::MolecularDataCall>();

    float scaling = 1.0f;
    if (c2 != NULL) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(1)) return false;

        // calculate scaling
        megamol::core::BoundingBoxes bboxs = c2->AccessBoundingBoxes();
        scaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }

        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(0)) return false;
        c2->AccessBoundingBoxes() = bboxs;

        megamol::core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<megamol::core::view::CallClipPlane>();
        if ((ccp != NULL) && (*ccp)()) {
            clipDat[0] = ccp->GetPlane().Normal().X();
            clipDat[1] = ccp->GetPlane().Normal().Y();
            clipDat[2] = ccp->GetPlane().Normal().Z();
            vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
            clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
            clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
            clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
            clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;

        } else {
            clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
            clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        }

        // resize the volume, if necessary
        this->resizeVolume();

        // volume generation method
        if( this->volGenSlot.IsDirty() ) {
            this->createEmptyVolume();
            this->volGenSlot.ResetDirty();
        }

        //time_t t = clock();
        if( this->volGenSlot.Param<megamol::core::param::EnumParam>()->Value() == 1 ) {
            this->createVolumeGLSL(*c2);
        }
        else {
            this->createVolumeCPU(*c2);
        }
        //std::cout << "time for volume generation: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;

		if(this->renderFlagSlot.Param<core::param::BoolParam>()->Value()) {
			this->renderParticles(cr, c2, scaling);
		}

        if (c2 != NULL) {
            c2->Unlock();
        }
    } else if (mol != NULL) {
        mol->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*mol)(protein_calls::MolecularDataCall::CallForGetExtent)) return false;

        // calculate scaling
        megamol::core::BoundingBoxes bboxs = mol->AccessBoundingBoxes();
        scaling = mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }

        mol->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*mol)(protein_calls::MolecularDataCall::CallForGetData)) return false;
        mol->AccessBoundingBoxes() = bboxs;

        // resize the volume, if necessary
        this->resizeVolume();

        // volume generation method
        if( this->volGenSlot.IsDirty() ) {
            this->createEmptyVolume();
            this->volGenSlot.ResetDirty();
        }

        //time_t t = clock();
        if( this->volGenSlot.Param<megamol::core::param::EnumParam>()->Value() == 1 ) {
            this->createVolumeGLSL(*mol);
        }
        else {
            this->createVolumeCPU(*mol);
        }
        //std::cout << "time for volume generation: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;

		if(this->renderFlagSlot.Param<core::param::BoolParam>()->Value()) {
			this->renderParticles(cr, mol, scaling);
		}

        if (mol != NULL) {
            mol->Unlock();
        }
    } else {
        return false;
    }

    // DEBUG ...
    /*
    int w = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
    int h = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
    int d = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();
    int vs = w * h * d;
    int ss = w * h;
    float *vol = new float[vs];
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    ::glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT, vol);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);
    for( unsigned int i = 1; i <= vs; i ++ ) {
        std::cout << "voxel " << i << " : " << vol[i-1];
        if( (i % ss) == 0 )
            std::cout << std::endl << std::endl;
        else if( (i % w) == 0 )
            std::cout << " ." << std::endl;
        else
            std::cout << std::endl;
    }
    delete[] vol;
    */
    // ... DEBUG

    return true;
}


/*
 * AOSphereRenderer::resizeVolume
 */
void AOSphereRenderer::resizeVolume() {
    if (this->volSizeXSlot.IsDirty() || this->volSizeYSlot.IsDirty() || this->volSizeZSlot.IsDirty()) {
        this->volSizeXSlot.ResetDirty();
        this->volSizeYSlot.ResetDirty();
        this->volSizeZSlot.ResetDirty();

        int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
        int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
        int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();

        float *dat = new float[sx * sy * sz];
        ::memset(dat, 0, sizeof(float) * sx * sy * sz); // empty volume to ensure initialization of border

        //for (int z = 0; z < sz; z++) {
        //    for (int y = 0; y < sy; y++) {
        //        for (int x = 0; x < sx; x++) {
        //            dat[x + (y + z * sy) * sx] = 1.0f;
        //                //static_cast<float>((x + y + z) % 2);
        //                //static_cast<float>(((x + 1) / 2 + (y + 1) / 2 + (z + 1) / 2) % 2);
        //                //static_cast<float>(x + y + z)
        //                /// static_cast<float>(sx + sy + sz - 3);
        //            if ((x == 0) || (y == 0) || (z == 0)
        //                    || (x == sx - 1) || (y == sy - 1) || (z == sz - 1)) {
        //                dat[x + (y + z * sy) * sx] = 0.0f;
        //            }
        //        }
        //    }
        //}

        ::glEnable(GL_TEXTURE_3D);
        ::glBindTexture(GL_TEXTURE_3D, this->volTex);
        ::glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE32F_ARB, sx, sy, sz,
            0, GL_LUMINANCE, GL_FLOAT, dat);
        ::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        ::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        //::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        //::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        ::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        ::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        ::glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        //float borCol[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        //::glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borCol);
        ::glBindTexture(GL_TEXTURE_3D, 0);
        ::glDisable(GL_TEXTURE_3D);

        delete[] dat;
    }

}

/*
 * AOSphereRenderer::renderParticles
 */
void AOSphereRenderer::renderParticles(megamol::core::view::AbstractCallRender3D *cr,
        megamol::core::moldyn::MultiParticleDataCall *c2, float scaling) {

    vislib::graphics::gl::GLSLShader *sphereShader = NULL;

    int shadMod = this->aoShadModeSlot.Param<megamol::core::param::EnumParam>()->Value();
    if (this->volAccSlot.Param<megamol::core::param::EnumParam>()->Value() == 1) {
        sphereShader = &this->sphereShaderAONormals[shadMod];
    } else {
        sphereShader = &this->sphereShaderAOMainAxes[shadMod];
    }

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

    sphereShader->Enable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    sphereShader->SetParameter("aoVol", 0);

    glUniform4fvARB(sphereShader->ParameterLocation("viewAttr"),
        1, viewportStuff);
    glUniform3fvARB(sphereShader->ParameterLocation("camIn"),
        1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camRight"),
        1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camUp"),
        1, cr->GetCameraParameters()->Up().PeekComponents());
    glUniform2f(sphereShader->ParameterLocation("frustumPlanes"),
        cr->GetCameraParameters()->NearClip(),
        cr->GetCameraParameters()->FarClip());

    glUniform4fvARB(sphereShader->ParameterLocation("clipDat"), 1, clipDat);
    glUniform3fvARB(sphereShader->ParameterLocation("clipCol"), 1, clipCol);


    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();
    float minOSx = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Left(); // * scaling;
    float minOSy = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom(); // * scaling;
    float minOSz = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Back(); // * scaling;
    float rangeOSx = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Width(); // * scaling;
    float rangeOSy = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Height(); // * scaling;
    float rangeOSz = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Depth(); // * scaling;

    rangeOSx /= (1.0f - 2.0f / static_cast<float>(sx));
    rangeOSy /= (1.0f - 2.0f / static_cast<float>(sy));
    rangeOSz /= (1.0f - 2.0f / static_cast<float>(sz));

    minOSx -= rangeOSx / static_cast<float>(sx);
    minOSy -= rangeOSy / static_cast<float>(sy);
    minOSz -= rangeOSz / static_cast<float>(sz);

    sphereShader->SetParameter("posOrigin", minOSx, minOSy, minOSz);
    sphereShader->SetParameter("posExtents", rangeOSx, rangeOSy, rangeOSz);
    float aoSampDist = this->aoStepLengthSlot.Param<megamol::core::param::FloatParam>()->Value();
    sphereShader->SetParameter("aoSampDist",
        aoSampDist * (c2->AccessBoundingBoxes().ObjectSpaceClipBox().Width()
            / this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (c2->AccessBoundingBoxes().ObjectSpaceClipBox().Height()
            / this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (c2->AccessBoundingBoxes().ObjectSpaceClipBox().Depth()
            / this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value()));
    sphereShader->SetParameter("aoSampFact",
        this->aoEvalFacSlot.Param<megamol::core::param::FloatParam>()->Value());

    glScalef(scaling, scaling, scaling);

    ::glActiveTexture(GL_TEXTURE1);

    if (c2 != NULL) {
        unsigned int cial = glGetAttribLocationARB(*sphereShader, "colIdx");

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
            float minC = 0.0f, maxC = 0.0f;
            unsigned int colTabSize = 0;

            // colour
            switch (parts.GetColourDataType()) {
                case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(parts.GetGlobalColour());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial);
                    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE,
                        parts.GetColourDataStride(), parts.GetColourData());

                    glEnable(GL_TEXTURE_1D);

                    megamol::core::view::CallGetTransferFunction *cgtf
                        = this->getTFSlot.CallAs<megamol::core::view::CallGetTransferFunction>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1iARB(sphereShader->ParameterLocation("colTab"), 1);
                    minC = parts.GetMinColourIndexValue();
                    maxC = parts.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
            }

            // radius and position
            switch (parts.GetVertexDataType()) {
                case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(sphereShader->ParameterLocation("inConsts1"),
                        parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(sphereShader->ParameterLocation("inConsts1"),
                        -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(4, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableVertexAttribArrayARB(cial);
            glDisable(GL_TEXTURE_1D);
        }

    }

    sphereShader->Disable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

/*
 * AOSphereRenderer::renderParticles
 */
void AOSphereRenderer::renderParticles(megamol::core::view::AbstractCallRender3D *cr,
    protein_calls::MolecularDataCall *mol, float scaling) {

    vislib::graphics::gl::GLSLShader *sphereShader = NULL;
    int shadMod = this->aoShadModeSlot.Param<megamol::core::param::EnumParam>()->Value();
    if (this->volAccSlot.Param<megamol::core::param::EnumParam>()->Value() == 1) {
        sphereShader = &this->sphereShaderAONormals[shadMod];
    } else {
        sphereShader = &this->sphereShaderAOMainAxes[shadMod];
    }

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

    sphereShader->Enable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    sphereShader->SetParameter("aoVol", 0);

    glUniform4fvARB(sphereShader->ParameterLocation("viewAttr"),
        1, viewportStuff);
    glUniform3fvARB(sphereShader->ParameterLocation("camIn"),
        1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camRight"),
        1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camUp"),
        1, cr->GetCameraParameters()->Up().PeekComponents());
    glUniform2f(sphereShader->ParameterLocation("frustumPlanes"),
        cr->GetCameraParameters()->NearClip(),
        cr->GetCameraParameters()->FarClip());

    glUniform4fvARB(sphereShader->ParameterLocation("clipDat"), 1, clipDat);
    glUniform3fvARB(sphereShader->ParameterLocation("clipCol"), 1, clipCol);

    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();
    float minOSx = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Left(); // * scaling;
    float minOSy = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom(); // * scaling;
    float minOSz = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Back(); // * scaling;
    float rangeOSx = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Width(); // * scaling;
    float rangeOSy = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Height(); // * scaling;
    float rangeOSz = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Depth(); // * scaling;

    rangeOSx /= (1.0f - 2.0f / static_cast<float>(sx));
    rangeOSy /= (1.0f - 2.0f / static_cast<float>(sy));
    rangeOSz /= (1.0f - 2.0f / static_cast<float>(sz));

    minOSx -= rangeOSx / static_cast<float>(sx);
    minOSy -= rangeOSy / static_cast<float>(sy);
    minOSz -= rangeOSz / static_cast<float>(sz);

    sphereShader->SetParameter("posOrigin", minOSx, minOSy, minOSz);
    sphereShader->SetParameter("posExtents", rangeOSx, rangeOSy, rangeOSz);
    float aoSampDist = this->aoStepLengthSlot.Param<megamol::core::param::FloatParam>()->Value();
    sphereShader->SetParameter("aoSampDist",
        aoSampDist * (mol->AccessBoundingBoxes().ObjectSpaceClipBox().Width()
            / this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (mol->AccessBoundingBoxes().ObjectSpaceClipBox().Height()
            / this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (mol->AccessBoundingBoxes().ObjectSpaceClipBox().Depth()
            / this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value()));
    sphereShader->SetParameter("aoSampFact",
        this->aoEvalFacSlot.Param<megamol::core::param::FloatParam>()->Value());

    glScalef(scaling, scaling, scaling);

    ::glActiveTexture(GL_TEXTURE1);

    if (mol != NULL) {
        unsigned int cial = glGetAttribLocationARB(*sphereShader, "colIdx");

        float minC = 0.0f, maxC = 0.0f;
        unsigned int colTabSize = 0;

        int cnt;

        // colour
        this->vertColours.SetCount( mol->AtomCount() * 4 );
#pragma omp parallel for
        for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
            this->vertColours[3*cnt+0] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[0];
            this->vertColours[3*cnt+1] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[1];
            this->vertColours[3*cnt+2] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Colour()[2];
        }
        glEnableClientState(GL_COLOR_ARRAY);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, this->vertColours.PeekElements());

        // radius and position
        this->vertSpheres.SetCount( mol->AtomCount() * 4 );
#pragma omp parallel for
        for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
            this->vertSpheres[4*cnt+0] = mol->AtomPositions()[3*cnt+0];
            this->vertSpheres[4*cnt+1] = mol->AtomPositions()[3*cnt+1];
            this->vertSpheres[4*cnt+2] = mol->AtomPositions()[3*cnt+2];
            this->vertSpheres[4*cnt+3] =
                mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
        }
        glEnableClientState(GL_VERTEX_ARRAY);
        glUniform4fARB(sphereShader->ParameterLocation("inConsts1"),
            -1.0, minC, maxC, float(colTabSize));
        glVertexPointer( 4, GL_FLOAT, 0, this->vertSpheres.PeekElements());

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mol->AtomCount()));

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableVertexAttribArrayARB(cial);
        glDisable(GL_TEXTURE_1D);
    }

    sphereShader->Disable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

/*
 * AOSphereRenderer::renderParticlesVBO
 */
void AOSphereRenderer::renderParticlesVBO(megamol::core::view::AbstractCallRender3D *cr,
        megamol::core::moldyn::MultiParticleDataCall *c2, float scaling) {

    vislib::graphics::gl::GLSLShader *sphereShader = NULL;
    int shadMod = this->aoShadModeSlot.Param<megamol::core::param::EnumParam>()->Value();
    if (this->volAccSlot.Param<megamol::core::param::EnumParam>()->Value() == 1) {
        sphereShader = &this->sphereShaderAONormals[shadMod];
    } else {
        sphereShader = &this->sphereShaderAOMainAxes[shadMod];
    }

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

    sphereShader->Enable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    sphereShader->SetParameter("aoVol", 0);

    glUniform4fvARB(sphereShader->ParameterLocation("viewAttr"),
        1, viewportStuff);
    glUniform3fvARB(sphereShader->ParameterLocation("camIn"),
        1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camRight"),
        1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camUp"),
        1, cr->GetCameraParameters()->Up().PeekComponents());

    glUniform4fvARB(sphereShader->ParameterLocation("clipDat"), 1, clipDat);
    glUniform3fvARB(sphereShader->ParameterLocation("clipCol"), 1, clipCol);

    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();
    float minOSx = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Left(); // * scaling;
    float minOSy = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom(); // * scaling;
    float minOSz = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Back(); // * scaling;
    float rangeOSx = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Width(); // * scaling;
    float rangeOSy = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Height(); // * scaling;
    float rangeOSz = c2->AccessBoundingBoxes().ObjectSpaceClipBox().Depth(); // * scaling;

    rangeOSx /= (1.0f - 2.0f / static_cast<float>(sx));
    rangeOSy /= (1.0f - 2.0f / static_cast<float>(sy));
    rangeOSz /= (1.0f - 2.0f / static_cast<float>(sz));

    minOSx -= rangeOSx / static_cast<float>(sx);
    minOSy -= rangeOSy / static_cast<float>(sy);
    minOSz -= rangeOSz / static_cast<float>(sz);

    sphereShader->SetParameter("posOrigin", minOSx, minOSy, minOSz);
    sphereShader->SetParameter("posExtents", rangeOSx, rangeOSy, rangeOSz);
    float aoSampDist = this->aoStepLengthSlot.Param<megamol::core::param::FloatParam>()->Value();
    sphereShader->SetParameter("aoSampDist",
        aoSampDist * (c2->AccessBoundingBoxes().ObjectSpaceClipBox().Width()
            / this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (c2->AccessBoundingBoxes().ObjectSpaceClipBox().Height()
            / this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (c2->AccessBoundingBoxes().ObjectSpaceClipBox().Depth()
            / this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value()));
    sphereShader->SetParameter("aoSampFact",
        this->aoEvalFacSlot.Param<megamol::core::param::FloatParam>()->Value());

    glScalef(scaling, scaling, scaling);

    ::glActiveTexture(GL_TEXTURE1);

    if( this->particleCountVBO > 0 ) {
        unsigned int cial = glGetAttribLocationARB(*sphereShader, "colIdx");

        glColor3ub( 255, 175, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        glUniform4fARB(sphereShader->ParameterLocation("inConsts1"),
            c2->AccessParticles( 0).GetGlobalRadius(), 0.0f, 0.0f, 0.0f);

        glBindBuffer( GL_ARRAY_BUFFER, this->particleVBO);
        glVertexPointer( 3, GL_FLOAT, 0, 0);
        glDrawArrays( GL_POINTS, 0, this->particleCountVBO);
        glBindBuffer( GL_ARRAY_BUFFER, 0);

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableVertexAttribArrayARB(cial);
        glDisable(GL_TEXTURE_1D);

    }

    sphereShader->Disable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

/*
 * AOSphereRenderer::renderParticlesVBO
 */
void AOSphereRenderer::renderParticlesVBO(megamol::core::view::AbstractCallRender3D *cr,
    protein_calls::MolecularDataCall *mol, float scaling) {

    vislib::graphics::gl::GLSLShader *sphereShader = NULL;
    int shadMod = this->aoShadModeSlot.Param<megamol::core::param::EnumParam>()->Value();
    if (this->volAccSlot.Param<megamol::core::param::EnumParam>()->Value() == 1) {
        sphereShader = &this->sphereShaderAONormals[shadMod];
    } else {
        sphereShader = &this->sphereShaderAOMainAxes[shadMod];
    }

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

    sphereShader->Enable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    sphereShader->SetParameter("aoVol", 0);

    glUniform4fvARB(sphereShader->ParameterLocation("viewAttr"),
        1, viewportStuff);
    glUniform3fvARB(sphereShader->ParameterLocation("camIn"),
        1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camRight"),
        1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fvARB(sphereShader->ParameterLocation("camUp"),
        1, cr->GetCameraParameters()->Up().PeekComponents());

    glUniform4fvARB(sphereShader->ParameterLocation("clipDat"), 1, clipDat);
    glUniform3fvARB(sphereShader->ParameterLocation("clipCol"), 1, clipCol);

    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();
    float minOSx = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Left(); // * scaling;
    float minOSy = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Bottom(); // * scaling;
    float minOSz = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Back(); // * scaling;
    float rangeOSx = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Width(); // * scaling;
    float rangeOSy = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Height(); // * scaling;
    float rangeOSz = mol->AccessBoundingBoxes().ObjectSpaceClipBox().Depth(); // * scaling;

    rangeOSx /= (1.0f - 2.0f / static_cast<float>(sx));
    rangeOSy /= (1.0f - 2.0f / static_cast<float>(sy));
    rangeOSz /= (1.0f - 2.0f / static_cast<float>(sz));

    minOSx -= rangeOSx / static_cast<float>(sx);
    minOSy -= rangeOSy / static_cast<float>(sy);
    minOSz -= rangeOSz / static_cast<float>(sz);

    sphereShader->SetParameter("posOrigin", minOSx, minOSy, minOSz);
    sphereShader->SetParameter("posExtents", rangeOSx, rangeOSy, rangeOSz);
    float aoSampDist = this->aoStepLengthSlot.Param<megamol::core::param::FloatParam>()->Value();
    sphereShader->SetParameter("aoSampDist",
        aoSampDist * (mol->AccessBoundingBoxes().ObjectSpaceClipBox().Width()
            / this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (mol->AccessBoundingBoxes().ObjectSpaceClipBox().Height()
            / this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value()),
        aoSampDist * (mol->AccessBoundingBoxes().ObjectSpaceClipBox().Depth()
            / this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value()));
    sphereShader->SetParameter("aoSampFact",
        this->aoEvalFacSlot.Param<megamol::core::param::FloatParam>()->Value());

    glScalef(scaling, scaling, scaling);

    ::glActiveTexture(GL_TEXTURE1);

    if( this->particleCountVBO > 0 ) {
        unsigned int cial = glGetAttribLocationARB(*sphereShader, "colIdx");

        glColor3ub( 255, 175, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
		//TODO fix radius
        glUniform4fARB(sphereShader->ParameterLocation("inConsts1"),
            1.7f, 0.0f, 0.0f, 0.0f);

        glBindBuffer( GL_ARRAY_BUFFER, this->particleVBO);
        glVertexPointer( 3, GL_FLOAT, 0, 0);
        glDrawArrays( GL_POINTS, 0, this->particleCountVBO);
        glBindBuffer( GL_ARRAY_BUFFER, 0);

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableVertexAttribArrayARB(cial);
        glDisable(GL_TEXTURE_1D);

    }

    sphereShader->Disable();

    ::glActiveTexture(GL_TEXTURE0);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

/*
 * AOSphereRenderer::createEmptyVolume
 */
void AOSphereRenderer::createEmptyVolume() {
    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value();
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value();
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value();
    float *vol = new float[sx * sy * sz];
    ::memset(vol, 0, sizeof(float) * sx * sy * sz);
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    ::glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, sx, sy, sz, GL_LUMINANCE, GL_FLOAT, vol);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);
    delete[] vol;
}


/*
 * AOSphereRenderer::createFullVolume
 */
void AOSphereRenderer::createFullVolume() {
    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    float *vol = new float[sx * sy * sz];
    for (int i = (sx * sy * sz) - 1; i >= 0; i--) {
        vol[i] = 1.0f;
    }
    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    ::glTexSubImage3D(GL_TEXTURE_3D, 0, 1, 1, 1, sx, sy, sz, GL_LUMINANCE, GL_FLOAT, vol);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);
    delete[] vol;
}


/*
 * AOSphereRenderer::createVolumeCPU
 */
void AOSphereRenderer::createVolumeCPU(class megamol::core::moldyn::MultiParticleDataCall& c2) {
    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
	float **vol = new float*[omp_get_max_threads()];
	int init, j;
#pragma omp parallel for
	for( init = 0; init < omp_get_max_threads(); init++ ) {
		vol[init] = new float[sx * sy * sz];
		::memset(vol[init], 0, sizeof(float) * sx * sy * sz);
	}

    float minOSx = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    float volGenFac = this->aoGenFacSlot.Param<megamol::core::param::FloatParam>()->Value();
    float voxelVol = (rangeOSx / static_cast<float>(sx))
        * (rangeOSy / static_cast<float>(sy))
        * (rangeOSz / static_cast<float>(sz));

    for (unsigned int i = 0; i < c2.GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2.AccessParticles(i);
        const float *pos = static_cast<const float*>(parts.GetVertexData());
        unsigned int posStride = parts.GetVertexDataStride();
        float globRad = parts.GetGlobalRadius();
        bool useGlobRad = (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }
        if (useGlobRad) {
            if (posStride < 12) posStride = 12;
        } else {
            if (posStride < 16) posStride = 16;
        }
        float globSpVol = 4.0f / 3.0f * static_cast<float>(M_PI) * globRad * globRad * globRad;
        bool clipAOData = this->aoClipFlagSlot.Param<core::param::BoolParam>()->Value();

#pragma omp parallel for
        for (j = 0; j < parts.GetCount(); j++ ) {
			const float *ppos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride * j);

			int x = static_cast<int>(((ppos[0] - minOSx) / rangeOSx) * static_cast<float>(sx));
            if (x < 0) x = 0; else if (x >= sx) x = sx - 1;

            int y = static_cast<int>(((ppos[1] - minOSy) / rangeOSy) * static_cast<float>(sy));
            if (y < 0) y = 0; else if (y >= sy) y = sy - 1;

            int z = static_cast<int>(((ppos[2] - minOSz) / rangeOSz) * static_cast<float>(sz));
            if (z < 0) z = 0; else if (z >= sz) z = sz - 1;

            if (clipAOData) {
                // Discard if outside clipping plane
                if (clipDat) {
                    vislib::math::Vector<float, 3> pos(ppos);
                    vislib::math::Vector<float, 3> clipNorm(clipDat);
                    float od = pos.Dot(clipNorm) - clipDat[3];
                    if (od >= 0) continue;
                }
            }

            float spVol = globSpVol;
            if (!useGlobRad) {
                float rad = ppos[3];
                spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;
            }

			vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (spVol / voxelVol) * volGenFac;

        }

    }

#pragma omp parallel for
    for (j = 0; j < sx * sy * sz; j++ ) {
		for ( unsigned int i = 1; i < omp_get_max_threads(); i++ ) {
			vol[0][j] += vol[i][j];
		}
    }

    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
	::glTexSubImage3D(GL_TEXTURE_3D, 0, 1, 1, 1, sx, sy, sz, GL_LUMINANCE, GL_FLOAT, vol[0]);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

#pragma omp parallel for
	for( init = 0; init < omp_get_max_threads(); init++ ) {
		delete[] vol[init];
	}
    delete[] vol;
}

/*
 * AOSphereRenderer::createVolumeGLSL
 */
void AOSphereRenderer::createVolumeGLSL(class megamol::core::moldyn::MultiParticleDataCall& c2) {
    // get volume size (number of voxels)
    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value() - 2;

    // get clip box origin and size
    float minOSx = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = c2.AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    // get the volume generation factor
    float volGenFac = this->aoGenFacSlot.Param<megamol::core::param::FloatParam>()->Value();
    // compute voxel volume
    float voxelVol = (rangeOSx / static_cast<float>(sx))
        * (rangeOSy / static_cast<float>(sy))
        * (rangeOSz / static_cast<float>(sz));

    // generate FBO, if necessary
    if( !glIsFramebufferEXT( this->volFBO ) ) {
        glGenFramebuffersEXT( 1, &this->volFBO);
    }

    // counter variable
    unsigned int z;

    // store current frame buffer object ID
    GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    glMatrixMode( GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // store old viewport
    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport);
    // set viewport
    glViewport( 0, 0, sx + 2, sy + 2);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->volFBO);
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    glColor4f( 1.0, 1.0, 1.0, 1.0);

    float bgColor[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor( 0.0, 0.0, 0.0, 0.0);
    // clear 3d texture
    for( z = 0; z < sz + 2; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volTex, 0, z);
        glClear( GL_COLOR_BUFFER_BIT);
    }
    glClearColor( bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    this->updateVolumeShader.Enable();

    // set shader params
    glUniform3f( this->updateVolumeShader.ParameterLocation( "minOS"), minOSx, minOSy, minOSz);
    glUniform3f( this->updateVolumeShader.ParameterLocation( "rangeOS"), rangeOSx, rangeOSy, rangeOSz);
    glUniform1f( this->updateVolumeShader.ParameterLocation( "genFac"), volGenFac);
    glUniform1f( this->updateVolumeShader.ParameterLocation( "voxelVol"), voxelVol);
    glUniform3f( this->updateVolumeShader.ParameterLocation( "volSize"), float( sx), float( sy), float( sz));

#if 1
    UINT64 numParticles = 0;
    for( unsigned int i = 0; i < c2.GetParticleListCount(); i++ ) {
        numParticles += c2.AccessParticles(i).GetCount();
    }

    UINT64 particleCnt = 0;

    if( this->sphereSlices.Count() != sz || this->sphereCountSlices.Count() != sz ) {
        // set number of slices
        this->sphereSlices.SetCount( sz);
        this->sphereCountSlices.SetCount( sz);
        // set slice capacity and reset sphere number per slice
        for( unsigned int sliceCnt = 0; sliceCnt < sz; ++sliceCnt ) {
            this->sphereSlices[sliceCnt].SetCount( numParticles * 4);
            this->sphereCountSlices[sliceCnt] = 0;
        }
    } else {
        // reset sphere number per slice
        for( unsigned int sliceCnt = 0; sliceCnt < sz; ++sliceCnt ) {
            this->sphereCountSlices[sliceCnt] = 0;
        }
    }

    // write particle list (VA, pos + rad)
    for (unsigned int i = 0; i < c2.GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2.AccessParticles(i);
        const float *pos = static_cast<const float*>(parts.GetVertexData());
        unsigned int posStride = parts.GetVertexDataStride();
        float globRad = parts.GetGlobalRadius();
        bool useGlobRad = (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }
        if (useGlobRad) {
            if (posStride < 12) posStride = 12;
        } else {
            if (posStride < 16) posStride = 16;
        }

        for (UINT64 j = 0; j < parts.GetCount(); j++,
                pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride)) {
            z = static_cast<int>(((pos[2] - minOSz) / rangeOSz) * static_cast<float>(sz));
			z = vislib::math::Min<int>( z, this->sphereCountSlices.Count()-1);
			this->sphereSlices[z][this->sphereCountSlices[z]*4+0] = pos[0];
			this->sphereSlices[z][this->sphereCountSlices[z]*4+1] = pos[1];
			this->sphereSlices[z][this->sphereCountSlices[z]*4+2] = pos[2];
			this->sphereSlices[z][this->sphereCountSlices[z]*4+3] = ( useGlobRad ? globRad : pos[3]);
			this->sphereCountSlices[z]++;
        }
    }

    glPointSize( 1.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glUniform1f( this->updateVolumeShader.ParameterLocation( "radius"), -1.0f);
    for( z = 0; z < sz; ++z ) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volTex, 0, z + 1);
        glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), float( z + 1));
        // set vertex pointers and draw them
        glVertexPointer( 4, GL_FLOAT, 0, this->sphereSlices[z].PeekElements());
        glDrawArrays( GL_POINTS, 0, this->sphereCountSlices[z]);
    }
    glDisableClientState(GL_VERTEX_ARRAY);

#else
    // START TEST
    glPointSize( 1.0f);
    for( z = 0; z < sz; ++z ) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volTex, 0, z + 1);
        glUniform1f( this->updateVolumeShader.ParameterLocation( "sliceDepth"), float( z + 1));
        for (unsigned int i = 0; i < c2.GetParticleListCount(); i++) {
            megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2.AccessParticles(i);

            // radius and position
            switch (parts.GetVertexDataType()) {
                case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform1f( this->updateVolumeShader.ParameterLocation("radius"),
                        parts.GetGlobalRadius());
                    glVertexPointer(3, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform1f( this->updateVolumeShader.ParameterLocation("radius"),
                        -1.0f);
                    glVertexPointer(4, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_VERTEX_ARRAY);
        }
    }
    // END TEST
#endif

    this->updateVolumeShader.Disable();

    // restore viewport
    glViewport( viewport[0], viewport[1], viewport[2], viewport[3]);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFBO);

    glDisable( GL_BLEND);
    glEnable( GL_DEPTH_TEST);
    glDepthMask( GL_TRUE);
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

/*
 * AOSphereRenderer::createVolumeGLSL
 */
void AOSphereRenderer::createVolumeGLSL(protein_calls::MolecularDataCall & mol) {
    // get volume size (number of voxels)
    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value() - 2;

    // get clip box origin and size
    float minOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    // get the volume generation factor
    float volGenFac = this->aoGenFacSlot.Param<megamol::core::param::FloatParam>()->Value();
    // compute voxel volume
    float voxelVol = (rangeOSx / static_cast<float>(sx))
        * (rangeOSy / static_cast<float>(sy))
        * (rangeOSz / static_cast<float>(sz));

    // generate FBO, if necessary
    if (!glIsFramebufferEXT(this->volFBO)) {
        glGenFramebuffersEXT(1, &this->volFBO);
    }

    // counter variable
    unsigned int z;

    // store current frame buffer object ID
    GLint prevFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &prevFBO);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // store old viewport
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    // set viewport
    glViewport(0, 0, sx + 2, sy + 2);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, this->volFBO);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    glColor4f(1.0, 1.0, 1.0, 1.0);

    float bgColor[4];
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    // clear 3d texture
    for (z = 0; z < sz + 2; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, this->volTex, 0, z);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glClearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    this->updateVolumeShader.Enable();

    // set shader params
    glUniform3f(this->updateVolumeShader.ParameterLocation("minOS"), minOSx, minOSy, minOSz);
    glUniform3f(this->updateVolumeShader.ParameterLocation("rangeOS"), rangeOSx, rangeOSy, rangeOSz);
    glUniform1f(this->updateVolumeShader.ParameterLocation("genFac"), volGenFac);
    glUniform1f(this->updateVolumeShader.ParameterLocation("voxelVol"), voxelVol);
    glUniform3f(this->updateVolumeShader.ParameterLocation("volSize"), float(sx), float(sy), float(sz));

    UINT64 particleCnt = 0;

    if (this->sphereSlices.Count() != sz || this->sphereCountSlices.Count() != sz) {
        // set number of slices
        this->sphereSlices.SetCount(sz);
        this->sphereCountSlices.SetCount(sz);
        // set slice capacity and reset sphere number per slice
        for (unsigned int sliceCnt = 0; sliceCnt < sz; ++sliceCnt) {
            this->sphereSlices[sliceCnt].SetCount(mol.AtomCount() * 4);
            this->sphereCountSlices[sliceCnt] = 0;
        }
    } else {
        // reset sphere number per slice
        for (unsigned int sliceCnt = 0; sliceCnt < sz; ++sliceCnt) {
            this->sphereCountSlices[sliceCnt] = 0;
        }
    }

    // write particle list (VA, pos + rad)
    for (unsigned int i = 0; i < mol.AtomCount(); i++) {
        z = static_cast<int>(((mol.AtomPositions()[i * 3 + 2] - minOSz) / rangeOSz) * static_cast<float>(sz));
        z = vislib::math::Min<int>(z, this->sphereCountSlices.Count() - 1);
        this->sphereSlices[z][this->sphereCountSlices[z] * 4 + 0] = mol.AtomPositions()[i * 3 + 0];
        this->sphereSlices[z][this->sphereCountSlices[z] * 4 + 1] = mol.AtomPositions()[i * 3 + 1];
        this->sphereSlices[z][this->sphereCountSlices[z] * 4 + 2] = mol.AtomPositions()[i * 3 + 2];
        this->sphereSlices[z][this->sphereCountSlices[z] * 4 + 3] = mol.AtomTypes()[mol.AtomTypeIndices()[i]].Radius();
        this->sphereCountSlices[z]++;
    }

    glPointSize(1.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glUniform1f(this->updateVolumeShader.ParameterLocation("radius"), -1.0f);
    for (z = 0; z < sz; ++z) {
        // attach texture slice to FBO
        glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, this->volTex, 0, z + 1);
        glUniform1f(this->updateVolumeShader.ParameterLocation("sliceDepth"), float(z + 1));
        // set vertex pointers and draw them
        glVertexPointer(4, GL_FLOAT, 0, this->sphereSlices[z].PeekElements());
        glDrawArrays(GL_POINTS, 0, this->sphereCountSlices[z]);
    }
    glDisableClientState(GL_VERTEX_ARRAY);

    this->updateVolumeShader.Disable();

    // restore viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, prevFBO);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}


/*
 * Write particle positions and radii to a VBO for rendering and processing in CUDA
 */
void AOSphereRenderer::writeParticlePositionsVBO(class megamol::core::moldyn::MultiParticleDataCall& c2) {
    // count total number of particles
    this->particleCountVBO = 0;
    for( unsigned int i = 0; i < c2.GetParticleListCount(); i++ ) {
        this->particleCountVBO += c2.AccessParticles(i).GetCount();
    }

    bool newlyGenerated = false;
    // generate buffer, if not already available
    if( !glIsBuffer( this->particleVBO) ) {
        glGenBuffers( 1, &this->particleVBO);
        newlyGenerated = true;
    }
    // bind buffer, enable the vertex array client state and resize the vbo accordingly
    glBindBuffer( GL_ARRAY_BUFFER, this->particleVBO);
    if( newlyGenerated )
        glBufferData( GL_ARRAY_BUFFER, this->particleCountVBO*3*sizeof(float), 0, GL_DYNAMIC_DRAW);
    float *particleVBOPtr = static_cast<float*>(glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE));

    unsigned int particleCnt = 0;
    for (unsigned int i = 0; i < c2.GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2.AccessParticles(i);
        // radius and position
        switch( parts.GetVertexDataType() ) {
            case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                //glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                ASSERT( parts.GetVertexDataStride() == 3*sizeof(float) || parts.GetVertexDataStride() == 0 );
                memcpy( particleVBOPtr + particleCnt*3, parts.GetVertexData(), parts.GetCount()*3*sizeof(float));
                particleCnt += parts.GetCount();
                break;
            case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                continue;
            default:
                continue;
        }
    }

    // unmap the buffer, disable the vertex array client state and unbind the vbo
    glUnmapBuffer( GL_ARRAY_BUFFER);
    glBindBuffer( GL_ARRAY_BUFFER, 0);

}

/*
 * Write particle positions and radii to a VBO for rendering and processing in CUDA
 */
void AOSphereRenderer::writeParticlePositionsVBO(protein_calls::MolecularDataCall& mol) {
    // count total number of particles
	this->particleCountVBO = mol.AtomCount();

    bool newlyGenerated = false;
    CHECK_FOR_OGL_ERROR();
    // generate buffer, if not already available
    if( !glIsBuffer( this->particleVBO) ) {
        glGenBuffers( 1, &this->particleVBO);
        newlyGenerated = true;
    }
    // bind buffer, enable the vertex array client state and resize the vbo accordingly
    glBindBuffer( GL_ARRAY_BUFFER, this->particleVBO);
    if( newlyGenerated )
        glBufferData( GL_ARRAY_BUFFER, mol.AtomCount()*3*sizeof(float), mol.AtomPositions(), GL_DYNAMIC_DRAW);
    //void *particleVBOPtr = glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	// radius and position
	//memcpy( particleVBOPtr, static_cast<const void*>(mol.AtomPositions()), mol.AtomCount()*3*sizeof(float));
    
    // unmap the buffer, disable the vertex array client state and unbind the vbo
    //if (!glUnmapBuffer( GL_ARRAY_BUFFER)) {
    //    printf("unmapping buffer exploded!\n");
    //}
    glBindBuffer( GL_ARRAY_BUFFER, 0);
    //CHECK_FOR_OGL_ERROR();
}

/*
 * AOSphereRenderer::createVolumeCPU
 */
void AOSphereRenderer::createVolumeCPU(protein_calls::MolecularDataCall& mol) {
    int sx = this->volSizeXSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sy = this->volSizeYSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
    int sz = this->volSizeZSlot.Param<megamol::core::param::IntParam>()->Value() - 2;
	float **vol = new float*[omp_get_max_threads()];
	int init, i, j;
#pragma omp parallel for
	for( init = 0; init < omp_get_max_threads(); init++ ) {
		vol[init] = new float[sx * sy * sz];
		::memset(vol[init], 0, sizeof(float) * sx * sy * sz);
	}

    float minOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    float volGenFac = this->aoGenFacSlot.Param<megamol::core::param::FloatParam>()->Value();
    float voxelVol = (rangeOSx / static_cast<float>(sx))
        * (rangeOSy / static_cast<float>(sy))
        * (rangeOSz / static_cast<float>(sz));

#pragma omp parallel for
    for (i = 0; i < mol.AtomCount(); i++ ) {
        int x = static_cast<int>(((mol.AtomPositions()[i*3+0] - minOSx) / rangeOSx) * static_cast<float>(sx));
        if (x < 0) x = 0; else if (x >= sx) x = sx - 1;
        int y = static_cast<int>(((mol.AtomPositions()[i*3+1] - minOSy) / rangeOSy) * static_cast<float>(sy));
        if (y < 0) y = 0; else if (y >= sy) y = sy - 1;
        int z = static_cast<int>(((mol.AtomPositions()[i*3+2] - minOSz) / rangeOSz) * static_cast<float>(sz));
        if (z < 0) z = 0; else if (z >= sz) z = sz - 1;
        float rad = mol.AtomTypes()[mol.AtomTypeIndices()[i]].Radius();
        float spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;

        vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (spVol / voxelVol) * volGenFac;

    }

#pragma omp parallel for
    for (j = 0; j < sx * sy * sz; j++ ) {
		for ( unsigned int i = 1; i < omp_get_max_threads(); i++ ) {
			vol[0][j] += vol[i][j];
		}
    }

    ::glEnable(GL_TEXTURE_3D);
    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
    ::glTexSubImage3D(GL_TEXTURE_3D, 0, 1, 1, 1, sx, sy, sz, GL_LUMINANCE, GL_FLOAT, vol[0]);
    ::glBindTexture(GL_TEXTURE_3D, 0);
    ::glDisable(GL_TEXTURE_3D);

#pragma omp parallel for
	for( init = 0; init < omp_get_max_threads(); init++ ) {
		delete[] vol[init];
	}
    delete[] vol;
}

} /* end namespace demos */
} /* end namespace megamol */