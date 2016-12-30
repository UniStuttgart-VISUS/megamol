/*
 * UncertaintyCartoonRenderer.cpp
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "UncertaintyCartoonRenderer" in megamol protein plugin (svn revision 1511).
 *
 */

//////////////////////////////////////////////////////////////////////////////////////////////
//
// TODO:
//
//    - ...
// 
//////////////////////////////////////////////////////////////////////////////////////////////


#include "stdafx.h"
#include "UncertaintyCartoonRenderer.h"

#include <inttypes.h>
#include <stdint.h>

#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/param/ButtonParam.h"

#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"


using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_uncertainty;


#define MAP_BUFFER_LOCALLY

//#define NGS_THE_INSTANCE "gl_InstanceID"
#define NGS_THE_INSTANCE "gl_VertexID"

// #define DEBUG_GL

const GLuint SSBObindingPoint = 2;


/*
* MyFunkyDebugCallback
*/
//typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
void APIENTRY MyFunkyDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const GLvoid* userParam) {
	const char *sourceText, *typeText, *severityText;
	switch (source) {
	case GL_DEBUG_SOURCE_API:
		sourceText = "API";
		break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
		sourceText = "Window System";
		break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER:
		sourceText = "Shader Compiler";
		break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:
		sourceText = "Third Party";
		break;
	case GL_DEBUG_SOURCE_APPLICATION:
		sourceText = "Application";
		break;
	case GL_DEBUG_SOURCE_OTHER:
		sourceText = "Other";
		break;
	default:
		sourceText = "Unknown";
		break;
	}
	switch (type) {
	case GL_DEBUG_TYPE_ERROR:
		typeText = "Error";
		break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		typeText = "Deprecated Behavior";
		break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		typeText = "Undefined Behavior";
		break;
	case GL_DEBUG_TYPE_PORTABILITY:
		typeText = "Portability";
		break;
	case GL_DEBUG_TYPE_PERFORMANCE:
		typeText = "Performance";
		break;
	case GL_DEBUG_TYPE_OTHER:
		typeText = "Other";
		break;
	case GL_DEBUG_TYPE_MARKER:
		typeText = "Marker";
		break;
	default:
		typeText = "Unknown";
		break;
	}
	switch (severity) {
	case GL_DEBUG_SEVERITY_HIGH:
		severityText = "High";
		break;
	case GL_DEBUG_SEVERITY_MEDIUM:
		severityText = "Medium";
		break;
	case GL_DEBUG_SEVERITY_LOW:
		severityText = "Low";
		break;
	case GL_DEBUG_SEVERITY_NOTIFICATION:
		severityText = "Notification";
		break;
	default:
		severityText = "Unknown";
		break;
	}
	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[%s %s] (%s %u) %s\n", sourceText, severityText, typeText, id, message);
}


/*
 * UncertaintyCartoonRenderer::UncertaintyCartoonRenderer (CTOR)
 */ 
UncertaintyCartoonRenderer::UncertaintyCartoonRenderer(void) : Renderer3DModule(),
        getPdbDataSlot(         "getPdbData", "Connects to the pdb data source."),
        uncertaintyDataSlot(    "uncertaintyDataSlot", "Connects the cartoon tesselation rendering with uncertainty data storage."),
		resSelectionCallerSlot( "getResSelection", "Connects the cartoon rendering with residue selection storage."),
        scalingParam(           "01 Scaling", "Scaling factor for particle radii."),
        sphereParam(            "02 Spheres", "Render atoms as spheres."),
        lineParam(              "03 Lines", "Render backbone as GL_LINE."),
        backboneParam(          "04 Backbone", "Render backbone as tubes."),
        backboneWidthParam(     "05 Backbone width", "The width of the backbone."),
        materialParam(          "06 Material", "Ambient, diffuse, specular components + exponent."),
        lineDebugParam(         "07 Wireframe", "Render in wireframe mode."),
        buttonParam(            "08 Reload shaders", "Reload the shaders."),
        colorInterpolationParam("09 Color interpolation", "Should the colors be interpolated?"),
		fences(), currBuf(0), bufSize(32 * 1024 * 1024), numBuffers(3), aminoAcidCount(0), resSelectionCall(NULL),
        // this variant should not need the fence
        singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT),
        singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT) 
        {

    // uncertainty data caller slot
    this->uncertaintyDataSlot.SetCompatibleCall<UncertaintyDataCallDescription>();
    this->MakeSlotAvailable(&this->uncertaintyDataSlot);

    // pdb data caller slot
    this->getPdbDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->getPdbDataSlot);

	// residue selection caller slot
	this->resSelectionCallerSlot.SetCompatibleCall<ResidueSelectionCallDescription>();
	this->MakeSlotAvailable(&this->resSelectionCallerSlot);


	this->sphereParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->sphereParam);

	this->lineParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->lineParam);

	this->backboneParam << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->backboneParam);

    this->scalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingParam);
    
	this->backboneWidthParam << new core::param::FloatParam(0.2f);
	this->MakeSlotAvailable(&this->backboneWidthParam);

	this->lineDebugParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->lineDebugParam);

	this->buttonParam << new core::param::ButtonParam(vislib::sys::KeyCode::KEY_F5);
	this->MakeSlotAvailable(&this->buttonParam);

	this->colorInterpolationParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->colorInterpolationParam);

	/*float components[4] = { 0.2f, 0.8f, 0.4f, 10.0f };
	vislib::math::Vector<float, 4U> myvec(components);
	this->materialParam << new core::param::Vector4fParam(myvec);
	this->MakeSlotAvailable(&this->materialParam);*/

	
	this->fences.resize(this->numBuffers);

#ifdef FIRSTFRAME_CHECK
	this->firstFrame = true;
#endif

}


/*
 * UncertaintyCartoonRenderer::~UncertaintyCartoonRenderer (DTOR)
 */
UncertaintyCartoonRenderer::~UncertaintyCartoonRenderer(void) {
    this->Release(); // DON'T change !
}


/*
 * UncertaintyCartoonRenderer::create
 */
bool UncertaintyCartoonRenderer::create(void) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;
    
#ifdef DEBUG_GL
    glDebugMessageCallback(MyFunkyDebugCallback, NULL);
#endif

    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
        return false;
    if (!vislib::graphics::gl::GLSLTesselationShader::InitialiseExtensions())
        return false;

	ShaderSource vertSrc; // sphere shader
	ShaderSource fragSrc; // sphere shader

    this->vert     = new ShaderSource();
    this->tessCont = new ShaderSource();
    this->tessEval = new ShaderSource();
    this->geom     = new ShaderSource();
    this->frag     = new ShaderSource();
    
    // load spline shader
    if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::vertex", *this->vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::tesscontrol", *this->tessCont)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::tesseval", *this->tessEval)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::geometry", *this->geom)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::fragment", *this->frag)) {
        return false;
    }
    try {
        // compile the shader
        if (!this->splineShader.Compile(this->vert->Code(),     this->vert->Count(),
                                        this->tessCont->Code(), this->tessCont->Count(),
                                        this->tessEval->Code(), this->tessEval->Count(),
                                        this->geom->Code(),     this->geom->Count(),
                                        this->frag->Code(),     this->frag->Count())) {
            throw vislib::Exception("Could not compile spline shader. ", __FILE__, __LINE__);
        }
        // link the shader
        if (!this->splineShader.Link()){
            throw vislib::Exception("Could not link spline shader", __FILE__, __LINE__);
        }
    }
    catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile tessellation shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()), ce.GetMsgA());
        return false;
    }
    catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile tessellation shader: %s\n", e.GetMsgA());
        return false;
    }
    catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile tessellation shader: Unknown exception\n");
        return false;
    }

	// load tube shader
	this->tubeVert     = new ShaderSource();
	this->tubeTessCont = new ShaderSource();
	this->tubeTessEval = new ShaderSource();
	this->tubeGeom     = new ShaderSource();
	this->tubeFrag     = new ShaderSource();
	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::vertex", *this->tubeVert)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::tesscontrol", *this->tubeTessCont)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::tesseval", *this->tubeTessEval)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::geometry", *this->tubeGeom)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::fragment", *this->tubeFrag)) {
		return false;
	}

	try {
		// compile the shader
		if (!this->tubeShader.Compile(this->tubeVert->Code(),     this->tubeVert->Count(),
                                      this->tubeTessCont->Code(), this->tubeTessCont->Count(),
                                      this->tubeTessEval->Code(), this->tubeTessEval->Count(),
                                      this->tubeGeom->Code(),     this->tubeGeom->Count(),
                                      this->tubeFrag->Code(),     this-> tubeFrag->Count())) {
			throw vislib::Exception("Could not compile tube shader. ", __FILE__, __LINE__);
		}
		// link the shader
		if (!this->tubeShader.Link()) {
			throw vislib::Exception("Could not link tube shader", __FILE__, __LINE__);
		}
	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile tessellation shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
				ce.FailedAction()), ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile tessellation shader: %s\n", e.GetMsgA());
		return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile tessellation shader: Unknown exception\n");
		return false;
	}

    // load sphere shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    try {
        if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
        return false;
    }

    glGenVertexArrays(1, &this->vertArray);
    glBindVertexArray(this->vertArray);
    glGenBuffers(1, &this->theSingleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, this->singleBufferCreationBits);
    this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, this->singleBufferMappingBits);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);

    return true;
}


/*
 * UncertaintyCartoonRenderer::release
 */
void UncertaintyCartoonRenderer::release(void) {
    
    glUnmapNamedBufferEXT(this->theSingleBuffer);
    for (auto &x : this->fences) {
        if (x) {
            glDeleteSync(x);
        }
    }
    
    // TODO release all shaders (done?)
    this->sphereShader.Release();
    this->splineShader.Release();
    this->tubeShader.Release();

    glDeleteVertexArrays(1, &this->vertArray);
    glDeleteBuffers(1, &this->theSingleBuffer);
}


/*
 * UncertaintyCartoonRenderer::GetCapabilities
 */
bool UncertaintyCartoonRenderer::GetCapabilities(Call& call) {
    
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING | view::CallRender3D::CAP_ANIMATION);

    return true;
}


/*
* UncertaintyCartoonRenderer::GetExtents
*/
bool UncertaintyCartoonRenderer::GetExtents(Call& call) {
    
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    // get pointer to UncertaintyDataCall
    UncertaintyDataCall *udc = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if (udc == NULL) return false;
    // execute the call
    if (!(*udc)(UncertaintyDataCall::CallForGetData)) return false;

    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->getPdbDataSlot.CallAs<MolecularDataCall>();
    if ((mol != NULL) && ((*mol)(MolecularDataCall::CallForGetExtent))) {
        cr->SetTimeFramesCount(mol->FrameCount());
        cr->AccessBoundingBoxes() = mol->AccessBoundingBoxes();

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        }
        else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    }
    else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


/*
 *  UncertaintyCartoonRenderer::GetData
 */
MolecularDataCall* UncertaintyCartoonRenderer::GetData(unsigned int t, float& outScaling) {
    
    MolecularDataCall *mol = this->getPdbDataSlot.CallAs<MolecularDataCall>();
    outScaling = 1.0f;
    
    if (mol != NULL) {

        mol->SetFrameID(t);
        if (!(*mol)(MolecularDataCall::CallForGetExtent)) return NULL;

        // calculate scaling
        outScaling = mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (outScaling > 0.0000001) {
            outScaling = 10.0f / outScaling;
        }
        else {
            outScaling = 1.0f;
        }

        mol->SetFrameID(t);
        if (!(*mol)(MolecularDataCall::CallForGetData)) return NULL;

        return mol;
    }
    else {
        return NULL;
    }
}


/*
* UncertaintyCartoonRenderer::GetUncertaintyData
*/
bool UncertaintyCartoonRenderer::GetUncertaintyData(UncertaintyDataCall *udc) {

	if (!udc) return false;

	// initialization
	this->aminoAcidCount = udc->GetAminoAcidCount();

	// reset arrays
	this->secStructColorRGB.Clear();
	this->secStructColorRGB.AssertCapacity(UncertaintyDataCall::secStructure::NOE);

	this->secStructColorHSL.Clear();
	this->secStructColorHSL.AssertCapacity(UncertaintyDataCall::secStructure::NOE);

	// get secondary structure type colors 
	for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
		this->secStructColorRGB.Add(udc->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(i)));
		// convert RGB secondary structure type colors from RGB to HSL
		// this->secStructColorHSL.Add(this->rgb2hsl(this->secStructColorRGB.Last()));
	}

	// reset arrays
	this->secUncertainty.Clear();
	this->secUncertainty.AssertCapacity(this->aminoAcidCount);

	this->sortedUncertainty.Clear();
	this->sortedUncertainty.AssertCapacity(this->aminoAcidCount);

	for (unsigned int i = 0; i < this->secStructAssignment.Count(); i++) {
		this->secStructAssignment.Clear();
	}
	this->secStructAssignment.Clear();

	for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); i++) {
		this->secStructAssignment.Add(vislib::Array<UncertaintyDataCall::secStructure>());
		this->secStructAssignment.Last().AssertCapacity(this->aminoAcidCount);
	}

	this->residueFlag.Clear();
	this->residueFlag.AssertCapacity(this->aminoAcidCount);

	// collect data from call
	for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

		// store the secondary structure element type of the current amino-acid for each assignment method
		for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
			this->secStructAssignment[k].Add(udc->GetSecStructure(static_cast<UncertaintyDataCall::assMethod>(k), aa));
		}

		// store residue flag
		this->residueFlag.Add(udc->GetResidueFlag(aa));
		// store uncerteiny values
		this->secUncertainty.Add(udc->GetSecStructUncertainty(aa));
		// store sorted uncertainty structure types
		this->sortedUncertainty.Add(udc->GetSortedSecStructureIndices(aa));
	}

	return true;
}


/*
 * UncertaintyCartoonRenderer::Render
 */
bool UncertaintyCartoonRenderer::Render(Call& call) {
    
#ifdef DEBUG_GL
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

	float scaling = 1.0f;

	// the pointer to the render call
	view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
	if (cr == NULL) return false;

	// get new data from the MolecularDataCall
	MolecularDataCall *mol = this->GetData(static_cast<unsigned int>(cr->Time()), scaling);
	if (mol == NULL) return false;

	// get pointer to UncertaintyDataCall
	UncertaintyDataCall *ud = this->uncertaintyDataSlot.CallAs<UncertaintyDataCall>();
    if(ud == NULL) return false;
    // execute the call
    if( !(*ud)(UncertaintyDataCall::CallForGetData)) return false;
    
	// if amino-acid count changed get new data
	if (ud->GetAminoAcidCount() != this->aminoAcidCount) {
		this->GetUncertaintyData(ud); // use return value ...?
	}


    // reload shaders
	if (this->buttonParam.IsDirty()) {
		this->buttonParam.ResetDirty();

		instance()->ShaderSourceFactory().LoadBTF("uncertaintycartoontessellation", true);

		// load tube shader
		this->tubeVert     = new ShaderSource();
		this->tubeTessCont = new ShaderSource();
		this->tubeTessEval = new ShaderSource();
		this->tubeGeom     = new ShaderSource();
		this->tubeFrag     = new ShaderSource();
		if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::vertex", *this->tubeVert)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::tesscontrol", *this->tubeTessCont)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::tesseval", *this->tubeTessEval)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::geometry", *this->tubeGeom)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("uncertaintycartoontessellation::fragment", *this->tubeFrag)) {
			return false;
		}

		try {
			// compile the shader
			if (!this->tubeShader.Compile(this->tubeVert->Code(),     this->tubeVert->Count(),
                                          this->tubeTessCont->Code(), this->tubeTessCont->Count(),
                                          this->tubeTessEval->Code(), this->tubeTessEval->Count(),
                                          this->tubeGeom->Code(),     this->tubeGeom->Count(),
                                          this->tubeFrag->Code(),     this->tubeFrag->Count())) {
				throw vislib::Exception("Could not compile tube shader. ", __FILE__, __LINE__);
			}
			// link the shader
			if (!this->tubeShader.Link()) {
				throw vislib::Exception("Could not link tube shader", __FILE__, __LINE__);
			}
		}
		catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile tessellation shader (@%s): %s\n",
				vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
					ce.FailedAction()), ce.GetMsgA());
			return false;
		}
		catch (vislib::Exception e) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile tessellation shader: %s\n", e.GetMsgA());
			return false;
		}
		catch (...) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Unable to compile tessellation shader: Unknown exception\n");
			return false;
		}
	}

	// timer.BeginFrame();

    float clipDat[4];
    float clipCol[4];
    clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
    clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
    clipCol[3] = 1.0f;
    
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

    glScalef(scaling, scaling, scaling);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->theSingleBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer);

    // matrices
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrix(&modelViewMatrix_column[0]);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    vislib::math::ShallowMatrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projMatrix(&projMatrix_column[0]);
    // Compute modelviewprojection matrix
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrixInv = modelViewMatrix;
    modelViewMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrix = projMatrix * modelViewMatrix;
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrixInv = modelViewProjMatrix;
    modelViewProjMatrixInv.Invert();
    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewProjMatrixTransp = modelViewProjMatrix;
    modelViewProjMatrixTransp.Transpose();
	vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> modelViewMatrixInvTrans = modelViewMatrixInv;
	modelViewMatrixInvTrans.Transpose();
	vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> projectionMatrixInv = projMatrix;
	projectionMatrixInv.Invert();

	GLfloat lightPos[4];
	glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);
	GLfloat lightAmbient[4];
	glGetLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
	GLfloat lightDiffuse[4];
	glGetLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
	GLfloat lightSpecular[4];
	glGetLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);

    // DEBUG
	/*std::cout << lightAmbient[0] << " " << lightAmbient[1] << " " << lightAmbient[2] << " " << lightAmbient[3] << std::endl;
	std::cout << lightDiffuse[0] << " " << lightDiffuse[1] << " " << lightDiffuse[2] << " " << lightDiffuse[3] << std::endl;
	std::cout << lightSpecular[0] << " " << lightSpecular[1] << " " << lightSpecular[2] << " " << lightSpecular[3] << std::endl;*/


	unsigned int firstResIdx = 0;
	unsigned int lastResIdx = 0;
	unsigned int firstAtomIdx = 0;
	unsigned int lastAtomIdx = 0;
	unsigned int atomTypeIdx = 0;
	unsigned int firstSecIdx = 0;
	unsigned int lastSecIdx = 0;
	unsigned int firstAAIdx = 0;
	unsigned int lastAAIdx = 0;

	// Render in wireframe mode
	if (this->lineDebugParam.Param<param::BoolParam>()->Value())
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// Render backbone as tubes
	if (this->backboneParam.Param<param::BoolParam>()->Value()) {

		// copy data to: mainChain
		firstResIdx = 0;
		lastResIdx = 0;
		firstAtomIdx = 0;
		lastAtomIdx = 0;
		atomTypeIdx = 0;
		firstSecIdx = 0;
		lastSecIdx = 0;
		firstAAIdx = 0;
		lastAAIdx = 0;

		unsigned int cIndex = 0;
		unsigned int oIndex = 0;

		this->mainChain.clear();

		CAlpha lastCalpha;

		int molCount = mol->MoleculeCount();
		std::vector<int> molSizes;

		// loop over all molecules of the protein
		for (unsigned int molIdx = 0; molIdx < mol->MoleculeCount(); molIdx++) {

			MolecularDataCall::Molecule chain = mol->Molecules()[molIdx];
			molSizes.push_back(0);

			bool firstset = false;

			// is the first residue an aminoacid?
			// if first residue is no aminoacid the whole secondary structure is skipped!
			if (mol->Residues()[chain.FirstResidueIndex()]->Identifier() != MolecularDataCall::Residue::AMINOACID) {
				continue;
			}

			firstSecIdx = chain.FirstSecStructIndex();
			lastSecIdx = firstSecIdx + chain.SecStructCount();

			// loop over all secondary structures of the molecule
			for (unsigned int secIdx = firstSecIdx; secIdx < lastSecIdx; secIdx++) {
				firstAAIdx = mol->SecondaryStructures()[secIdx].FirstAminoAcidIndex();
				lastAAIdx = firstAAIdx + mol->SecondaryStructures()[secIdx].AminoAcidCount();

				// loop over all aminoacids inside the secondary structure
				for (unsigned int aaIdx = firstAAIdx; aaIdx < lastAAIdx; aaIdx++) {

					MolecularDataCall::AminoAcid * acid;

					// is the current residue really an aminoacid?
					if (mol->Residues()[aaIdx]->Identifier() == MolecularDataCall::Residue::AMINOACID)
						acid = (MolecularDataCall::AminoAcid*)(mol->Residues()[aaIdx]);
					else
						continue;

					// extract relevant positions and other values
					CAlpha calpha;
					calpha.pos[0] = mol->AtomPositions()[3 * acid->CAlphaIndex()];
					calpha.pos[1] = mol->AtomPositions()[3 * acid->CAlphaIndex() + 1];
					calpha.pos[2] = mol->AtomPositions()[3 * acid->CAlphaIndex() + 2];
					calpha.pos[3] = 1.0f;

					// direction is vector from C_alpha atom to O(xygen) atom
					calpha.dir[0] = mol->AtomPositions()[3 * acid->OIndex()] - calpha.pos[0];
					calpha.dir[1] = mol->AtomPositions()[3 * acid->OIndex() + 1] - calpha.pos[1];
					calpha.dir[2] = mol->AtomPositions()[3 * acid->OIndex() + 2] - calpha.pos[2];

					auto type = mol->SecondaryStructures()[secIdx].Type(); // TYPE_COIL  = 0, TYPE_SHEET = 1, TYPE_HELIX = 2, TYPE_TURN  = 3
					calpha.type = (int)type;
					
					// TODO: do this on GPU?
					// orientation check for the direction
					if (this->mainChain.size() != 0)
					{
						CAlpha before = this->mainChain[this->mainChain.size() - 1];
						float dotProd = calpha.dir[0] * before.dir[0] + calpha.dir[1] * before.dir[1] + calpha.dir[2] * before.dir[2];

						if (dotProd < 0) // flip direction if the orientation is wrong
						{
							calpha.dir[0] = -calpha.dir[0];
							calpha.dir[1] = -calpha.dir[1];
							calpha.dir[2] = -calpha.dir[2];
						}
					}

					this->mainChain.push_back(calpha);
					molSizes[molIdx]++;

					lastCalpha = calpha;

					// add the first atom 3 times - for every different secondary structure
					if (!firstset) {
						this->mainChain.push_back(calpha);
						this->mainChain.push_back(calpha);
						molSizes[molIdx] += 2;
						firstset = true;
					}
				}
			}

			// add the last atom 3 times
			this->mainChain.push_back(lastCalpha);
			this->mainChain.push_back(lastCalpha);
			molSizes[molIdx] += 2;
		}
	
// DEBUG
#ifdef FIRSTFRAME_CHECK
		if (this->firstFrame) {
			for (int i = 0; i < this->mainChain.size(); i++) {
				std::cout << this->mainChain[i].type << std::endl;
			}
			this->firstFrame = false;
		}
#endif

		// draw backbone as tubes
		unsigned int colBytes, vertBytes, colStride, vertStride;
		this->GetBytesAndStride(*mol, colBytes, vertBytes, colStride, vertStride);
		//this->currBuf = 0;

		// number of different secondary structure types
		const unsigned int structCount = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);

		this->tubeShader.Enable();
		glColor4f(1.0f / this->mainChain.size(), 0.75f, 0.25f, 1.0f);
		colIdxAttribLoc = glGetAttribLocationARB(this->splineShader, "colIdx"); // NOT USED in shader (?)

		// vertex
		glUniform4fv(this->tubeShader.ParameterLocation("viewAttr"), 1, viewportStuff);

		glUniform1f(this->tubeShader.ParameterLocation("scaling"), this->scalingParam.Param<param::FloatParam>()->Value());

		glUniform3fv(this->tubeShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
		glUniform3fv(this->tubeShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
		glUniform3fv(this->tubeShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
		
		glUniform4fv(this->tubeShader.ParameterLocation("clipDat"), 1, clipDat);
		glUniform4fv(this->tubeShader.ParameterLocation("clipCol"), 1, clipCol);

		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MV"), 1, GL_FALSE, modelViewMatrix.PeekComponents());

		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVinvtrans"), 1, GL_FALSE, modelViewMatrixInvTrans.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());

		// only vertex shader
		float minC = 0.0f, maxC = 0.0f;
		unsigned int colTabSize = 0;
		glUniform4f(this->tubeShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
		
		// only tesselation evaluation shader
		glUniform1f(this->tubeShader.ParameterLocation("pipeWidth"), this->backboneWidthParam.Param<param::FloatParam>()->Value());
		glUniform1i(this->tubeShader.ParameterLocation("interpolateColors"), this->colorInterpolationParam.Param<param::BoolParam>()->Value());

		glUniform4fv(this->tubeShader.ParameterLocation("structColRGB"), structCount, (GLfloat *)this->secStructColorRGB.PeekElements());

		// only fragment shader
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("ProjInv"), 1, GL_FALSE, projectionMatrixInv.PeekComponents());
		glUniform4f(this->tubeShader.ParameterLocation("lightPos"), lightPos[0], lightPos[1], lightPos[2], lightPos[3]);
		glUniform4f(this->tubeShader.ParameterLocation("ambientColor"), lightAmbient[0], lightAmbient[1], lightAmbient[2], lightAmbient[3]);
		glUniform4f(this->tubeShader.ParameterLocation("diffuseColor"), lightDiffuse[0], lightDiffuse[1], lightDiffuse[2], lightDiffuse[3]);
		glUniform4f(this->tubeShader.ParameterLocation("specularColor"), lightSpecular[0], lightSpecular[1], lightSpecular[2], lightSpecular[3]);


		UINT64 numVerts;
		numVerts = this->bufSize / vertStride;                                                                                                       // bufSize = 32*1024*1024 - WHY? | vertStride = (unsigned int)sizeof(CAlpha)
		                                                                                                                                             // numVert = number of vertices fitting into bufSize
		UINT64 stride = 0;                                                                                                                           // aminoacid index in mainChain 

		for (int i = 0; i < (int)molSizes.size(); i++) {                                                                                             // loop over all secondary structures
			UINT64 vertCounter = 0;
			while (vertCounter < molSizes[i]) {                                                                                                      // loop over all aminoacids inside of one secondary structure - WHY ?

				const char *currVert = (const char *)(&this->mainChain[(unsigned int)vertCounter + (unsigned int)stride]);                           // pointer to current vertex data in mainChain

				void *mem            = static_cast<char*>(this->theSingleMappedMem) + this->bufSize * this->currBuf;                                 // pointer to the mapped memory - ?
				const char *whence   = currVert;                                                                                                     // copy of pointer currVert
				UINT64 vertsThisTime = vislib::math::Min(molSizes[i] - vertCounter, numVerts);                                                       // try to take all vertices of current secondary structure at once ... 
				                                                                                                                                     // ... or at least as many as fit into buffer of size bufSize
				this->WaitSignal(this->fences[this->currBuf]);                                                                                       // wait for buffer 'currBuf' to be "ready" - ?

				memcpy(mem, whence, (size_t)vertsThisTime * vertStride);                                                                             // copy data of current vertex data in mainChain to mapped memory - ?
				                             
				glFlushMappedNamedBufferRangeEXT(theSingleBuffer, this->bufSize * this->currBuf, (GLsizeiptr)vertsThisTime * vertStride);            // parameter: buffer, offset, length 

				glUniform1i(this->tubeShader.ParameterLocation("instanceOffset"), 0);                                                                // unused?

				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, this->bufSize * this->currBuf, this->bufSize);  // bind Shader Storage Buffer Object
				glPatchParameteri(GL_PATCH_VERTICES, 1);                                                                                             // set parameter GL_PATCH_VERTICES to 1 (the number of vertices that will be used to make up a single patch primitive)
				glDrawArrays(GL_PATCHES, 0, (GLsizei)(vertsThisTime - 3));                                                                           // draw as many as (vertsThisTime-3) patches 
				                                                                                                                                     // -3 ? - because the first atom is added 3 times for each different secondary structure
				this->QueueSignal(this->fences[this->currBuf]);                                                                                      // queue signal - tell that mapped memory 'operations' are done - ?

				this->currBuf = (this->currBuf + 1) % this->numBuffers;                                                                              // switch to next buffer in range 0...3
				vertCounter += vertsThisTime;                                                                                                        // increase counter of processed vertices
				currVert += vertsThisTime * vertStride;                                                                                              // unused - will be overwritten in next loop cycle
			}
			stride += molSizes[i];
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisable(GL_TEXTURE_1D);
		this->tubeShader.Disable();
	}


	// copy data to: positionCA and positionO for backbone line rendering or sphere rendering
	if (this->lineParam.Param<param::BoolParam>()->Value() || this->sphereParam.Param<param::BoolParam>()->Value()) {

		firstResIdx = 0;
		lastResIdx = 0;
		firstAtomIdx = 0;
		lastAtomIdx = 0;
		atomTypeIdx = 0;

		if (this->positionsCa.Count() != mol->MoleculeCount()) {
			this->positionsCa.SetCount(mol->MoleculeCount());
			this->positionsO.SetCount(mol->MoleculeCount());
		}

		for (unsigned int molIdx = 0; molIdx < mol->MoleculeCount(); molIdx++){
			this->positionsCa[molIdx].Clear();
			this->positionsCa[molIdx].AssertCapacity(mol->Molecules()[molIdx].ResidueCount() * 4 + 16);
			this->positionsO[molIdx].Clear();
			this->positionsO[molIdx].AssertCapacity(mol->Molecules()[molIdx].ResidueCount() * 4 + 16);

			//bool first;
			firstResIdx = mol->Molecules()[molIdx].FirstResidueIndex();
			lastResIdx = firstResIdx + mol->Molecules()[molIdx].ResidueCount();
			for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++){
				firstAtomIdx = mol->Residues()[resIdx]->FirstAtomIndex();
				lastAtomIdx = firstAtomIdx + mol->Residues()[resIdx]->AtomCount();
				for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++){
					unsigned int atomTypeIdx = mol->AtomTypeIndices()[atomIdx];
					if (mol->AtomTypes()[atomTypeIdx].Name().Equals("CA")){
						this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
						this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
						this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
						this->positionsCa[molIdx].Add(1.0f);
						// write first and last Ca position three times
						if ((resIdx == firstResIdx) || (resIdx == (lastResIdx - 1))){
							this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
							this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
							this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
							this->positionsCa[molIdx].Add(1.0f);
							this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
							this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
							this->positionsCa[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
							this->positionsCa[molIdx].Add(1.0f);
						}
					}
					if (mol->AtomTypes()[atomTypeIdx].Name().Equals("O")){
						this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
						this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
						this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
						this->positionsO[molIdx].Add(1.0f);
						// write first and last Ca position three times
						if ((resIdx == firstResIdx) || (resIdx == (lastResIdx - 1))){
							this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
							this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
							this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
							this->positionsO[molIdx].Add(1.0f);
							this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx]);
							this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 1]);
							this->positionsO[molIdx].Add(mol->AtomPositions()[3 * atomIdx + 2]);
							this->positionsO[molIdx].Add(1.0f);
						}
					}
				}
			}
		}
		// DEBUG
		//std::cout << "cIndex " << cIndex << " oIndex " << oIndex << " molCount " << mol->MoleculeCount() << std::endl;
	}

    // Render backbone as GL_LINE
	if (lineParam.Param<param::BoolParam>()->Value())
	{
		//this->currBuf = 0;
		for (unsigned int i = 0; i < this->positionsCa.Count(); i++) {
			unsigned int colBytes, vertBytes, colStride, vertStride;
			this->GetBytesAndStrideLines(*mol, colBytes, vertBytes, colStride, vertStride);

			this->splineShader.Enable();
			glColor4f(1.0f / this->positionsCa.Count() * (i + 1), 0.75f, 0.25f, 1.0f);
			colIdxAttribLoc = glGetAttribLocationARB(this->splineShader, "colIdx");
			glUniform4fv(this->splineShader.ParameterLocation("viewAttr"),        1, viewportStuff);
			glUniform3fv(this->splineShader.ParameterLocation("camIn"),           1, cr->GetCameraParameters()->Front().PeekComponents());
			glUniform3fv(this->splineShader.ParameterLocation("camRight"),        1, cr->GetCameraParameters()->Right().PeekComponents());
			glUniform3fv(this->splineShader.ParameterLocation("camUp"),           1, cr->GetCameraParameters()->Up().PeekComponents());
			glUniform4fv(this->splineShader.ParameterLocation("clipDat"),         1, clipDat);
			glUniform4fv(this->splineShader.ParameterLocation("clipCol"),         1, clipCol);
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVinv"),     1, GL_FALSE, modelViewMatrixInv.PeekComponents());
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVP"),       1, GL_FALSE, modelViewProjMatrix.PeekComponents());
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVPinv"),    1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
			glUniform1f(this->splineShader.ParameterLocation("scaling"),          this->scalingParam.Param<param::FloatParam>()->Value());
			float minC = 0.0f;
            float maxC = 0.0f;
			unsigned int colTabSize = 0;
			glUniform4f(this->splineShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));

			UINT64 numVerts;
            UINT64 vertCounter;
			numVerts = this->bufSize / vertStride;
			const char *currVert = (const char *)(this->positionsCa[i].PeekElements());
			const char *currCol = 0;
			vertCounter = 0;
			while (vertCounter < this->positionsCa[i].Count() / 4) {
				void *mem = static_cast<char*>(this->theSingleMappedMem) + this->bufSize * this->currBuf;
				const char *whence = currVert;
				UINT64 vertsThisTime = vislib::math::Min(this->positionsCa[i].Count() / 4 - vertCounter, numVerts);
				this->WaitSignal(this->fences[this->currBuf]);
				memcpy(mem, whence, (size_t)vertsThisTime * vertStride);
				glFlushMappedNamedBufferRangeEXT(theSingleBuffer, this->bufSize * this->currBuf, (GLsizeiptr)vertsThisTime * vertStride);
				glUniform1i(this->splineShader.ParameterLocation("instanceOffset"), 0);

				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, this->bufSize * this->currBuf, this->bufSize);
				glPatchParameteri(GL_PATCH_VERTICES, 1);
				glDrawArrays(GL_PATCHES, 0, (GLsizei)vertsThisTime - 3);
				this->QueueSignal(this->fences[this->currBuf]);

				this->currBuf = (this->currBuf + 1) % this->numBuffers;
				vertCounter += vertsThisTime;
				currVert += vertsThisTime * vertStride;
				currCol += vertsThisTime * colStride;
			}

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			glDisableClientState(GL_COLOR_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisable(GL_TEXTURE_1D);
			this->splineShader.Disable();
		}
	}

    // DEBUG
    // Render atoms as spheres
	if (this->sphereParam.Param<param::BoolParam>()->Value())
	{
		glEnable(GL_BLEND);
		glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
		// enable sphere shader
		this->sphereShader.Enable();
		// set shader variables
		glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
		glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"),    1, cr->GetCameraParameters()->Front().PeekComponents());
		glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
		glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"),    1, cr->GetCameraParameters()->Up().PeekComponents());
		// set vertex and color pointers and draw them
		glBegin(GL_POINTS);
		// Ca atoms
		for (unsigned int i = 0; i < this->positionsCa.Count(); i++) {
			for (unsigned int j = 0; j < this->positionsCa[i].Count() / 4; j++) {
				glColor4f(0.75f, 0.5f, 0.1f, 1.0f);
				glVertex4f(this->positionsCa[i][4 * j], this->positionsCa[i][4 * j + 1], this->positionsCa[i][4 * j + 2], 0.3f);
			}
		}
		// O atoms
		for (unsigned int i = 0; i < this->positionsO.Count(); i++) {
			for (unsigned int j = 0; j < this->positionsO[i].Count() / 4; j++) {
				glColor4f(0.75f, 0.1f, 0.1f, 1.0f);
				glVertex4f(this->positionsO[i][4 * j], this->positionsO[i][4 * j + 1], this->positionsO[i][4 * j + 2], 0.3f);
			}
		}
		// all atoms
		for (unsigned int i = 0; i < mol->AtomCount(); i++) {
			unsigned int atomTypeIdx = mol->AtomTypeIndices()[i];
			if (mol->AtomTypes()[atomTypeIdx].Name().Equals("CA")) {
				glColor4f(0.5f, 0.75f, 0.1f, 0.5f);
				glVertex4f(mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2], 1.0f);
			}
			else {
				glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
				glVertex4f(mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2], 0.5f);
			}

		}
		glEnd();
		// disable sphere shader
		this->sphereShader.Disable();
	}

    mol->Unlock();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
    
#ifdef DEBUG_GL
    glDisable(GL_DEBUG_OUTPUT);
    glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

//	timer.EndFrame();

    return true;
}


/*
 * UncertaintyCartoonRenderer::GetBytesAndStride
 */
void UncertaintyCartoonRenderer::GetBytesAndStride(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes, 
                                                   unsigned int &colStride, unsigned int &vertStride) {
    
    vertBytes = 0; 
    colBytes = 0;
    //colBytes = vislib::math::Max(colBytes, 3 * 4U);
    vertBytes = vislib::math::Max(vertBytes, (unsigned int)sizeof(CAlpha));

    colStride = 0;
    colStride = (colStride < colBytes) ? (colBytes) : (colStride);
    vertStride = 0;
    vertStride = (vertStride < vertBytes) ? (vertBytes) : (vertStride);
}


/*
 * UncertaintyCartoonRenderer::GetBytesAndStrideLines 
 */
void UncertaintyCartoonRenderer::GetBytesAndStrideLines(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes,
                                                        unsigned int &colStride, unsigned int &vertStride) {
	vertBytes = 0; 
    colBytes = 0;
	//colBytes = vislib::math::Max(colBytes, 3 * 4U);
	vertBytes = vislib::math::Max(vertBytes, 4 * 4U);

	colStride  = 0;
	colStride  = (colStride < colBytes) ? (colBytes) : (colStride);
	vertStride = 0;
	vertStride = (vertStride < vertBytes) ? (vertBytes) : (vertStride);
}


/*
* UncertaintyCartoonRenderer::QueueSignal
*/
void UncertaintyCartoonRenderer::QueueSignal(GLsync& syncObj) {
    
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}


/*
* UncertaintyCartoonRenderer::WaitSignal
*/
void UncertaintyCartoonRenderer::WaitSignal(GLsync& syncObj) {
    
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}
