/*
 * CartoonTessellationRenderer.cpp
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CartoonTessellationRenderer.h"
#include "protein_calls/MolecularDataCall.h"
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
#include <inttypes.h>
#include <stdint.h>

using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

#define RENDER_ATOMS_AS_SPHERES 1

#define MAP_BUFFER_LOCALLY
#define DEBUG_BLAHBLAH

const GLuint SSBObindingPoint = 2;
//#define NGS_THE_INSTANCE "gl_InstanceID"
#define NGS_THE_INSTANCE "gl_VertexID"

//typedef void (APIENTRY *GLDEBUGPROC)(GLenum source,GLenum type,GLuint id,GLenum severity,GLsizei length,const GLchar *message,const void *userParam);
void APIENTRY MyFunkyDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
    const GLchar* message, const GLvoid* userParam) {
        const char *sourceText, *typeText, *severityText;
        switch(source) {
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
        switch(type) {
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
        switch(severity) {
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
 * moldyn::CartoonTessellationRenderer::CartoonTessellationRenderer
 */
CartoonTessellationRenderer::CartoonTessellationRenderer(void) : Renderer3DModule(),
    getDataSlot("getdata", "Connects to the data source"),
    fences(), currBuf(0), bufSize(32 * 1024 * 1024), numBuffers(3),
    scalingParam("scaling", "scaling factor for particle radii"),
	sphereParam("spheres", "render atoms as spheres"),
	lineParam("lines", "render backbone as GL_LINE"),
	backboneParam("backbone", "render backbone as tubes"),
	backboneWidthParam("backbone width", "the width of the backbone"),
	materialParam("material", "ambient, diffuse, specular components + exponent"),
	lineDebugParam("wireframe", "render in wireframe mode"),
	buttonParam("reload shaders", "reload the shaders"),
	colorInterpolationParam("color interpolation", "should the colors be interpolated?"),
    // this variant should not need the fence
    singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT),
    singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT) {

    this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

	this->sphereParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->sphereParam);

	this->lineParam << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->lineParam);

	this->backboneParam << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->backboneParam);

    this->scalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingParam);
    
	this->backboneWidthParam << new core::param::FloatParam(0.25f);
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
	
	fences.resize(numBuffers);

#ifdef FIRSTFRAME_CHECK
	firstFrame = true;
#endif
}


/*
 * moldyn::CartoonTessellationRenderer::~CartoonTessellationRenderer
 */
CartoonTessellationRenderer::~CartoonTessellationRenderer(void) {
    this->Release();
}

void CartoonTessellationRenderer::queueSignal(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void CartoonTessellationRenderer::waitSignal(GLsync& syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool CartoonTessellationRenderer::create(void) {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;
#ifdef DEBUG_BLAHBLAH
    glDebugMessageCallback(MyFunkyDebugCallback, NULL);
#endif
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
        return false;
    if (!vislib::graphics::gl::GLSLTesselationShader::InitialiseExtensions())
        return false;

    //vislib::graphics::gl::ShaderSource vert, frag;
    vert = new ShaderSource();
    tessCont = new ShaderSource();
    tessEval = new ShaderSource();
    geom = new ShaderSource();
    frag = new ShaderSource();
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::vertex", *vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::tesscontrol", *tessCont)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::tesseval", *tessEval)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::geometry", *geom)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellation::fragment", *frag)) {
        return false;
    }
    try {
        // compile the shader
        if (!this->splineShader.Compile(vert->Code(), vert->Count(),
            tessCont->Code(), tessCont->Count(),
            tessEval->Code(), tessEval->Count(),
            geom->Code(), geom->Count(),
            frag->Code(), frag->Count())) {
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
	tubeVert = new ShaderSource();
	tubeTessCont = new ShaderSource();
	tubeTessEval = new ShaderSource();
	tubeGeom = new ShaderSource();
	tubeFrag = new ShaderSource();
	if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::vertex", *tubeVert)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::tesscontrol", *tubeTessCont)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::tesseval", *tubeTessEval)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::geometry", *tubeGeom)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::fragment", *tubeFrag)) {
		return false;
	}

	try {
		// compile the shader
		if (!this->tubeShader.Compile(tubeVert->Code(), tubeVert->Count(),
			tubeTessCont->Code(), tubeTessCont->Count(),
			tubeTessEval->Code(), tubeTessEval->Count(),
			tubeGeom->Code(), tubeGeom->Count(),
			tubeFrag->Code(), tubeFrag->Count())) {
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

    // Load sphere shader
    ShaderSource vertSrc;
    ShaderSource fragSrc;
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
        "protein::std::sphereVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource(
        "protein::std::sphereFragment", fragSrc)) {
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
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, this->bufSize * this->numBuffers, nullptr, singleBufferCreationBits);
    this->theSingleMappedMem = glMapNamedBufferRangeEXT(this->theSingleBuffer, 0, this->bufSize * this->numBuffers, singleBufferMappingBits);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);

    return true;
}

/*
 * moldyn::SimpleSphereRenderer::release
 */
void CartoonTessellationRenderer::release(void) {
    glUnmapNamedBufferEXT(this->theSingleBuffer);
    for (auto &x : fences) {
        if (x) {
            glDeleteSync(x);
        }
    }
    //this->sphereShader.Release();
    // TODO release all shaders!
    glDeleteVertexArrays(1, &this->vertArray);
    glDeleteBuffers(1, &this->theSingleBuffer);
}


void CartoonTessellationRenderer::getBytesAndStride(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes,
    unsigned int &colStride, unsigned int &vertStride) {
    vertBytes = 0; colBytes = 0;
    //colBytes = vislib::math::Max(colBytes, 3 * 4U);
    vertBytes = vislib::math::Max(vertBytes, (unsigned int)sizeof(CAlpha));

    colStride = 0;
    colStride = colStride < colBytes ? colBytes : colStride;
    vertStride = 0;
    vertStride = vertStride < vertBytes ? vertBytes : vertStride;
}

void CartoonTessellationRenderer::getBytesAndStrideLines(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes,
	unsigned int &colStride, unsigned int &vertStride) {
	vertBytes = 0; colBytes = 0;
	//colBytes = vislib::math::Max(colBytes, 3 * 4U);
	vertBytes = vislib::math::Max(vertBytes, 4 * 4U);

	colStride = 0;
	colStride = colStride < colBytes ? colBytes : colStride;
	vertStride = 0;
	vertStride = vertStride < vertBytes ? vertBytes : vertStride;
}


/*
* GetExtents
*/
bool CartoonTessellationRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MolecularDataCall *mol = this->getDataSlot.CallAs<MolecularDataCall>();
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
 *  getData
 */
MolecularDataCall* CartoonTessellationRenderer::getData(unsigned int t, float& outScaling) {
    MolecularDataCall *mol = this->getDataSlot.CallAs<MolecularDataCall>();
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
 * moldyn::SimpleSphereRenderer::Render
 */
bool CartoonTessellationRenderer::Render(Call& call) {
#ifdef DEBUG_BLAHBLAH
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    float scaling = 1.0f;
    MolecularDataCall *mol = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (mol == NULL) return false;

	if (this->buttonParam.IsDirty()) {
		this->buttonParam.ResetDirty();

		instance()->ShaderSourceFactory().LoadBTF("cartoontessellationnew", true);

		// load tube shader
		tubeVert = new ShaderSource();
		tubeTessCont = new ShaderSource();
		tubeTessEval = new ShaderSource();
		tubeGeom = new ShaderSource();
		tubeFrag = new ShaderSource();
		if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::vertex", *tubeVert)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::tesscontrol", *tubeTessCont)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::tesseval", *tubeTessEval)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::geometry", *tubeGeom)) {
			return false;
		}
		if (!instance()->ShaderSourceFactory().MakeShaderSource("cartoontessellationnew::fragment", *tubeFrag)) {
			return false;
		}

		try {
			// compile the shader
			if (!this->tubeShader.Compile(tubeVert->Code(), tubeVert->Count(),
				tubeTessCont->Code(), tubeTessCont->Count(),
				tubeTessEval->Code(), tubeTessEval->Count(),
				tubeGeom->Code(), tubeGeom->Count(),
				tubeFrag->Code(), tubeFrag->Count())) {
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

//	timer.BeginFrame();

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
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, theSingleBuffer);
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


	/*std::cout << lightAmbient[0] << " " << lightAmbient[1] << " " << lightAmbient[2] << " " << lightAmbient[3] << std::endl;
	std::cout << lightDiffuse[0] << " " << lightDiffuse[1] << " " << lightDiffuse[2] << " " << lightDiffuse[3] << std::endl;
	std::cout << lightSpecular[0] << " " << lightSpecular[1] << " " << lightSpecular[2] << " " << lightSpecular[3] << std::endl;*/

    // copy data
    if (this->positionsCa.Count() != mol->MoleculeCount()) {
        this->positionsCa.SetCount(mol->MoleculeCount());
        this->positionsO.SetCount(mol->MoleculeCount());
    }
    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int firstAtomIdx = 0;
    unsigned int lastAtomIdx = 0;
    unsigned int atomTypeIdx = 0;
	unsigned int firstSecIdx = 0;
	unsigned int lastSecIdx = 0;
	unsigned int firstAAIdx = 0;
	unsigned int lastAAIdx = 0;

	unsigned int cIndex = 0;
	unsigned int oIndex = 0;

	mainchain.clear();

	CAlpha lastCalpha;

	int molCount = mol->MoleculeCount();
	std::vector<int> molSizes;

	// loop over all molecules of the protein
	for (unsigned int molIdx = 0; molIdx < mol->MoleculeCount(); molIdx++) {

		MolecularDataCall::Molecule chain = mol->Molecules()[molIdx];
		molSizes.push_back(0);

		bool firstset = false;

		// is the first residue an aminoacid?
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

				calpha.dir[0] = mol->AtomPositions()[3 * acid->OIndex()] - calpha.pos[0];
				calpha.dir[1] = mol->AtomPositions()[3 * acid->OIndex() + 1] - calpha.pos[1];
				calpha.dir[2] = mol->AtomPositions()[3 * acid->OIndex() + 2] - calpha.pos[2];

				auto type = mol->SecondaryStructures()[secIdx].Type();
				calpha.type = (int)type;
				molSizes[molIdx]++;

				// TODO do this on GPU?
				// orientation check for the direction
				if (mainchain.size() != 0)
				{
					CAlpha before = mainchain[mainchain.size() - 1];
					float dotProd = calpha.dir[0] * before.dir[0] + calpha.dir[1] * before.dir[1] + calpha.dir[2] * before.dir[2];

					if (dotProd < 0) // flip direction if the orientation is wrong
					{
						calpha.dir[0] = -calpha.dir[0];
						calpha.dir[1] = -calpha.dir[1];
						calpha.dir[2] = -calpha.dir[2];
					}
				}

				mainchain.push_back(calpha);

				lastCalpha = calpha;

				// add the first atom 3 times
				if (!firstset) {
					mainchain.push_back(calpha);
					mainchain.push_back(calpha);
					molSizes[molIdx] += 2;
					firstset = true;
				}
			}
		}

		// add the last atom 3 times
		mainchain.push_back(lastCalpha);
		mainchain.push_back(lastCalpha);
		molSizes[molIdx] += 2;
	}
	
#ifdef FIRSTFRAME_CHECK
	if (firstFrame) {
		for (int i = 0; i < mainchain.size(); i++) {
			std::cout << mainchain[i].type << std::endl;
		}
		firstFrame = false;
	}
#endif

	firstResIdx = 0;
	lastResIdx = 0;
	firstAtomIdx = 0;
	lastAtomIdx = 0;
	atomTypeIdx = 0;

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
	//std::cout << "cIndex " << cIndex << " oIndex " << oIndex << " molCount " << mol->MoleculeCount() << std::endl;

#if 1
	if (lineParam.Param<param::BoolParam>()->Value())
	{
		//currBuf = 0;
		for (unsigned int i = 0; i < this->positionsCa.Count(); i++) {
			unsigned int colBytes, vertBytes, colStride, vertStride;
			this->getBytesAndStrideLines(*mol, colBytes, vertBytes, colStride, vertStride);

			this->splineShader.Enable();
			glColor4f(1.0f / this->positionsCa.Count() * (i + 1), 0.75f, 0.25f, 1.0f);
			colIdxAttribLoc = glGetAttribLocationARB(this->splineShader, "colIdx");
			glUniform4fv(this->splineShader.ParameterLocation("viewAttr"), 1, viewportStuff);
			glUniform3fv(this->splineShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
			glUniform3fv(this->splineShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
			glUniform3fv(this->splineShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
			glUniform4fv(this->splineShader.ParameterLocation("clipDat"), 1, clipDat);
			glUniform4fv(this->splineShader.ParameterLocation("clipCol"), 1, clipCol);
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
			glUniformMatrix4fv(this->splineShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
			glUniform1f(this->splineShader.ParameterLocation("scaling"), this->scalingParam.Param<param::FloatParam>()->Value());
			float minC = 0.0f, maxC = 0.0f;
			unsigned int colTabSize = 0;
			glUniform4f(this->splineShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));

			UINT64 numVerts, vertCounter;
			numVerts = this->bufSize / vertStride;
			const char *currVert = (const char *)(this->positionsCa[i].PeekElements());
			const char *currCol = 0;
			vertCounter = 0;
			while (vertCounter < this->positionsCa[i].Count() / 4) {
				void *mem = static_cast<char*>(this->theSingleMappedMem) + bufSize * currBuf;
				const char *whence = currVert;
				UINT64 vertsThisTime = vislib::math::Min(this->positionsCa[i].Count() / 4 - vertCounter, numVerts);
				this->waitSignal(fences[currBuf]);
				memcpy(mem, whence, (size_t)vertsThisTime * vertStride);
				glFlushMappedNamedBufferRangeEXT(theSingleBuffer, bufSize * currBuf, (GLsizeiptr)vertsThisTime * vertStride);
				glUniform1i(this->splineShader.ParameterLocation("instanceOffset"), 0);

				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, bufSize * currBuf, bufSize);
				glPatchParameteri(GL_PATCH_VERTICES, 1);
				glDrawArrays(GL_PATCHES, 0, (GLsizei)vertsThisTime - 3);
				this->queueSignal(fences[currBuf]);

				currBuf = (currBuf + 1) % this->numBuffers;
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
#endif

#if 1
	if (this->lineDebugParam.Param<param::BoolParam>()->Value())
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (backboneParam.Param<param::BoolParam>()->Value()) {
		//currBuf = 0;
		unsigned int colBytes, vertBytes, colStride, vertStride;
		this->getBytesAndStride(*mol, colBytes, vertBytes, colStride, vertStride);

		this->tubeShader.Enable();
		glColor4f(1.0f / mainchain.size(), 0.75f, 0.25f, 1.0f);
		colIdxAttribLoc = glGetAttribLocationARB(this->splineShader, "colIdx");
		glUniform4fv(this->tubeShader.ParameterLocation("viewAttr"), 1, viewportStuff);
		glUniform3fv(this->tubeShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
		glUniform3fv(this->tubeShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
		glUniform3fv(this->tubeShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
		glUniform4fv(this->tubeShader.ParameterLocation("clipDat"), 1, clipDat);
		glUniform4fv(this->tubeShader.ParameterLocation("clipCol"), 1, clipCol);
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MV"), 1, GL_FALSE, modelViewMatrix.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVinvtrans"), 1, GL_FALSE, modelViewMatrixInvTrans.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("ProjInv"), 1, GL_FALSE, projectionMatrixInv.PeekComponents());
		glUniform1f(this->tubeShader.ParameterLocation("scaling"), this->scalingParam.Param<param::FloatParam>()->Value());
		glUniform1f(this->tubeShader.ParameterLocation("pipeWidth"), this->backboneWidthParam.Param<param::FloatParam>()->Value());
		glUniform1i(this->tubeShader.ParameterLocation("interpolateColors"), this->colorInterpolationParam.Param<param::BoolParam>()->Value());
		float minC = 0.0f, maxC = 0.0f;
		unsigned int colTabSize = 0;
		glUniform4f(this->tubeShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
		//auto v = this->materialParam.Param<param::Vector4fParam>()->Value();
		glUniform4f(this->tubeShader.ParameterLocation("ambientColor"), lightAmbient[0], lightAmbient[1], lightAmbient[2], lightAmbient[3]);
		glUniform4f(this->tubeShader.ParameterLocation("diffuseColor"), lightDiffuse[0], lightDiffuse[1], lightDiffuse[2], lightDiffuse[3]);
		glUniform4f(this->tubeShader.ParameterLocation("lightPos"), lightPos[0], lightPos[1], lightPos[2], lightPos[3]);

		UINT64 numVerts;
		numVerts = this->bufSize / vertStride;
		UINT64 stride = 0;

		for (int i = 0; i < (int)molSizes.size(); i++) {
			UINT64 vertCounter = 0;
			while (vertCounter < molSizes[i]) {
				const char *currVert = (const char *)(&mainchain[(unsigned int)vertCounter + (unsigned int)stride]);
				void *mem = static_cast<char*>(this->theSingleMappedMem) + bufSize * currBuf;
				const char *whence = currVert;
				UINT64 vertsThisTime = vislib::math::Min(molSizes[i] - vertCounter, numVerts);
				this->waitSignal(fences[currBuf]);
				memcpy(mem, whence, (size_t)vertsThisTime * vertStride);
				glFlushMappedNamedBufferRangeEXT(theSingleBuffer, bufSize * currBuf, (GLsizeiptr)vertsThisTime * vertStride);
				glUniform1i(this->tubeShader.ParameterLocation("instanceOffset"), 0);

				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, this->theSingleBuffer, bufSize * currBuf, bufSize);
				glPatchParameteri(GL_PATCH_VERTICES, 1);
				glDrawArrays(GL_PATCHES, 0, (GLsizei)(vertsThisTime - 3));
				this->queueSignal(fences[currBuf]);

				currBuf = (currBuf + 1) % this->numBuffers;
				vertCounter += vertsThisTime;
				currVert += vertsThisTime * vertStride;
			}
			stride += molSizes[i];
		}

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisable(GL_TEXTURE_1D);
		this->tubeShader.Disable();
	}
#endif

    // DEBUGGING CODE
#if RENDER_ATOMS_AS_SPHERES
	if (this->sphereParam.Param<param::BoolParam>()->Value())
	{
		glEnable(GL_BLEND);
		glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
		// enable sphere shader
		this->sphereShader.Enable();
		// set shader variables
		glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
		glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
		glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
		glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
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
#endif

    mol->Unlock();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
#ifdef DEBUG_BLAHBLAH
    glDisable(GL_DEBUG_OUTPUT);
    glDisable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif

//	timer.EndFrame();

    return true;
}
