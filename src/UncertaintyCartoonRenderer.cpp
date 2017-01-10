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
//    - 
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
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"

#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"

#include "UncertaintyColor.h" 

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
		backboneParam(          "03 Backbone", "Render backbone as tubes."),
		backboneWidthParam(     "04 Backbone width", "The width of the backbone."),
		tessLevelParam(         "05 Tesselation level", "The tesselation level."),
		lineDebugParam(         "06 Wireframe", "Render in wireframe mode."),
		onlyTubesParam(         "07 Only tubes", "Render only tubes."),
		uncVisParam(            "08 Uncertainty visualisation", "The uncertainty visualisation."),
		uncDistorParam(         "09 Uncertainty distortion", "(0) amplification, (1) repeat of sin(2*PI), (2) apply uncertainty all over (true if value = 0.0), (3) unused."),
        ditherParam(            "10 Dithering", "enable and add additional dithering passes, dithering is disabled for 0."),
		uncertainMaterialParam( "11 Uncertain material", "material properties for uncertain structure assignment: Ambient, diffuse, specular components + exponent"),
		materialParam(          "12 Material", "Ambient, diffuse, specular components + exponent."),
		colorModeParam(         "13 Color mode", "Coloring mode for secondary structure."),
		colorInterpolationParam("14 Color interpolation", "Should the colors be interpolated?"),
		lightPosParam(          "15 Light position", "The light position."),
		buttonParam(            "16 Reload shaders", "Reload the shaders."),
		colorTableFileParam(    "17 Color Table Filename", "The filename of the color table."),
		fences(), currBuf(0), bufSize(32 * 1024 * 1024), numBuffers(3), aminoAcidCount(0), resSelectionCall(NULL), molAtomCount(0),
        // this variant should not need the fence
        singleBufferCreationBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT),
        singleBufferMappingBits(GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT) 
        {

	// number of different secondary structure types
	this->structCount = static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE);
    
	// uncertainty data caller slot
	this->uncertaintyDataSlot.SetCompatibleCall<UncertaintyDataCallDescription>();
	this->MakeSlotAvailable(&this->uncertaintyDataSlot);

	// pdb data caller slot
	this->getPdbDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->getPdbDataSlot);

	// residue selection caller slot
	this->resSelectionCallerSlot.SetCompatibleCall<ResidueSelectionCallDescription>();
	this->MakeSlotAvailable(&this->resSelectionCallerSlot);


	this->onlyTubesParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->onlyTubesParam);

	this->sphereParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->sphereParam);

	this->backboneParam << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->backboneParam);

	this->lineDebugParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->lineDebugParam);

	this->colorInterpolationParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->colorInterpolationParam);

	this->buttonParam << new core::param::ButtonParam(vislib::sys::KeyCode::KEY_F5);
	this->MakeSlotAvailable(&this->buttonParam);
                                              // init min max
	this->ditherParam << new core::param::IntParam(0, 0, this->structCount);
	this->MakeSlotAvailable(&this->ditherParam);
	                                              // init min max
	this->tessLevelParam << new core::param::IntParam(16, 1, 64);
	this->MakeSlotAvailable(&this->tessLevelParam);

	this->scalingParam << new core::param::FloatParam(1.0f);
	this->MakeSlotAvailable(&this->scalingParam);

	this->backboneWidthParam << new core::param::FloatParam(0.2f);
	this->MakeSlotAvailable(&this->backboneWidthParam);

	vislib::math::Vector<float, 4> tmpVec4(0.2f, 0.8f, 0.4f, 10.0f);
	this->materialParam << new core::param::Vector4fParam(tmpVec4);
	this->MakeSlotAvailable(&this->materialParam);

	this->uncertainMaterialParam << new core::param::Vector4fParam(tmpVec4);
	this->MakeSlotAvailable(&this->uncertainMaterialParam);

	tmpVec4 = vislib::math::Vector<float, 4>(0.0f, 0.0f, 1.0f, 0.0f);
	this->lightPosParam << new core::param::Vector4fParam(tmpVec4);
	this->MakeSlotAvailable(&this->lightPosParam);

	tmpVec4 = vislib::math::Vector<float, 4>(1.0f, 5.0f, 0.0f, 0.0f);
	this->uncDistorParam << new core::param::Vector4fParam(tmpVec4);
	this->MakeSlotAvailable(&this->uncDistorParam);

	param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(COLOR_MODE_STRUCT));
	tmpEnum->SetTypePair(COLOR_MODE_STRUCT,        "Secondary Structure");
	tmpEnum->SetTypePair(COLOR_MODE_UNCERTAIN,     "Uncertainty");
	tmpEnum->SetTypePair(COLOR_MODE_CHAIN,         "Chains");
	tmpEnum->SetTypePair(COLOR_MODE_AMINOACID,     "Aminoacids");
	tmpEnum->SetTypePair(COLOR_MODE_RESIDUE_DEBUG, "DEBUG residues");
	this->colorModeParam << tmpEnum;
	this->MakeSlotAvailable(&this->colorModeParam);

	tmpEnum = new param::EnumParam(static_cast<int>(UNC_VIS_SIN_UV));
	tmpEnum->SetTypePair(UNC_VIS_NONE,   "None");
	tmpEnum->SetTypePair(UNC_VIS_SIN_U,  "Sinus U");
	tmpEnum->SetTypePair(UNC_VIS_SIN_V,  "Sinus V");
	tmpEnum->SetTypePair(UNC_VIS_SIN_UV, "Sinus UV");
	this->uncVisParam << tmpEnum;
	this->MakeSlotAvailable(&this->uncVisParam);

	// fill color table with default values and set the filename param
	vislib::StringA filename("colors.txt");
	this->colorTableFileParam.SetParameter(new param::FilePathParam(A2T(filename)));
	this->MakeSlotAvailable(&this->colorTableFileParam);
	UncertaintyColor::ReadColorTableFromFile(T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorTable);

	this->currentTessLevel         = 16;
	this->currentUncVis            = uncVisualisations::UNC_VIS_SIN_U;
	this->currentColoringMode      = coloringModes::COLOR_MODE_STRUCT;
	this->currentScaling           = 1.0f;
	this->currentBackboneWidth     = 0.2f;
	this->currentMaterial          = vislib::math::Vector<float, 4>(0.4f, 0.8f, 0.6f, 10.0f);
	this->currentUncertainMaterial = vislib::math::Vector<float, 4>(0.4f, 0.8f, 0.6f, 10.0f);
	this->currentColoringMode      = coloringModes::COLOR_MODE_STRUCT;
	this->currentUncVis            = uncVisualisations::UNC_VIS_SIN_UV;
	this->currentLightPos          = vislib::math::Vector<float, 4>(0.0f, 0.0f, 1.0f, 0.0f);
	this->currentUncDist           = vislib::math::Vector<float, 4>(1.0f, 5.0f, 0.0f, 0.0f);
    this->currentDitherMode        = 0;
    
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
* UncertaintyCartoonRenderer::loadTubeShader
*/
bool UncertaintyCartoonRenderer::loadTubeShader(void) {

	if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions())
		return false;
	if (!vislib::graphics::gl::GLSLTesselationShader::InitialiseExtensions())
		return false;

	instance()->ShaderSourceFactory().LoadBTF("uncertaintycartoontessellation", true);

	// load tube shader
	this->tubeVert = new ShaderSource();
	this->tubeTessCont = new ShaderSource();
	this->tubeTessEval = new ShaderSource();
	this->tubeGeom = new ShaderSource();
	this->tubeFrag = new ShaderSource();
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
		if (!this->tubeShader.Compile(this->tubeVert->Code(), this->tubeVert->Count(),
			this->tubeTessCont->Code(), this->tubeTessCont->Count(),
			this->tubeTessEval->Code(), this->tubeTessEval->Count(),
			this->tubeGeom->Code(), this->tubeGeom->Count(),
			this->tubeFrag->Code(), this->tubeFrag->Count())) {
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

	return true;
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

	// load tube shader
	if (!this->loadTubeShader()) {
		return false;
	}

    // load sphere shader
	ShaderSource vertSrc; 
	ShaderSource fragSrc; 

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
bool UncertaintyCartoonRenderer::GetUncertaintyData(UncertaintyDataCall *udc, MolecularDataCall *mol) {

	if (!udc) return false;
	if (!mol) return false;

    // execute the call
    if( !(*udc)(UncertaintyDataCall::CallForGetData)) return false;
    
	// check molecular data
	// if (!(*mol)(MolecularDataCall::CallForGetData)) return false; // don't call twice ... ?

	// initialization
	this->aminoAcidCount = udc->GetAminoAcidCount();
	this->molAtomCount = mol->AtomCount();

	// reset arrays
	this->secStructColorRGB.Clear();
	this->secStructColorRGB.AssertCapacity(UncertaintyDataCall::secStructure::NOE);

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

	this->diffUncertainty.Clear();
	this->diffUncertainty.AssertCapacity(this->aminoAcidCount);

	this->pdbIndex.Clear();
	this->pdbIndex.AssertCapacity(this->aminoAcidCount);

	this->chainColors.Clear();
	this->chainColors.AssertCapacity(this->aminoAcidCount);

	this->aminoAcidColors.Clear();
	this->aminoAcidColors.AssertCapacity(this->aminoAcidCount);

	// get secondary structure type colors 
	for (unsigned int i = 0; i < static_cast<unsigned int>(UncertaintyDataCall::secStructure::NOE); i++) {
		this->secStructColorRGB.Add(udc->GetSecStructColor(static_cast<UncertaintyDataCall::secStructure>(i)));
	}

	unsigned int cCnt = 0;
	char currentChainID = udc->GetChainID(0);

	// collect data from call
	for (unsigned int aa = 0; aa < this->aminoAcidCount; aa++) {

		// store the secondary structure element type of the current amino-acid for each assignment method
		for (unsigned int k = 0; k < static_cast<unsigned int>(UncertaintyDataCall::assMethod::NOM); k++) {
			this->secStructAssignment[k].Add(udc->GetSecStructure(static_cast<UncertaintyDataCall::assMethod>(k), aa));
		}

		// store residue flag
		this->residueFlag.Add(static_cast<unsigned int>(udc->GetResidueFlag(aa)));
		// store uncerteiny values
		this->secUncertainty.Add(udc->GetSecStructUncertainty(aa));
		// store sorted uncertainty structure types
		this->sortedUncertainty.Add(udc->GetSortedSecStructureIndices(aa));
		// store the uncertainty difference
		this->diffUncertainty.Add(udc->GetDifference(aa));
		// store the original pdb index
		this->pdbIndex.Add(udc->GetPDBAminoAcidIndex(aa));


		// count different chains and set chain color
		if (udc->GetChainID(aa) != currentChainID) {
			currentChainID = udc->GetChainID(aa);
			cCnt++;
		}
		this->chainColors.Add(this->colorTable[cCnt]);
		
		// set colors for amino acids
		
		char tmpAA = udc->GetAminoAcidOneLetterCode(aa);
		if (tmpAA == '?') {
			this->aminoAcidColors.Add(this->colorTable[this->colorTable.Count()-1]);
		}
		else {
			this->aminoAcidColors.Add(this->colorTable[((unsigned int)tmpAA - 65)]);
		}
	}


	// Synchronize data array index from MoleculeDataCall with data array index from UncertaintyDataCall via the original pdb index
	unsigned int firstMol;
	unsigned int firstStruct;
	unsigned int uncIndex;
	unsigned int molIndex;
	unsigned int origMolIndex;

	this->synchronizedIndex.Clear();
	this->synchronizedIndex.AssertCapacity(this->molAtomCount);

	// loop over all chains of the molecular data
	for (unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++) { // all chains

		firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
		for (unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount(); mCnt++) { // molecules in chain (?)

			firstStruct = mol->Molecules()[mCnt].FirstSecStructIndex();
			for (unsigned int sCnt = 0; sCnt < mol->Molecules()[mCnt].SecStructCount(); sCnt++) { // secondary structures in chain

				for (unsigned int rCnt = 0; rCnt < mol->SecondaryStructures()[firstStruct + sCnt].AminoAcidCount(); rCnt++) { // aminoacids in secondary structures

					uncIndex = 0;
					origMolIndex = mol->Residues()[(mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt)]->OriginalResIndex();

					// go to right chain in uncertainty data
					while ((uncIndex < this->aminoAcidCount) && (mol->Chains()[cCnt].Name() != udc->GetChainID(uncIndex))) {
						uncIndex++;
					}
					// search for matching original pdb indices in both data loaders
					while (uncIndex < this->aminoAcidCount) {
						if (static_cast<std::string>(udc->GetPDBAminoAcidIndex(uncIndex).PeekBuffer()).find_first_not_of("0123456789") == std::string::npos) { // C++11 function ...
							if (std::atoi(udc->GetPDBAminoAcidIndex(uncIndex)) == origMolIndex) {
								break;
							}
						}
						uncIndex++;
					}

					// when indices in molecular data are missing, fill 'synchronizedIndex' with 'dummy' indices.
					// in the end molecular index must always match array length !
					molIndex = mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt;
					while (this->synchronizedIndex.Count() < molIndex) {
						this->synchronizedIndex.Add(0);
					}

					this->synchronizedIndex.Add(uncIndex); 

					// DEBUG
					/*
					unsigned int molIndex = (mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt);
					std::cout << " mol index: " << mol->SecondaryStructures()[firstStruct + sCnt].FirstAminoAcidIndex() + rCnt
					          << " | Chain Name: " << mol->Chains()[cCnt].Name()
							  << " | mol orig index: " << origMolIndex
							  << " | unc orig index: " << this->synchronizedIndex.Last()
							  << " | count: " << this->synchronizedIndex.Count()
							  << std::endl;
					*/
				}
			}
		}
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
    
	// if amino-acid count changed get new data
	if ((ud->GetAminoAcidCount() != this->aminoAcidCount) && (mol->AtomCount() != this->molAtomCount)) {
		this->GetUncertaintyData(ud, mol); // use return value ...?
	}


	// get dither mode
	if (this->ditherParam.IsDirty()) {
		this->ditherParam.ResetDirty();
		this->currentDitherMode = this->ditherParam.Param<param::IntParam>()->Value();
    }
	// get scaling factor
	if (this->scalingParam.IsDirty()) {
		this->scalingParam.ResetDirty();
		this->currentScaling = static_cast<float>(this->scalingParam.Param<param::FloatParam>()->Value());
	}
	// get backbone width
	if (this->backboneWidthParam.IsDirty()) {
		this->backboneWidthParam.ResetDirty();
		this->currentBackboneWidth = static_cast<float>(this->backboneWidthParam.Param<param::FloatParam>()->Value());
	}
	// get material lighting properties
	if (this->materialParam.IsDirty()) {
		this->materialParam.ResetDirty();
		this->currentMaterial = static_cast<vislib::math::Vector<float, 4>>(this->materialParam.Param<param::Vector4fParam>()->Value());
	}
	// get material lighting properties
	if (this->uncertainMaterialParam.IsDirty()) {
		this->uncertainMaterialParam.ResetDirty();
		this->currentUncertainMaterial = static_cast<vislib::math::Vector<float, 4>>(this->uncertainMaterialParam.Param<param::Vector4fParam>()->Value());
	}
	// get uncertainty distortion
	if (this->uncDistorParam.IsDirty()) {
		this->uncDistorParam.ResetDirty();
		this->currentUncDist = static_cast<vislib::math::Vector<float, 4>>(this->uncDistorParam.Param<param::Vector4fParam>()->Value());
	}
	// get uncertainty visualisation mode
	if (this->uncVisParam.IsDirty()) {
		this->uncVisParam.ResetDirty();
		this->currentUncVis = static_cast<uncVisualisations>(this->uncVisParam.Param<param::EnumParam>()->Value());
	}
	// read and update the color table, if necessary
	if (this->colorTableFileParam.IsDirty()) {
		UncertaintyColor::ReadColorTableFromFile(T2A(this->colorTableFileParam.Param<param::FilePathParam>()->Value()), this->colorTable);
		this->colorTableFileParam.ResetDirty();
	}
	// get lighting position
	if (this->lightPosParam.IsDirty()) {
		this->lightPosParam.ResetDirty();
		this->currentLightPos = static_cast<vislib::math::Vector<float, 4>>(this->lightPosParam.Param<param::Vector4fParam>()->Value());
	}
	// get coloring mode
	if (this->colorModeParam.IsDirty()) {
		this->colorModeParam.ResetDirty();
		this->currentColoringMode = static_cast<coloringModes>(this->colorModeParam.Param<param::EnumParam>()->Value());
	}
	// get new tesselation level
	if (this->tessLevelParam.IsDirty()) {
		this->tessLevelParam.ResetDirty();
		this->currentTessLevel = this->tessLevelParam.Param<param::IntParam>()->Value();
	}
    // reload shaders
	if (this->buttonParam.IsDirty()) {
		this->buttonParam.ResetDirty();
		if (!this->loadTubeShader()) {
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


	// Render in wireframe mode
	if (this->lineDebugParam.Param<param::BoolParam>()->Value())
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


	unsigned int firstResIdx = 0;
	unsigned int lastResIdx = 0;
	unsigned int firstAtomIdx = 0;
	unsigned int lastAtomIdx = 0;
	unsigned int atomTypeIdx = 0;
	unsigned int firstSecIdx = 0;
	unsigned int lastSecIdx = 0;
	unsigned int firstAAIdx = 0;
	unsigned int lastAAIdx = 0;

	unsigned int uncIndex = 0;

	// Render backbone as tubes
	if (this->backboneParam.Param<param::BoolParam>()->Value()) {

		// copy data to: mainChain
		firstSecIdx = 0;
		lastSecIdx = 0;
		firstAAIdx = 0;
		lastAAIdx = 0;

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

					// DEBUG
					/*
					std::cout << " mol index: " << aaIdx
						      << " | unc index: " << this->synchronizedIndex[aaIdx]
							  << std::endl;
					*/
					uncIndex = this->synchronizedIndex[aaIdx];
					for (unsigned int k = 0; k < this->structCount; k++) {
						// calpha.sortedStruct[k] = static_cast<int>(this->secStructAssignment[(int)UncertaintyDataCall::assMethod::DSSP][uncIndex]);
						// calpha.sortedStruct[k] = static_cast<int>(this->secStructAssignment[(int)UncertaintyDataCall::assMethod::PDB][uncIndex]);
						// calpha.sortedStruct[k] = static_cast<int>(this->secStructAssignment[(int)UncertaintyDataCall::assMethod::STRIDE][uncIndex]);
						calpha.sortedStruct[k] = static_cast<int>(this->sortedUncertainty[uncIndex][k]);

						calpha.unc[k] = this->secUncertainty[uncIndex][k];
					}
					if (this->currentColoringMode == (int)COLOR_MODE_CHAIN) {
						for (unsigned int k = 0; k < 3; k++) {
							calpha.col[k] = this->chainColors[uncIndex][k];
						}
					}
					else if (this->currentColoringMode == (int)COLOR_MODE_AMINOACID) {
						for (unsigned int k = 0; k < 3; k++) {
							calpha.col[k] = this->aminoAcidColors[uncIndex][k];
						}
					}
					else
					{
						for (unsigned int k = 0; k < 3; k++) {
							calpha.col[k] = 0.0f;
						}
					}
					calpha.flag = this->residueFlag[uncIndex];
					calpha.diff = this->diffUncertainty[uncIndex];
					
					calpha.pos[0] = mol->AtomPositions()[3 * acid->CAlphaIndex()];
					calpha.pos[1] = mol->AtomPositions()[3 * acid->CAlphaIndex() + 1];
					calpha.pos[2] = mol->AtomPositions()[3 * acid->CAlphaIndex() + 2];
					calpha.pos[3] = 1.0f;

					// direction is vector from C_alpha atom to O(xygen) atom
					calpha.dir[0] = mol->AtomPositions()[3 * acid->OIndex()] - calpha.pos[0];
					calpha.dir[1] = mol->AtomPositions()[3 * acid->OIndex() + 1] - calpha.pos[1];
					calpha.dir[2] = mol->AtomPositions()[3 * acid->OIndex() + 2] - calpha.pos[2];

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
				std::cout << this->mainChain[i].sortedStruct[0] << std::endl;
			}
			this->firstFrame = false;
		}
#endif
        
		// draw backbone as tubes
		unsigned int colBytes, vertBytes, colStride, vertStride;
		this->GetBytesAndStride(*mol, colBytes, vertBytes, colStride, vertStride);
		//this->currBuf = 0;

		this->tubeShader.Enable();
		glColor4f(1.0f / this->mainChain.size(), 0.75f, 0.25f, 1.0f);
		colIdxAttribLoc = glGetAttribLocationARB(this->tubeShader, "colIdx"); // UNUSED (?) ... 

		glUniform4fv(this->tubeShader.ParameterLocation("viewAttr"), 1, viewportStuff);

		glUniform1f(this->tubeShader.ParameterLocation("scaling"), this->currentScaling);

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
		
		// only tesselation control 
		glUniform1i(this->tubeShader.ParameterLocation("tessLevel"), this->currentTessLevel);

		// only tesselation evaluation 
		glUniform1f(this->tubeShader.ParameterLocation("pipeWidth"), this->currentBackboneWidth);
		glUniform1i(this->tubeShader.ParameterLocation("interpolateColors"), (GLint)this->colorInterpolationParam.Param<param::BoolParam>()->Value());
		glUniform4fv(this->tubeShader.ParameterLocation("structColRGB"), this->structCount, (GLfloat *)this->secStructColorRGB.PeekElements());
		glUniform1i(this->tubeShader.ParameterLocation("colorMode"), (GLint)this->currentColoringMode);
		glUniform1i(this->tubeShader.ParameterLocation("onlyTubes"), (GLint)this->onlyTubesParam.Param<param::BoolParam>()->Value());
		glUniform1i(this->tubeShader.ParameterLocation("uncVisMode"), (GLint)this->currentUncVis);
		glUniform4fv(this->tubeShader.ParameterLocation("uncDistor"), 1, (GLfloat *)this->currentUncDist.PeekComponents());
        
		// only fragment shader
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("ProjInv"), 1, GL_FALSE, projectionMatrixInv.PeekComponents());
		glUniform4fv(this->tubeShader.ParameterLocation("lightPos"), 1, (GLfloat *)this->currentLightPos.PeekComponents());
			// glUniform4f(this->tubeShader.ParameterLocation("lightPos"), lightPos[0], lightPos[1], lightPos[2], lightPos[3]);
		glUniform4f(this->tubeShader.ParameterLocation("ambientColor"), lightAmbient[0], lightAmbient[1], lightAmbient[2], lightAmbient[3]);
		glUniform4f(this->tubeShader.ParameterLocation("diffuseColor"), lightDiffuse[0], lightDiffuse[1], lightDiffuse[2], lightDiffuse[3]);
		glUniform4f(this->tubeShader.ParameterLocation("specularColor"), lightSpecular[0], lightSpecular[1], lightSpecular[2], lightSpecular[3]);
		glUniform4fv(this->tubeShader.ParameterLocation("phong"), 1, (GLfloat *)this->currentMaterial.PeekComponents());
		glUniform4fv(this->tubeShader.ParameterLocation("phongUncertain"), 1, (GLfloat *)this->currentUncertainMaterial.PeekComponents());
        
        // dither matrix
        GLfloat dithM[80] = {14,  6,  9,  3,  7,  2, 10,  6, 12, 11, 13,  5, 10,  0,  4,  1,  9,  5, 15,  8, 
                              4,  1,  5, 15,  0,  8, 14,  9,  3, 15,  7,  2,  6, 12,  3, 11, 13, 10,  0, 12, 
                             11,  7, 13, 10, 12,  4, 11,  1,  5,  0,  8,  4, 14,  9, 15,  7,  8,  2,  6,  3,
                             15,  8,  2, 14,  6,  3,  7, 13,  2, 10, 12, 11,  1, 13,  5,  0,  4, 14,  1,  9}; 
                                    
		glUniform1fv(this->tubeShader.ParameterLocation("dithM"), 80, (GLfloat *)dithM);
                                
        // DITHERING 
        for (int d = 0; d <= this->currentDitherMode; d++) {
        
			// if dithering is enabled skip first render pass for d = 0 and start with d = 1
			if ((this->currentDitherMode > 0) && (d == 0)) {
				continue;
			}

            // fragment and tesselation eval
            glUniform1i(this->tubeShader.ParameterLocation("ditherCount"), (GLint)d);                                                                  // dithering pass
            
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
        }

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisable(GL_TEXTURE_1D);
		this->tubeShader.Disable();
	}


	// copy data to: positionCA and positionO for backbone line rendering or sphere rendering
	if (this->sphereParam.Param<param::BoolParam>()->Value()) {

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

		// draw spheres
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
/*
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
*/

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
