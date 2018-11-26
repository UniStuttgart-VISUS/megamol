/*
 *	SecStructRenderer2D.cpp
 *	
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All Rights reserved.
 */

#include "stdafx.h"

#include "SecStructRenderer2D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Rectangle.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include <map>

#include "protein_calls/MolecularDataCall.h"
#include "PlaneDataCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;

const GLuint SSBOBindingPoint = 2;

/*
 *	MyFunkyDebugCallback
 */
//void APIENTRY MyFunkyDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
//	const GLchar* message, const GLvoid* userParam) {
//	const char *sourceText, *typeText, *severityText;
//	switch (source) {
//	case GL_DEBUG_SOURCE_API:
//		sourceText = "API";
//		break;
//	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
//		sourceText = "Window System";
//		break;
//	case GL_DEBUG_SOURCE_SHADER_COMPILER:
//		sourceText = "Shader Compiler";
//		break;
//	case GL_DEBUG_SOURCE_THIRD_PARTY:
//		sourceText = "Third Party";
//		break;
//	case GL_DEBUG_SOURCE_APPLICATION:
//		sourceText = "Application";
//		break;
//	case GL_DEBUG_SOURCE_OTHER:
//		sourceText = "Other";
//		break;
//	default:
//		sourceText = "Unknown";
//		break;
//	}
//	switch (type) {
//	case GL_DEBUG_TYPE_ERROR:
//		typeText = "Error";
//		break;
//	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
//		typeText = "Deprecated Behavior";
//		break;
//	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
//		typeText = "Undefined Behavior";
//		break;
//	case GL_DEBUG_TYPE_PORTABILITY:
//		typeText = "Portability";
//		break;
//	case GL_DEBUG_TYPE_PERFORMANCE:
//		typeText = "Performance";
//		break;
//	case GL_DEBUG_TYPE_OTHER:
//		typeText = "Other";
//		break;
//	case GL_DEBUG_TYPE_MARKER:
//		typeText = "Marker";
//		break;
//	default:
//		typeText = "Unknown";
//		break;
//	}
//	switch (severity) {
//	case GL_DEBUG_SEVERITY_HIGH:
//		severityText = "High";
//		break;
//	case GL_DEBUG_SEVERITY_MEDIUM:
//		severityText = "Medium";
//		break;
//	case GL_DEBUG_SEVERITY_LOW:
//		severityText = "Low";
//		break;
//	case GL_DEBUG_SEVERITY_NOTIFICATION:
//		severityText = "Notification";
//		break;
//	default:
//		severityText = "Unknown";
//		break;
//	}
//	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "[%s %s] (%s %u) %s\n", sourceText, severityText, typeText, id, message);
//}

/*
 *	SecStructRenderer2D::SecStructRenderer2D
 */
SecStructRenderer2D::SecStructRenderer2D(void) : Renderer2DModule(),
	dataInSlot("getData", "Connects the secondary structure renderer with data storage."),
	planeInSlot("getPlane", "Connect the secondary structure renderer with a plane source."),
	coilWidthParam("Dimensions::coilWidth", "Screen width of random coil."),
	structureWidthParam("Dimensions::structureWidth", "Screen width of sheets and helices."),
	showBackboneParam("Show::showBackbone", "Show the backbone of the main chain."),
	showDirectConnectionsParam("Show::showDirectConnections", "Show the direct atom-atom connections between all atoms of the main chain."),
	showAtomPositionsParam("Show::showAtomPositions", "Show the positions of the C-Alpha atoms."),
	showHydrogenBondsParam("Show::showHydrogenBonds", "Show the hydrogen bonds."),
	showTubesParam("Show::showTubes", "Show the tubes around the backbone.") {

	this->dataInSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->dataInSlot);

	this->planeInSlot.SetCompatibleCall<PlaneDataCallDescription>();
	this->MakeSlotAvailable(&this->planeInSlot);

	this->coilWidthParam.SetParameter(new param::FloatParam(0.002f, 0.0f, 0.1f));
	this->MakeSlotAvailable(&this->coilWidthParam);

	this->structureWidthParam.SetParameter(new param::FloatParam(0.01f, 0.0f, 0.1f));
	this->MakeSlotAvailable(&this->structureWidthParam);

	this->showAtomPositionsParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->showAtomPositionsParam);

	this->showBackboneParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->showBackboneParam);

	this->showDirectConnectionsParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->showDirectConnectionsParam);

	this->showHydrogenBondsParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->showHydrogenBondsParam);

	this->showTubesParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->showTubesParam);

	this->lastDataHash = 0;
	this->lastPlaneHash = 0;

	this->ssbo = 0;

	this->transformationMatrix.SetIdentity();
}

/*
 *	SecStructRenderer2D::~SecStructRenderer2D
 */
SecStructRenderer2D::~SecStructRenderer2D(void) {
	this->Release();
}

/*
 *	SecStructRenderer2D::create
 */
bool SecStructRenderer2D::create(void) {
	using namespace vislib::sys;
	using namespace vislib::graphics::gl;

	if (!GLSLShader::InitialiseExtensions()) {
		return false;
	}
	if (!GLSLTesselationShader::InitialiseExtensions()) {
		return false;
	}

	/**************************** Line Shader ********************************/
	vislib::SmartPtr<ShaderSource> vert, tessCont, tessEval, geom, frag;
	vert = new ShaderSource();
	tessCont = new ShaderSource();
	tessEval = new ShaderSource();
	geom = new ShaderSource();
	frag = new ShaderSource();
	if (!instance()->ShaderSourceFactory().MakeShaderSource("linetessellation::vertex", *vert)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("linetessellation::tesscontrol", *tessCont)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("linetessellation::tesseval", *tessEval)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("linetessellation::geometry", *geom)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("linetessellation::fragment", *frag)) {
		return false;
	}
	try {
		// compile
		if (!this->lineShader.Compile(vert->Code(), vert->Count(),
			tessCont->Code(), tessCont->Count(),
			tessEval->Code(), tessEval->Count(),
			geom->Code(), geom->Count(),
			frag->Code(), frag->Count())) {
			throw vislib::Exception("Could not compile line shader. ", __FILE__, __LINE__);
		}
		// link
		if (!this->lineShader.Link()) {
			throw vislib::Exception("Could not link line shader", __FILE__, __LINE__);
		}
	} catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile line shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
			ce.FailedAction()), ce.GetMsgA());
		return false;
	} catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile line shader: %s\n", e.GetMsgA());
		return false;
	} catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile line shader: Unknown exception\n");
		return false;
	}

	/**************************** Tube Shader ********************************/

	vert = new ShaderSource();
	tessCont = new ShaderSource();
	tessEval = new ShaderSource();
	geom = new ShaderSource();
	frag = new ShaderSource();
	if (!instance()->ShaderSourceFactory().MakeShaderSource("tubetessellation::vertex", *vert)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("tubetessellation::tesscontrol", *tessCont)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("tubetessellation::tesseval", *tessEval)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("tubetessellation::geometry", *geom)) {
		return false;
	}
	if (!instance()->ShaderSourceFactory().MakeShaderSource("tubetessellation::fragment", *frag)) {
		return false;
	}
	try {
		// compile
		if (!this->tubeShader.Compile(vert->Code(), vert->Count(),
			tessCont->Code(), tessCont->Count(),
			tessEval->Code(), tessEval->Count(),
			geom->Code(), geom->Count(),
			frag->Code(), frag->Count())) {
			throw vislib::Exception("Could not compile tube shader. ", __FILE__, __LINE__);
		}
		// link
		if (!this->tubeShader.Link()) {
			throw vislib::Exception("Could not link tube shader", __FILE__, __LINE__);
		}
	}
	catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile tube shader (@%s): %s\n",
			vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
			ce.FailedAction()), ce.GetMsgA());
		return false;
	}
	catch (vislib::Exception e) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile tube shader: %s\n", e.GetMsgA());
		return false;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile tube shader: Unknown exception\n");
		return false;
	}

	return true;
}

/*
 *	SecStructRenderer2D::release
 */
void SecStructRenderer2D::release(void) {
	glDeleteBuffers(1, &this->ssbo);
}

/*
 *	SecStructRenderer2D::GetExtents
 */
bool SecStructRenderer2D::GetExtents(view::CallRender2D& call) {

	MolecularDataCall * mdc = this->dataInSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;

	PlaneDataCall * pdc = this->planeInSlot.CallAs<PlaneDataCall>();
	if (pdc == nullptr) return false;

	if (!(*mdc)(MolecularDataCall::CallForGetExtent)) return false;
	if (!(*pdc)(PlaneDataCall::CallForGetExtent)) return false;

	if (!(*mdc)(MolecularDataCall::CallForGetData)) return false;
	if (!(*pdc)(PlaneDataCall::CallForGetData)) return false;

	// should the transformation matrix be recomputed?
	if (pdc->DataHash() != this->lastPlaneHash) {
		if (pdc->GetPlaneCnt() > 0) {
			auto plane = pdc->GetPlaneData()[0];
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Plane set to p=(%f %f %f) n=(%f %f %f)", plane.Point()[0], plane.Point()[1], plane.Point()[2],
				plane.Normal()[0], plane.Normal()[1], plane.Normal()[2]);
			this->transformationMatrix = this->rotatePlaneToXY(plane);
		}
		else {
			this->transformationMatrix.SetIdentity();
		}
		this->lastPlaneHash = pdc->DataHash();
	}

	// do we have new molecule positions? if yes, load them
	if (mdc->DataHash() != this->lastDataHash) {
		this->cAlphas.clear();
		this->molSizes.clear();
		this->cAlphaIndices.clear();

		unsigned int firstResIdx = 0;
		unsigned int lastResIdx = 0;
		unsigned int firstAtomIdx = 0;
		unsigned int lastAtomIdx = 0;
		unsigned int atomTypeIdx = 0;
		unsigned int firstSecIdx = 0;
		unsigned int lastSecIdx = 0;
		unsigned int firstAAIdx = 0;
		unsigned int lastAAIdx = 0;

		this->cAlphaMap.resize(mdc->AtomCount(), 0);

		// loop over all molecules of the protein
		for (unsigned int molIdx = 0; molIdx < mdc->MoleculeCount(); molIdx++) {
			MolecularDataCall::Molecule chain = mdc->Molecules()[molIdx];
			this->molSizes.push_back(0);

			// is the first residue an aminoacid?
			if (mdc->Residues()[chain.FirstResidueIndex()]->Identifier() != MolecularDataCall::Residue::AMINOACID) {
				continue;
			}

			firstSecIdx = chain.FirstSecStructIndex();
			lastSecIdx = firstSecIdx + chain.SecStructCount();

			// loop over all secondary structures of the molecule
			for (unsigned int secIdx = firstSecIdx; secIdx < lastSecIdx; secIdx++) {
				firstAAIdx = mdc->SecondaryStructures()[secIdx].FirstAminoAcidIndex();
				lastAAIdx = firstAAIdx + mdc->SecondaryStructures()[secIdx].AminoAcidCount();

				// loop over all aminoacids inside the secondary structure
				for (unsigned int aaIdx = firstAAIdx; aaIdx < lastAAIdx; aaIdx++) {
					MolecularDataCall::AminoAcid * acid;

					// is the current residue really an aminoacid?
					if (mdc->Residues()[aaIdx]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
						acid = (MolecularDataCall::AminoAcid*)(mdc->Residues()[aaIdx]);
					}
					else {
						continue;
					}

					CAlpha calpha;
					calpha.pos[0] = mdc->AtomPositions()[3 * acid->CAlphaIndex() + 0];
					calpha.pos[1] = mdc->AtomPositions()[3 * acid->CAlphaIndex() + 1];
					calpha.pos[2] = mdc->AtomPositions()[3 * acid->CAlphaIndex() + 2];
					calpha.type = static_cast<int>(mdc->SecondaryStructures()[secIdx].Type());
					//calpha.print();

					vislib::math::Vector<float, 4> helpVec(calpha.pos[0], calpha.pos[1], calpha.pos[2], 1.0);
					helpVec = this->transformationMatrix * helpVec;
					calpha.pos[0] = helpVec[0];
					calpha.pos[1] = helpVec[1];
					calpha.pos[2] = helpVec[2];

					this->cAlphas.push_back(calpha);
					this->cAlphaIndices.push_back(acid->CAlphaIndex());
					//calpha.print(); std::cout << std::endl;

					this->molSizes[molIdx]++;

					firstAtomIdx = acid->FirstAtomIndex();
					lastAtomIdx = firstAtomIdx + acid->AtomCount();

					for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++){
						this->cAlphaMap[atomIdx] = static_cast<unsigned int>(this->cAlphas.size() - 1);
					}

					// add the first atom 3 times
					if (secIdx == firstSecIdx && aaIdx == firstAAIdx) {
						this->cAlphas.push_back(calpha);
						this->cAlphas.push_back(calpha);
						this->cAlphaIndices.push_back(acid->CAlphaIndex());
						this->cAlphaIndices.push_back(acid->CAlphaIndex());
						this->molSizes[molIdx] += 2;
					}

					// add the last atom 3 times
					if (secIdx == lastSecIdx - 1 && aaIdx == lastAAIdx - 1) {
						this->cAlphas.push_back(calpha);
						this->cAlphas.push_back(calpha);
						this->cAlphaIndices.push_back(acid->CAlphaIndex());
						this->cAlphaIndices.push_back(acid->CAlphaIndex());
						this->molSizes[molIdx] += 2;
					}
				}
			}
		}

		this->lastDataHash = mdc->DataHash();

		// compute bounding box of the c alphas
		this->bbRect.Set(FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN);
		for (unsigned int i = 0; i < this->cAlphas.size(); i++) {
			if (this->cAlphas[i].pos[0] < this->bbRect.Left()) {
				this->bbRect.SetLeft(this->cAlphas[i].pos[0]);
			}
			if (this->cAlphas[i].pos[1] < this->bbRect.Bottom()) {
				this->bbRect.SetBottom(this->cAlphas[i].pos[1]);
			}
			if (this->cAlphas[i].pos[0] > this->bbRect.Right()) {
				this->bbRect.SetRight(this->cAlphas[i].pos[0]);
			}
			if (this->cAlphas[i].pos[1] > this->bbRect.Top()) {
				this->bbRect.SetTop(this->cAlphas[i].pos[1]);
			}
		}
		this->bbRect.EnforcePositiveSize();
		// grow the bounding box by 1.5 angstrom in each direction
		this->bbRect.SetLeft(bbRect.Left() - 1.5f);
		this->bbRect.SetBottom(bbRect.Bottom() - 1.5f);
		this->bbRect.SetRight(bbRect.Right() + 1.5f);
		this->bbRect.SetTop(bbRect.Top() + 1.5f);


		for (unsigned int i = 0; i < this->cAlphas.size(); i++) {
			this->cAlphas[i].pos[0] = (2.0f * static_cast<float>(this->bbRect.AspectRatio()) * (this->cAlphas[i].pos[0] - this->bbRect.Left()) / this->bbRect.Width()) - 1.0f * static_cast<float>(this->bbRect.AspectRatio());
			this->cAlphas[i].pos[1] = (2.0f * (this->cAlphas[i].pos[1] - this->bbRect.Bottom()) / this->bbRect.Height()) - 1.0f;
		}

		if (this->ssbo == 0) {
			glGenBuffers(1, &this->ssbo);
		}

		// load the positions into the ssbo
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOBindingPoint, this->ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, this->cAlphas.size() * sizeof(CAlpha), this->cAlphas.data(), GL_DYNAMIC_COPY);

		/*glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);
		CAlpha* ptr;
		ptr = (CAlpha*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		for (int i = 0; i < this->cAlphas.size(); i++) {
			ptr[i].print();
		}
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);*/
	}

	float ar = static_cast<float>(this->bbRect.AspectRatio());

	call.SetBoundingBox(vislib::math::Rectangle<float>(-1.0f * ar, -1.0f, 1.0f * ar, 1.0f));

	return true;
}

/*
 *	SecStructRenderer2D::MouseEvents
 */
bool SecStructRenderer2D::MouseEvent(float x, float y, view::MouseFlags flags) {

	// TODO

	return false;
}

/*
 *	SecStructRenderer2D::Render
 */
bool SecStructRenderer2D::Render(view::CallRender2D& call) {

	MolecularDataCall * mdc = this->dataInSlot.CallAs<MolecularDataCall>();
	if (mdc == nullptr) return false;
	
	PlaneDataCall * pdc = this->planeInSlot.CallAs<PlaneDataCall>();
	if (pdc == nullptr) return false;

	glDisable(GL_BLEND);

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

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBOBindingPoint, this->ssbo);

	

	if (this->showBackboneParam.Param<param::BoolParam>()->Value()) {
		this->lineShader.Enable();
		glUniformMatrix4fv(this->lineShader.ParameterLocation("MV"), 1, GL_FALSE, modelViewMatrix.PeekComponents());
		glUniformMatrix4fv(this->lineShader.ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->lineShader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
		glUniformMatrix4fv(this->lineShader.ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->lineShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
		glUniformMatrix4fv(this->lineShader.ParameterLocation("MVinvtrans"), 1, GL_FALSE, modelViewMatrixInvTrans.PeekComponents());
		glUniformMatrix4fv(this->lineShader.ParameterLocation("ProjInv"), 1, GL_FALSE, projectionMatrixInv.PeekComponents());

		glPointSize(5.0f);
		glLineWidth(1.0f);
		unsigned int startingIndex = 0;
		for (unsigned int i = 0; i < molSizes.size(); i++) {
			glUniform1i(this->lineShader.ParameterLocation("instanceOffset"), startingIndex);
			glPatchParameteri(GL_PATCH_VERTICES, 1);
			glDrawArrays(GL_PATCHES, 0, molSizes[i] - 3);
			startingIndex += molSizes[i];
		}
		this->lineShader.Disable();
	}

	if (this->showTubesParam.Param<param::BoolParam>()->Value()) {
		this->tubeShader.Enable();
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MV"), 1, GL_FALSE, modelViewMatrix.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVinv"), 1, GL_FALSE, modelViewMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVP"), 1, GL_FALSE, modelViewProjMatrix.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVPinv"), 1, GL_FALSE, modelViewProjMatrixInv.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, modelViewProjMatrixTransp.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("MVinvtrans"), 1, GL_FALSE, modelViewMatrixInvTrans.PeekComponents());
		glUniformMatrix4fv(this->tubeShader.ParameterLocation("ProjInv"), 1, GL_FALSE, projectionMatrixInv.PeekComponents());

		glUniform1f(this->tubeShader.ParameterLocation("tubewidth"), this->coilWidthParam.Param<param::FloatParam>()->Value());
		glUniform1f(this->tubeShader.ParameterLocation("structurewidth"), this->structureWidthParam.Param<param::FloatParam>()->Value());

		glPointSize(5.0f);
		glLineWidth(1.0f);
		unsigned int startingIndex = 0;
		for (unsigned int i = 0; i < molSizes.size(); i++) {
			glUniform1i(this->tubeShader.ParameterLocation("instanceOffset"), startingIndex);
			glPatchParameteri(GL_PATCH_VERTICES, 1);
			glDrawArrays(GL_PATCHES, 0, molSizes[i] - 3);
			startingIndex += molSizes[i];
		}
		this->tubeShader.Disable();
	}
	
	if (this->showAtomPositionsParam.Param<param::BoolParam>()->Value()) {
		glPointSize(5.0f);
		glBegin(GL_POINTS);
		for (unsigned int i = 0; i < this->cAlphas.size(); i++) {
			switch (this->cAlphas[i].type)
			{
			case 0:
				glColor4f(0.0f, 0.0f, 0.0f, 1.0f); break;
			case 1:
				glColor4f(0.0f, 0.0f, 1.0f, 1.0f); break;
			case 2:
				glColor4f(1.0f, 0.0f, 0.0f, 1.0f); break;
			case 3:
				glColor4f(0.0f, 1.0f, 0.0f, 1.0f); break;
			default:
				glColor4f(0.0f, 0.0f, 0.0f, 1.0f); break;
			}
			glVertex4f(cAlphas[i].pos[0], cAlphas[i].pos[1], 0.0f, 1.0f);
		}
		glEnd();
	}

	if (this->showDirectConnectionsParam.Param<param::BoolParam>()->Value()) {
		unsigned int start = 0;
		for (unsigned int i = 0; i < this->molSizes.size(); i++) {
			glBegin(GL_LINE_STRIP);
			for (unsigned int j = start; j < start + this->molSizes[i]; j++) {
				switch (this->cAlphas[j].type)
				{
				case 0:
					glColor4f(0.0f, 0.0f, 0.0f, 1.0f); break;
				case 1:
					glColor4f(0.0f, 0.0f, 1.0f, 1.0f); break;
				case 2:
					glColor4f(1.0f, 0.0f, 0.0f, 1.0f); break;
				case 3:
					glColor4f(0.0f, 1.0f, 0.0f, 1.0f); break;
				default:
					glColor4f(0.0f, 0.0f, 0.0f, 1.0f); break;
				}

				glVertex4f(cAlphas[j].pos[0], cAlphas[j].pos[1], 0.0f, 1.0f);
			}
			glEnd();
			start += molSizes[i];
		}
	}

	if (this->showHydrogenBondsParam.Param<param::BoolParam>()->Value()) {
		glBegin(GL_LINES);
		//glColor4f(0.87f, 0.92f, 0.97f, 1.0f);
		glColor4f(0.75f, 0.5f, 0.0f, 1.0f);

		if (mdc->AtomHydrogenBondsFake()) {
			for (unsigned int i = 0; i < mdc->HydrogenBondCount() * 2; i++) {
				unsigned int idx = this->cAlphaMap[mdc->GetHydrogenBonds()[i]];
				glVertex4f(cAlphas[idx].pos[0], cAlphas[idx].pos[1], 0.0f, 1.0f);
			}
		} else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "One needs to have fake hydrogen bonds to render them correctly");
		}

		glEnd();
	}

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	return true;
}

/*
 *	SecStructRenderer2D::rotatePlaneToXY
 */
vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> SecStructRenderer2D::rotatePlaneToXY(const vislib::math::Plane<float> plane) {

	/**
	 *	http://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane
	 */

	// translate plane to origin
	vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> transMat;
	transMat.SetIdentity();

	if (abs(plane.C()) > 0.000000001) {
		transMat.SetAt(2, 3, -(plane.D() / plane.C()));
	}

	// rotate plane to the xy plane
	vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> rotMat;
	rotMat.SetIdentity();

	vislib::math::Vector<float, 3> v(plane.A(), plane.B(), plane.C());
	vislib::math::Vector<float, 3> k(0.0f, 0.0f, 1.0f);
	float theta = acos(plane.C() / v.Length());
	
	vislib::math::Vector<float, 3> u = v.Cross(k) / v.Length();
	float ct = cos(theta);
	float st = sin(theta);

	// set matrix values
	float val = ct + u[0] * u[0] * (1.0f - ct);
	rotMat.SetAt(0, 0, val);
	val = u[0] * u[1] * (1 - ct);
	rotMat.SetAt(0, 1, val);
	val = u[1] * st;
	rotMat.SetAt(0, 2, val);
	val = u[0] * u[1] * (1 - ct);
	rotMat.SetAt(1, 0, val);
	val = ct + u[1] * u[1] * (1.0f - ct);
	rotMat.SetAt(1, 1, val);
	val = -u[0] * st;
	rotMat.SetAt(1, 2, val);
	val = -u[1] * st;
	rotMat.SetAt(2, 0, val);
	val = u[0] * st;
	rotMat.SetAt(2, 1, val);
	val = ct;
	rotMat.SetAt(2, 2, val);

	// perform the translation to origin first
	return rotMat * transMat;
}