/*
 *	QuickSurfRaycaster.cpp
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#include "stdafx.h"

//#define _USE_MATH_DEFINES 1

#include "QuickSurfRaycaster.h"
#include "mmcore/CoreInstance.h"
#include <gl/GLU.h>

#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::moldyn;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;

/*
 *	QuickSurfRaycaster::QuickSurfRaycaster
 */
QuickSurfRaycaster::QuickSurfRaycaster(void) : Renderer3DModule(),
	particleDataSlot("getData", "Connects the surface renderer with the particle data storage"),
	qualityParam("quicksurf::quality", "Quality"),
	radscaleParam("quicksurf::radscale", "Radius scale"),
	gridspacingParam("quicksurf::gridspacing", "Grid spacing"),
	isovalParam("quicksurf::isoval", "Isovalue"),
	setCUDAGLDevice(true),
	particlesSize(0)
{
	this->particleDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
	this->particleDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->particleDataSlot);

	this->qualityParam.SetParameter(new param::IntParam(1, 0, 4));
	this->MakeSlotAvailable(&this->qualityParam);

	this->radscaleParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->radscaleParam);

	this->gridspacingParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->gridspacingParam);

	this->isovalParam.SetParameter(new param::FloatParam(0.5f, 0.0f));
	this->MakeSlotAvailable(&this->isovalParam);

	lastViewport.Set(0, 0);

	cudaqsurf = NULL;
	cudaImage = NULL;
	volumeArray = NULL;
	particles = NULL;
	texHandle = 0;
}

/*
 *	QuickSurfRaycaster::~QuickSurfRaycaster
 */
QuickSurfRaycaster::~QuickSurfRaycaster(void) {
	if (cudaqsurf) {
		CUDAQuickSurf *cqs = (CUDAQuickSurf *)cudaqsurf;
		delete cqs;
		cqs = nullptr;
		cudaqsurf = nullptr;
	}
	
	this->Release();
}

/*
 *	QuickSurfRaycaster::release
 */
void QuickSurfRaycaster::release(void) {

}

void QuickSurfRaycaster::calcVolume(MultiParticleDataCall * mpdc, float* positions, int quality, float radscale, float gridspacing,
	float isoval, bool useCol) {

	
}

/*
 *	QuickSurfRaycaster::create
 */
bool QuickSurfRaycaster::create(void) {
	return initOpenGL();
}

/*
 *	QuickSurfRaycaster::GetCapabilities
 */
bool QuickSurfRaycaster::GetCapabilities(Call& call) {
	view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
	if (cr3d == NULL) return false;

	cr3d->SetCapabilities(view::AbstractCallRender3D::CAP_RENDER
		| view::AbstractCallRender3D::CAP_LIGHTING
		| view::AbstractCallRender3D::CAP_ANIMATION);

	return true;
}

/*
 *	QuickSurfRaycaster::GetExtents
 */
bool QuickSurfRaycaster::GetExtents(Call& call) {
	view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
	if (cr3d == NULL) return false;

	MultiParticleDataCall *mpdc = this->particleDataSlot.CallAs<MultiParticleDataCall>();
	MolecularDataCall *mdc = this->particleDataSlot.CallAs<MolecularDataCall>();

	if (mpdc == NULL && mdc == NULL) return false;

	// MultiParticleDataCall in use
	if (mpdc != NULL) {
		if (!(*mpdc)(1)) return false;

		float scale;
		if (!vislib::math::IsEqual(mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		else {
			scale = 1.0f;
		}
		cr3d->AccessBoundingBoxes() = mpdc->AccessBoundingBoxes();
		cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
		cr3d->SetTimeFramesCount(mpdc->FrameCount());
	} // MolecularDataCall in use
	else if (mdc != NULL) {
		if (!(*mdc)(1)) return false;

		float scale;
		if (!vislib::math::IsEqual(mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		else {
			scale = 1.0f;
		}
		cr3d->AccessBoundingBoxes() = mdc->AccessBoundingBoxes();
		cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
		cr3d->SetTimeFramesCount(mdc->FrameCount());
	}

	return true;
}

/*
 *	QuickSurfRaycaster::initCuda
 */
bool QuickSurfRaycaster::initCuda(view::CallRender3D& cr3d) {
	
	// set cuda device
	if (setCUDAGLDevice) {
#ifdef _WIN32
		if (cr3d.IsGpuAffinity()) {
			HGPUNV gpuId = cr3d.GpuAffinity<HGPUNV>();
			int devId;
			cudaWGLGetDevice(&devId, gpuId);
			cudaGLSetGLDevice(devId);
		}
		else {
			cudaGLSetGLDevice(cudaUtilGetMaxGflopsDeviceId());
		}
#else
		cudaGLSetGLDevice(cudaUtilGetMaxGflopsDeviceId());
#endif
		cudaError err = cudaGetLastError();
		if (err != 0) {
			printf("cudaGLSetGLDevice: %s\n", cudaGetErrorString(err));
			return false;
		}
		setCUDAGLDevice = false;
	}

	return true;
}

/*
 *	QuickSurfRaycaster::initPixelBuffer
 */
bool QuickSurfRaycaster::initPixelBuffer(view::CallRender3D& cr3d) {

	auto viewport = cr3d.GetViewport().GetSize();

	if (lastViewport == viewport) {
		return true;
	} else {
		lastViewport = viewport;
	}

	if (!texHandle) {
		glGenTextures(1, &texHandle);
		glActiveTexture(GL_TEXTURE15);
		glBindTexture(GL_TEXTURE_2D, texHandle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport.GetWidth(), viewport.GetHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	if (cudaImage) {
		checkCudaErrors(cudaFreeHost(cudaImage));
		cudaImage = NULL;
	}

	checkCudaErrors(cudaMallocHost((void**)&cudaImage, viewport.GetWidth() * viewport.GetHeight() * sizeof(unsigned int)));

	return true;
}

/*
 *	QuickSurfRaycaster::initOpenGL
 */
bool QuickSurfRaycaster::initOpenGL() {

	Vertex v0(-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f);
	Vertex v1(-1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
	Vertex v2(1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f);
	Vertex v3(1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f);

	std::vector<Vertex> verts = { v0, v2, v1, v3 };

	glGenVertexArrays(1, &textureVAO);
	glGenBuffers(1, &textureVBO);

	glBindVertexArray(textureVAO);
	glBindBuffer(GL_ARRAY_BUFFER, textureVBO);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)* verts.size(), verts.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, r));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	using namespace vislib::sys;
	using namespace vislib::graphics::gl;

	ShaderSource vertSrc;
	ShaderSource fragSrc;

	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quicksurfraycast::texture::textureVertex", vertSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for texture shader");
		return false;
	}

	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("quicksurfraycast::texture::textureFragment", fragSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for texture shader");
		return false;
	}

	this->textureShader.Compile(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());
	this->textureShader.Link();

	return true;
}

/*
 *	QuickSurfRaycaster::Render
 */
bool QuickSurfRaycaster::Render(Call& call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL) return false;

	this->cameraInfo = cr3d->GetCameraParameters();

	callTime = cr3d->Time();
	if (callTime < 1.0f) callTime = 1.0f;

	MultiParticleDataCall * mpdc = particleDataSlot.CallAs<MultiParticleDataCall>();
	MolecularDataCall * mdc = particleDataSlot.CallAs<MolecularDataCall>();

	float3 bbMin;
	float3 bbMax;

	if (mpdc == NULL && mdc == NULL) return false;

	if (mpdc != NULL) {
		mpdc->SetFrameID(static_cast<int>(callTime));
		if (!(*mpdc)(1)) return false;
		if (!(*mpdc)(0)) return false;

		numParticles = 0;
		//printf("ListCount: %i \n", mpdc->GetParticleListCount());
		for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
			numParticles += mpdc->AccessParticles(i).GetCount();
		}

		if (numParticles == 0) {
			return true;
		}

		//printf("NumParticles: %i\n", numParticles);

		if (this->particlesSize < this->numParticles * 4) {
			if (this->particles) {
				delete[] this->particles;
				this->particles = nullptr;
			}
			this->particles = new float[this->numParticles * 4];
			this->particlesSize = this->numParticles * 4;
		}
		memset(this->particles, 0, this->numParticles * 4 * sizeof(float));

		UINT64 particleCnt = 0;
		this->colorTable.clear();
		this->colorTable.reserve(numParticles * 3);

		for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
			MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);
			const float *pos = static_cast<const float*>(parts.GetVertexData());
			const float *colorPos = static_cast<const float*>(parts.GetColourData());
			unsigned int posStride = parts.GetVertexDataStride();
			unsigned int colStride = parts.GetColourDataStride();
			float globalRadius = parts.GetGlobalRadius();
			bool useGlobRad = (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
			int numColors = 0;

			switch (parts.GetColourDataType()) {
			case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: numColors = 1; break;
			case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: numColors = 3; break;
			case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: numColors = 4; break;
			case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: numColors = 0; break; // TODO
			case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: numColors = 0; break; // TODO
			}

			// if the vertices have no type, take the next list
			if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) { 
				continue;
			}
			if (useGlobRad) { // TODO is this correct?
				if (posStride < 12) posStride = 12;
			}
			else {
				if (posStride < 16) posStride = 16;
			}

			// Why does this work?
			for (UINT64 j = 0; j < parts.GetCount(); j++, pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride),
				colorPos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(colorPos) + colStride)) {

				particles[particleCnt * 4 + 0] = pos[0];
				particles[particleCnt * 4 + 1] = pos[1];
				particles[particleCnt * 4 + 2] = pos[2];
				if (useGlobRad) { // TODO take the concentration instead of the radius
#define SCIVISCONTEST2016
#ifdef SCIVISCONTEST2016
					particles[particleCnt * 4 + 3] = colorPos[3]; // concentration
#else
					particles[particleCnt * 4 + 3] = globalRadius;
#endif
				}
				else {
					particles[particleCnt * 4 + 3] = pos[3];
				}

				this->colorTable.push_back(1.0f);
				this->colorTable.push_back(1.0f);
				this->colorTable.push_back(1.0f);

				// TODO delete that, when possible
				if (numColors > 3)
					numColors = 3;

				for (int k = 0; k < numColors; k++) {
					for (int l = 0; l < 3 - k; l++) {
						this->colorTable[particleCnt * 3 + k + l] = colorPos[k];
					}
				}

				particleCnt++;
			}
		}

		glPushMatrix();
		float scale = 1.0f;
		if (!vislib::math::IsEqual(mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		glScalef(scale, scale, scale);

		auto bb = mpdc->GetBoundingBoxes().ObjectSpaceBBox();
		bbMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
		bbMax = make_float3(bb.Right(), bb.Top(), bb.Front());

	} else if (mdc != NULL) {
		// TODO
		printf("MolecularDataCall currently not supported\n");
		return false;
	}

	initCuda(*cr3d);
	initPixelBuffer(*cr3d);

	auto viewport = cr3d->GetViewport().GetSize();

    GLfloat m[16];
	GLfloat m_proj[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glGetFloatv(GL_PROJECTION_MATRIX, m_proj);
    Mat4f modelMatrix(&m[0]);
	Mat4f projectionMatrix(&m_proj[0]);
    modelMatrix.Invert();
	projectionMatrix.Invert();
	float3 camPos = make_float3(modelMatrix.GetAt(0, 3), modelMatrix.GetAt(1, 3), modelMatrix.GetAt(2, 3));
	float3 camDir = norm(make_float3(modelMatrix.GetAt(0, 2), modelMatrix.GetAt(1, 2), modelMatrix.GetAt(2, 2)));
	float3 camUp = norm(make_float3(-modelMatrix.GetAt(0, 1), -modelMatrix.GetAt(1, 1), -modelMatrix.GetAt(2, 1)));
	float3 camRight = norm(make_float3(-modelMatrix.GetAt(0, 0), -modelMatrix.GetAt(1, 0), -modelMatrix.GetAt(2, 0)));
	// be careful: the minuses in the up and right vector computation are a dirty trick that makes everything work

	auto cam = cr3d->GetCameraParameters();

	float fovy = (float)(cam->ApertureAngle() * M_PI / 180.0f);

	float aspect = (float)viewport.GetWidth() / (float)viewport.GetHeight();
	if (viewport.GetHeight() == 0)
		aspect = 0.0f;

	float fovx = 2.0f * atan(tan(fovy / 2.0f) * aspect);
	float zNear = (2.0f * projectionMatrix.GetAt(2, 3)) / (2.0f * projectionMatrix.GetAt(2, 2) - 2.0f);

	float density = 0.05f;
	float brightness = 1.0f;
	float transferOffset = 0.0f;
	float transferScale = 1.0f;

	// TODO render volume
	dim3 blockSize = dim3(8, 8);
	dim3 gridSize = dim3(iDivUp(viewport.GetWidth(), blockSize.x), iDivUp(viewport.GetHeight(), blockSize.y));

	/*printf("min: %f %f %f \n", bbMin.x, bbMin.y, bbMin.z);
	printf("max: %f %f %f \n", bbMax.x, bbMax.y, bbMax.z);*/

	render_kernel(gridSize, blockSize, cudaImage, viewport.GetWidth(), viewport.GetHeight(), fovx, fovy, camPos, camDir, camUp, camRight, zNear, density, brightness, transferOffset, transferScale, bbMin, bbMax);
	getLastCudaError("kernel failed");

	checkCudaErrors(cudaDeviceSynchronize());

	/*for (int i = 0; i < viewport.GetWidth() * viewport.GetHeight(); i++)
		printf("%u \n", cudaImage[i]);*/

	glActiveTexture(GL_TEXTURE15);
	glBindTexture(GL_TEXTURE_2D, texHandle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport.GetWidth(), viewport.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, cudaImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_ALWAYS);

	textureShader.Enable();
	
	glBindVertexArray(textureVAO);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindVertexArray(0);
	textureShader.Disable();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	/*glDisable(GL_CULL_FACE);
	glBegin(GL_TRIANGLES);
	glColor4f(1, 0, 0, 1);
	glVertex3f(bbMin.x, bbMin.y, bbMin.z);
	glVertex3f(bbMax.x, bbMin.y, bbMin.z);
	glVertex3f((bbMax.x - bbMin.x) / 2.0f, bbMax.y, bbMin.z);
	glEnd();*/

	return true;
}