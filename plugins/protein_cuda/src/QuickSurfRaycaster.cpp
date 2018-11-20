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
//#include "vislib/graphics/gl/IncludeAllGL.h"

#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/StringTokeniser.h"

#include <channel_descriptor.h>
#include <driver_functions.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::moldyn;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;
using namespace megamol::core::misc;

/*
 *	QuickSurfRaycaster::QuickSurfRaycaster
 */
QuickSurfRaycaster::QuickSurfRaycaster(void) : Renderer3DModule(),
	particleDataSlot("getData", "Connects the surface renderer with the particle data storage"),
	getClipPlaneSlot("getClipPlane", "Connects the surface renderer with a clipplane module"),
	qualityParam("quicksurf::quality", "Quality"),
	radscaleParam("quicksurf::radscale", "Radius scale"),
	gridspacingParam("quicksurf::gridspacing", "Grid spacing"),
	isovalParam("quicksurf::isoval", "Isovalue"),
	selectedIsovals("render::selectedIsovals", "Semicolon seperated list of normalized isovalues we want to ray cast the isoplanes from"),
	scalingFactor("quicksurf::scalingFactor", "Scaling factor for the density values and particle radii"),
	concFactorParam("quicksurf::concentrationFactor", "Scaling factor for particle radii based on their concentration"),
	maxRadius("quicksurf::maxRadius", "The maximal particle influence radius the quicksurf algorithm uses"),
	convertedIsoValueParam("render::convertedIsovalue", "The isovalue the mesh gets generated from"),
	triggerConvertButtonParam("render::triggerConversion", "Button starting the conversion from volume to mesh data"),
	showDepthTextureParam("render::showDepthTexture", "Toggles the display of the depth texture"),
	setCUDAGLDevice(true),
	firstTransfer(true),
	recomputeVolume(true),
	particlesSize(0)
{
	this->particleDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
	this->particleDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->particleDataSlot.SetCompatibleCall<VolumetricDataCallDescription>();
	this->MakeSlotAvailable(&this->particleDataSlot);

	this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
	this->MakeSlotAvailable(&this->getClipPlaneSlot);

	this->qualityParam.SetParameter(new param::IntParam(1, 0, 4));
	this->MakeSlotAvailable(&this->qualityParam);

	this->radscaleParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->radscaleParam);

	//this->gridspacingParam.SetParameter(new param::FloatParam(0.01953125f, 0.0f));
	this->gridspacingParam.SetParameter(new param::FloatParam(0.2f, 0.0f));
	this->MakeSlotAvailable(&this->gridspacingParam);

	this->isovalParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->isovalParam);

	this->selectedIsovals.SetParameter(new param::StringParam("0.1,0.9"));
	this->MakeSlotAvailable(&this->selectedIsovals);
	this->selectedIsovals.ForceSetDirty(); // necessary for initial update
	isoVals.push_back(0.1f);
	isoVals.push_back(0.9f);

	this->convertedIsoValueParam.SetParameter(new param::FloatParam(0.1f, 0.0f, 1.0f));
	this->MakeSlotAvailable(&this->convertedIsoValueParam);

	this->triggerConvertButtonParam.SetParameter(new param::ButtonParam('I'));
	this->MakeSlotAvailable(&this->triggerConvertButtonParam);

	this->scalingFactor.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->scalingFactor);

	this->concFactorParam.SetParameter(new param::FloatParam(0.5f, 0.0f));
	this->MakeSlotAvailable(&this->concFactorParam);

	this->maxRadius.SetParameter(new param::FloatParam(0.5, 0.0f));
	this->MakeSlotAvailable(&this->maxRadius);

	this->showDepthTextureParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->showDepthTextureParam);

	lastViewport.Set(0, 0);

	volumeExtent = make_cudaExtent(0, 0, 0);

	cudaqsurf = nullptr;
	cudaImage = nullptr;
	cudaDepthImage = nullptr;
	volumeArray = nullptr;
	particles = nullptr;
	texHandle = 0;
	mcVertices_d = nullptr;
	mcNormals_d = nullptr;
	mcColors_d = nullptr;

	curTime = 0;
}

/*
 *	QuickSurfRaycaster::~QuickSurfRaycaster
 */
QuickSurfRaycaster::~QuickSurfRaycaster(void) {
	if (cudaqsurf) {
		CUDAQuickSurfAlternative *cqs = (CUDAQuickSurfAlternative *)cudaqsurf;
		delete cqs;
		cqs = nullptr;
		cudaqsurf = nullptr;
	}

	if (mcVertices_d != NULL) {
		cudaFree(mcVertices_d);
		mcVertices_d = NULL;
	}

	if (mcNormals_d != NULL) {
		cudaFree(mcNormals_d);
		mcNormals_d = NULL;
	}

	if (mcColors_d != NULL) {
		cudaFree(mcColors_d);
		mcColors_d = NULL;
	}
	
	this->Release();
}

/*
 *	QuickSurfRaycaster::release
 */
void QuickSurfRaycaster::release(void) {

}

bool QuickSurfRaycaster::calcVolume(float3 bbMin, float3 bbMax, float* positions, int quality, float radscale, float gridspacing,
	float isoval, float minConcentration, float maxConcentration, bool useCol, int timestep) {

	float x = bbMax.x - bbMin.x;
	float y = bbMax.y - bbMin.y;
	float z = bbMax.z - bbMin.z;

	int numVoxels[3];
	numVoxels[0] = (int)ceil(x / gridspacing);
	numVoxels[1] = (int)ceil(y / gridspacing);
	numVoxels[2] = (int)ceil(z / gridspacing);

	x = (numVoxels[0] - 1) * gridspacing;
	y = (numVoxels[1] - 1) * gridspacing;
	z = (numVoxels[2] - 1) * gridspacing;

	//printf("vox %i %i %i \n", numVoxels[0], numVoxels[1], numVoxels[2]);

	volumeExtent = make_cudaExtent(numVoxels[0], numVoxels[1], numVoxels[2]);
	volumeExtentSmall = make_cudaExtent(numVoxels[0], numVoxels[1], numVoxels[2]);

	float gausslim = 2.0f;
	switch (quality) { // TODO adjust values
	case 3: gausslim = 4.0f; break;
	case 2: gausslim = 3.0f; break;
	case 1: gausslim = 2.5f; break;
	case 0:
	default: gausslim = 2.0f; break;
	}

	float origin[3] = { bbMin.x, bbMin.y, bbMin.z };

	if (cudaqsurf == NULL) {
		cudaqsurf = new CUDAQuickSurfAlternative();
	}

	CUDAQuickSurfAlternative *cqs = (CUDAQuickSurfAlternative*)cudaqsurf;

	int result = -1;
	result = cqs->calc_map((long)particleCnt, positions, colorTable.data(), 1, 
		origin, numVoxels, maxConcentration, radscale, gridspacing, 
		isoval, gausslim, false, timestep, 20);

	checkCudaErrors(cudaDeviceSynchronize());

	volumeExtent = make_cudaExtent(cqs->getMapSizeX(), cqs->getMapSizeY(), cqs->getMapSizeZ());

	// make the initial plane more beautiful
	if (volumeExtent.depth > volumeExtentSmall.depth)
		volumeExtentSmall.depth = volumeExtentSmall.depth + 1;

	return (result == 0);
}

/*
 *	QuickSurfRaycaster::create
 */
bool QuickSurfRaycaster::create(void) {
	return initOpenGL();
}

/*
 *	QuickSurfRaycaster::convertToMesh
 */
void QuickSurfRaycaster::convertToMesh(float * volumeData, cudaExtent volSize, float3 bbMin, float3 bbMax, float isoValue, float concMin, float concMax) {

	uint3 extents = make_uint3(static_cast<unsigned int>(volSize.width), static_cast<unsigned int>(volSize.height), static_cast<unsigned int>(volSize.depth));
	unsigned int chunkmaxverts = 3 * extents.x * extents.y * extents.z;

	float * data = volumeData;

#define DOWNSAMPLE
#ifdef DOWNSAMPLE
	extents = make_uint3(extents.x / 2, extents.y / 2, extents.z / 2);
	chunkmaxverts = 3 * extents.x * extents.y * extents.z;

	data = new float[extents.x * extents.y * extents.z];

	float newMin = FLT_MAX;
	float newMax = FLT_MIN;

	for (int index = 0; index < static_cast<int>(chunkmaxverts / 3); index++) {
		int i = (index % (extents.x * extents.y)) % extents.x;
		int j = (index % (extents.x * extents.y)) / extents.x;
		int k = index / (extents.x * extents.y);

		float val = 0.0f;
		val += volumeData[(2*k)   * extents.x * extents.y + (2*j)   * extents.x + (2*i)];
		val += volumeData[(2*k)   * extents.x * extents.y + (2*j)   * extents.x + (2*i+1)];
		val += volumeData[(2*k)   * extents.x * extents.y + (2*j+1) * extents.x + (2*i)];
		val += volumeData[(2*k)   * extents.x * extents.y + (2*j+1) * extents.x + (2*i+1)];
		val += volumeData[(2*k+1) * extents.x * extents.y + (2*j)   * extents.x + (2*i)];
		val += volumeData[(2*k+1) * extents.x * extents.y + (2*j)   * extents.x + (2*i+1)];
		val += volumeData[(2*k+1) * extents.x * extents.y + (2*j+1) * extents.x + (2*i)];
		val += volumeData[(2*k+1) * extents.x * extents.y + (2*j+1) * extents.x + (2*i+1)];
		data[k * extents.x * extents.y + j * extents.x + i] = val / 8.0f;

		if (val / 8.0f < newMin) newMin = val / 8.0f;
		if (val / 8.0f > newMax) newMax = val / 8.0f;
	}
#else
	float newMin = concMin;
	float newMax = concMax;
#endif

#define NORMALIZE
#ifdef NORMALIZE
	for (int i = 0; i < static_cast<int>(chunkmaxverts / 3); i++) {
		data[i] = (data[i] - newMin) / (newMax - newMin);
	}
#endif

	if (cudaMarching == NULL) {
		cudaMarching = new CUDAMarchingCubes();
		((CUDAMarchingCubes*)cudaMarching)->Initialize(extents);
	}

	CUDAMarchingCubes * cmc = (CUDAMarchingCubes*)cudaMarching;
	uint3 oldExtents = cmc->GetMaxGridSize();
	
	if (extents.x > oldExtents.x || extents.y > oldExtents.y || extents.z > oldExtents.z) {
		cmc->Initialize(extents);
	}

	float3 bbSize = make_float3(bbMax.x - bbMin.x, bbMax.y - bbMin.y, bbMax.z - bbMin.z);
	float3 bbNewMin = make_float3(0.0f, 0.0f, 0.0f);

	//std::cout << chunkmaxverts * sizeof(float3) << std::endl;
	printf("min %f ; max %f\n", newMin, newMax);

	float myIsoVal = isoValue;
	//myIsoVal = 0.2f;

	// TODO is the linear interpolation necessary? (use only isoValue instead)
	//cmc->SetIsovalue((1.0f - myIsoVal) * newMin + myIsoVal * newMax);
	cmc->SetIsovalue(myIsoVal);
	if (!cmc->SetVolumeData(data, NULL, extents, bbMin, bbSize, false)) {
		printf("SetVolumeData failed!\n");
	}

	if (mcVertices_d != NULL) {
		cudaFree(mcVertices_d);
		mcVertices_d = NULL;
	}

	if (mcNormals_d != NULL) {
		cudaFree(mcNormals_d);
		mcNormals_d = NULL;
	}

	if (mcColors_d != NULL) {
		cudaFree(mcColors_d);
		mcColors_d = NULL;
	}

	checkCudaErrors(cudaMalloc((void**)&mcVertices_d, chunkmaxverts * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&mcNormals_d, chunkmaxverts * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&mcColors_d, chunkmaxverts * sizeof(float3)));

	cmc->computeIsosurface(mcVertices_d, mcNormals_d, mcColors_d, chunkmaxverts);
	checkCudaErrors(cudaDeviceSynchronize());

	int64_t count = cmc->GetVertexCount();

	float *mcVertices_h = new float[count * 3 * sizeof(float)];
	float *mcNormals_h = new float[count * 3 * sizeof(float)];

	checkCudaErrors(cudaMemcpy(mcVertices_h, mcVertices_d, count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mcNormals_h, mcNormals_d, count * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "Found " << count / 3 << " triangles" << std::endl;

	std::ofstream file("mesh.obj");
	for (int i = 0; i < count; i++) { // vertices
		file << "v " << std::to_string(mcVertices_h[i * 3 + 0]) << " " << std::to_string(mcVertices_h[i * 3 + 1]) << " " << std::to_string(mcVertices_h[i * 3 + 2]) << std::endl;
	}
	file << std::endl;
	for (int i = 0; i < count; i++) { // normals
		file << "vn " << std::to_string(mcNormals_h[i * 3 + 0]) << " " << std::to_string(mcNormals_h[i * 3 + 1]) << " " << std::to_string(mcNormals_h[i * 3 + 2]) << std::endl;
	}
	file << std::endl;
	for (int i = 0; i < count / 3; i++) { // triangles
		std::string i1 = std::to_string(i * 3 + 1);
		std::string i2 = std::to_string(i * 3 + 2);
		std::string i3 = std::to_string(i * 3 + 3);

		file << "f " << i1 << "//" << i1 << " " << i2 << "//" << i2 << " " << i3 << "//" << i3 << std::endl;
	}

	file.close();

	delete[] mcVertices_h;
	mcVertices_h = nullptr;
	delete[] mcNormals_h;
	mcNormals_h = nullptr;

#ifdef DOWNSAMPLE
	delete[] data;
	data = nullptr;
#endif
}

/*
 *	QuickSurfRaycaster::GetExtents
 */
bool QuickSurfRaycaster::GetExtents(Call& call) {
	view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
	if (cr3d == NULL) return false;

	MultiParticleDataCall *mpdc = this->particleDataSlot.CallAs<MultiParticleDataCall>();
	MolecularDataCall *mdc = this->particleDataSlot.CallAs<MolecularDataCall>();
	VolumetricDataCall *vdc = this->particleDataSlot.CallAs<VolumetricDataCall>();

	if (mpdc == NULL && mdc == NULL && vdc == NULL) return false;

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
	else if (vdc != NULL) {
		if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;

		float scale;
		if (!vislib::math::IsEqual(vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		else {
			scale = 1.0f;
		}

		auto bb = vdc->AccessBoundingBoxes().ObjectSpaceBBox();

		cr3d->AccessBoundingBoxes() = vdc->AccessBoundingBoxes();
		cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
		cr3d->SetTimeFramesCount(vdc->FrameCount());
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
		GLint texID;
		glGetIntegerv(GL_ACTIVE_TEXTURE, &texID);
		glGenTextures(1, &texHandle);
		glActiveTexture(GL_TEXTURE15);
		glBindTexture(GL_TEXTURE_2D, texHandle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport.GetWidth(), viewport.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
		glActiveTexture(texID);
	}

	if (!depthTexHandle) {
		GLint texID;
		glGetIntegerv(GL_ACTIVE_TEXTURE, &texID);
		glGenTextures(1, &depthTexHandle);
		glActiveTexture(GL_TEXTURE16);
		glBindTexture(GL_TEXTURE_2D, depthTexHandle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, viewport.GetWidth(), viewport.GetHeight(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
		glActiveTexture(texID);
	}

	if (cudaImage) {
		checkCudaErrors(cudaFreeHost(cudaImage));
		cudaImage = NULL;
	}

	if (cudaDepthImage) {
		checkCudaErrors(cudaFreeHost(cudaDepthImage));
		cudaDepthImage = NULL;
	}

	checkCudaErrors(cudaMallocHost((void**)&cudaImage, viewport.GetWidth() * viewport.GetHeight() * sizeof(unsigned int)));
	checkCudaErrors(cudaMallocHost((void**)&cudaDepthImage, viewport.GetWidth() * viewport.GetHeight() * sizeof(float)));

	return true;
}

/*
 *	QuickSurfRaycaster::initOpenGL
 */
bool QuickSurfRaycaster::initOpenGL() {

	Vertex v0(-1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f);
	Vertex v1(-1.0f, 1.0f, 0.5f, 0.0f, 1.0f, 1.0f, 1.0f);
	Vertex v2(1.0f, -1.0f, 0.5f, 1.0f, 0.0f, 1.0f, 1.0f);
	Vertex v3(1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f);

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

	if (isoVals.size() < 1) return true;

	this->cameraInfo = cr3d->GetCameraParameters();

	callTime = cr3d->Time();

	if (!this->volume_fbo.IsValid() || 
		this->volume_fbo.GetWidth() != cameraInfo->VirtualViewSize().GetWidth() || 
		this->volume_fbo.GetHeight() != cameraInfo->VirtualViewSize().GetHeight()) {
		
		unsigned int width = cameraInfo->VirtualViewSize().GetWidth();
		unsigned int height = cameraInfo->VirtualViewSize().GetHeight();
		this->volume_fbo.Create(width, height);
	}

	//int myTime = curTime; // only for writing of velocity data
	int myTime = static_cast<int>(callTime);

	if (lastTimeVal != myTime) {
		recomputeVolume = true;
	}

	if (concFactorParam.IsDirty() || gridspacingParam.IsDirty() 
		|| isovalParam.IsDirty() || qualityParam.IsDirty() 
		|| radscaleParam.IsDirty() || scalingFactor.IsDirty()) {

		concFactorParam.ResetDirty();
		gridspacingParam.ResetDirty();
		isovalParam.ResetDirty(); 
		qualityParam.ResetDirty();
		radscaleParam.ResetDirty();
		scalingFactor.ResetDirty();

		recomputeVolume = true;
	}

	MultiParticleDataCall * mpdc = particleDataSlot.CallAs<MultiParticleDataCall>();
	MolecularDataCall * mdc = particleDataSlot.CallAs<MolecularDataCall>();
	VolumetricDataCall * vdc = particleDataSlot.CallAs<VolumetricDataCall>();

	float3 bbMin;
	float3 bbMax;

	float3 clipBoxMin;
	float3 clipBoxMax;

	float concMin = FLT_MAX;
	float concMax = FLT_MIN;

	bool onlyVolumetric = false;

	if (mpdc == NULL && mdc == NULL && vdc == NULL) return false;

	if (mpdc != NULL) {

		mpdc->SetFrameID(myTime);
		if (!(*mpdc)(1)) return false;
		if (!(*mpdc)(0)) return false;

		auto bb = mpdc->GetBoundingBoxes().ObjectSpaceBBox();
		bbMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
		bbMax = make_float3(bb.Right(), bb.Top(), bb.Front());
		bb = mpdc->GetBoundingBoxes().ClipBox();
		clipBoxMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
		clipBoxMax = make_float3(bb.Right(), bb.Top(), bb.Front());

		if (recomputeVolume) {

			numParticles = 0;
			for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
				numParticles += mpdc->AccessParticles(i).GetCount();
			}

			if (numParticles == 0) {
				return true;
			}

			if (this->particlesSize < this->numParticles * 4) {
				if (this->particles) {
					delete[] this->particles;
					this->particles = nullptr;
				}
				this->particles = new float[this->numParticles * 4];
				this->particlesSize = this->numParticles * 4;
			}
			memset(this->particles, 0, this->numParticles * 4 * sizeof(float));

			particleCnt = 0;
			this->colorTable.clear();
			this->colorTable.resize(numParticles * 4, 0.0f);

			//printf("bbMin %f %f %f\n", bbMin.x, bbMin.y, bbMin.z);
			//exit(-1);

			//#define FILTER
#ifdef FILTER // filtering: calculate min and max beforehand
			for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
				MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);
				const float *colorPos = static_cast<const float*>(parts.GetColourData());
				unsigned int colStride = parts.GetColourDataStride();
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

				if (numColors > 0) {

					for (UINT64 j = 0; j < parts.GetCount(); j++, colorPos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(colorPos)+colStride)) {

						if (colorPos[numColors - 1] < concMin) concMin = colorPos[numColors - 1];
						if (colorPos[numColors - 1] > concMax) concMax = colorPos[numColors - 1];
					}
				}
			}
#endif

			particleCnt = 0;

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

				for (UINT64 j = 0; j < parts.GetCount(); j++, pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos)+posStride),
					colorPos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(colorPos)+colStride)) {

#ifdef FILTER
					if (colorPos[numColors - 1] > (concMax - concMin) * isoVals[0] + concMin
						&& colorPos[numColors - 1] < (concMax - concMin) * isoVals[isoVals.size() - 1] + concMin) {
#endif FILTER
						particles[particleCnt * 4 + 0] = pos[0] - bbMin.x;
						particles[particleCnt * 4 + 1] = pos[1] - bbMin.y;
						particles[particleCnt * 4 + 2] = pos[2] - bbMin.z;
						if (useGlobRad) {
							particles[particleCnt * 4 + 3] = globalRadius;
						}
						else {
							particles[particleCnt * 4 + 3] = pos[3];
						}

#ifndef FILTER // calculate the min and max here if no filtering performed
						if (colorPos[numColors - 1] < concMin) concMin = colorPos[numColors - 1];
						if (colorPos[numColors - 1] > concMax) concMax = colorPos[numColors - 1];
#endif

						/*---------------------------------choose-one---------------------------------------------------------*/
#define ALWAYS4COLORS
#ifndef ALWAYS4COLORS
						// 1. copy all available values into the color, the rest gets filled up with the last available value
						for (int k = 0; k < numColors; k++) {
							for (int l = 0; l < 3 - k; l++) {
								this->colorTable[particleCnt * 4 + k + l] = colorPos[k];
							}
						}
#else
						for (int k = 0; k < 4; k++) {
							this->colorTable[particleCnt * 4 + k] = colorPos[k];
						}
#endif

#ifdef FILTER
						// normalize concentration, multiply it with a factor and write it
						// TODO do weird things with the concentration so it results in a nice iso-surface
						this->colorTable[particleCnt * 4 + 3] = ((colorPos[numColors - 1] - concMin) / (concMax - concMin)) * concFactorParam.Param<param::FloatParam>()->Value();
#else
						// normalization of the values happens later
						this->colorTable[particleCnt * 4 + 3] = colorPos[numColors - 1];
#endif


						// 2. fill r,g & b with the last available color value (should be density)
						/*for (int k = 0; k < 3; k++) {
							this->colorTable[particleCnt * 4 + k] = colorPos[numColors - 1];
							}*/

						/*---------------------------------------------------------------------------------------------------*/

						particleCnt++;
#ifdef FILTER
					}
#endif FILTER
				}
			}

#ifndef FILTER // no filtering: we need to normalize the concentration values
			for (int i = 0; i < particleCnt; i++) {
				this->colorTable[i * 4 + 3] = ((this->colorTable[i * 4 + 3] - concMin) / (concMax - concMin)) * concFactorParam.Param<param::FloatParam>()->Value();
			}
#endif

			if (particleCnt == 0) {
				return true;
			}

			//this->particlesSize = this->particleCnt * 4; // adapt size of the particle list
			//this->colorTable.resize(particleCnt * 4); // shrink color vector

			//printf("conc: %f %f\n", concMin, concMax);
			//printf("part: %llu\n", particleCnt);
			//printf("col: %u\n", colorTable.size());
		}

		//glPushMatrix();
		float scale = 1.0f;
		if (!vislib::math::IsEqual(mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		glScalef(scale, scale, scale);

		mpdc->Unlock();

		onlyVolumetric = false;

	} else if (mdc != NULL) {
		// TODO
		printf("MolecularDataCall currently not supported\n");
		mdc->Unlock();
		onlyVolumetric = false;
		return false;
	} else if (vdc != NULL) {

		vdc->SetFrameID(myTime);
		if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;
		if (!(*vdc)(vdc->IDX_GET_METADATA)) return false;
		if (!(*vdc)(vdc->IDX_GET_DATA)) return false;
		
		//glPushMatrix();
		float scale = 1.0f;
		if (!vislib::math::IsEqual(vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		glScalef(scale, scale, scale);
		onlyVolumetric = true;
	}

	initCuda(*cr3d);
	initPixelBuffer(*cr3d);

	float factor = scalingFactor.Param<param::FloatParam>()->Value();

	auto viewport = cr3d->GetViewport().GetSize();

	// get the clip plane
	view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
	bool clipplaneAvailable = false;
	float4 plane;

	if ((ccp != nullptr) && (*ccp)(0)) {
		float a, b, c, d;
		a = ccp->GetPlane().A();
		b = ccp->GetPlane().B();
		c = ccp->GetPlane().C();
		d = ccp->GetPlane().D();
		// we have to normalise the parameters:
		float len = std::sqrtf(a * a + b * b + c * c);
		if (!vislib::math::IsEqual(len, 0.0f)) {
			a = a / len;
			b = b / len;
			c = c / len;
			d = d / len;
			clipplaneAvailable = true;
			plane = make_float4(a, b, c, d);
		} else {
			a = b = c = d = 0.0f;
			clipplaneAvailable = false;
		}
	}

    GLfloat m[16];
	GLfloat m_proj[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glGetFloatv(GL_PROJECTION_MATRIX, m_proj);
    Mat4f modelMatrix(&m[0]);
	Mat4f projectionMatrix(&m_proj[0]);
	Mat4f mvpMatrix = projectionMatrix * modelMatrix;
    modelMatrix.Invert();
	projectionMatrix.Invert();
	float3 camPos = make_float3(modelMatrix.GetAt(0, 3), modelMatrix.GetAt(1, 3), modelMatrix.GetAt(2, 3));
	float3 camDir = norm(make_float3(-modelMatrix.GetAt(0, 2), -modelMatrix.GetAt(1, 2), -modelMatrix.GetAt(2, 2)));
	float3 camUp = norm(make_float3(modelMatrix.GetAt(0, 1), modelMatrix.GetAt(1, 1), modelMatrix.GetAt(2, 1)));
	float3 camRight = norm(make_float3(modelMatrix.GetAt(0, 0), modelMatrix.GetAt(1, 0), modelMatrix.GetAt(2, 0)));
	// the direction has to be negated because of the right-handed view space of OpenGL

	mvpMatrix.Transpose();
	copyMVPMatrix(mvpMatrix.PeekComponents(), 4 * sizeof(float4));

	auto cam = cr3d->GetCameraParameters();

	float fovy = (float)(cam->ApertureAngle() * M_PI / 180.0f);

	float aspect = (float)viewport.GetWidth() / (float)viewport.GetHeight();
	if (viewport.GetHeight() == 0)
		aspect = 0.0f;

	float fovx = 2.0f * atan(tan(fovy / 2.0f) * aspect);
	float zNear = (2.0f * projectionMatrix.GetAt(2, 3)) / (2.0f * projectionMatrix.GetAt(2, 2) - 2.0f);
	float zFar = ((projectionMatrix.GetAt(2, 2) - 1.0f) * zNear) / (projectionMatrix.GetAt(2, 2) + 1.0f);

	float density = 0.5f;
	float brightness = 1.0f;
	float transferOffset = 0.0f;
	float transferScale = 1.0f;

	/*printf("min: %f %f %f \n", clipBoxMin.x, clipBoxMin.y, clipBoxMin.z);
	printf("max: %f %f %f \n\n", clipBoxMax.x, clipBoxMax.y, clipBoxMax.z);*/

	dim3 blockSize = dim3(8, 8);
	dim3 gridSize = dim3(iDivUp(viewport.GetWidth(), blockSize.x), iDivUp(viewport.GetHeight(), blockSize.y));

	if (recomputeVolume && !onlyVolumetric) {
		if (cudaqsurf == NULL) {
			cudaqsurf = new CUDAQuickSurfAlternative();
		}

#ifdef FILTER
		bool suc = this->calcVolume(bbMin, bbMax, particles, 
			this->qualityParam.Param<param::IntParam>()->Value(),
			this->radscaleParam.Param<param::FloatParam>()->Value(),
			this->gridspacingParam.Param<param::FloatParam>()->Value(),
			1.0f, // necessary to switch off velocity scaling
			(concMax - concMin) * isoVals[0] + concMin, (concMax - concMin) * isoVals[isoVals.size() - 1] + concMin, true,
			myTime);
#else
		bool suc = this->calcVolume(bbMin, bbMax, particles,
			this->qualityParam.Param<param::IntParam>()->Value(),
			this->radscaleParam.Param<param::FloatParam>()->Value(),
			this->gridspacingParam.Param<param::FloatParam>()->Value(),
			1.0f, // necessary to switch off velocity scaling
			0.0f, this->maxRadius.Param<param::FloatParam>()->Value(), true,
			myTime);
#endif
		
		//if (!suc) return false;
	}

	if (recomputeVolume && !onlyVolumetric) {
		CUDAQuickSurfAlternative * cqs = (CUDAQuickSurfAlternative*)cudaqsurf;
		map = cqs->getMap();
	}

		//transferNewVolume(map, volumeExtent);

	if (!onlyVolumetric) { // quicksurfed data
		transferNewVolume(map, volumeExtentSmall);
	}
	else { // pure volume data
		auto xDir = vdc->GetResolution(0);
		auto yDir = vdc->GetResolution(1);
		auto zDir = vdc->GetResolution(2);
		volumeExtentSmall = make_cudaExtent(xDir, yDir, zDir);
		volumeExtent = volumeExtentSmall;

		auto bb = vdc->GetBoundingBoxes().ObjectSpaceBBox();
		bbMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
		bbMax = make_float3(bb.Right(), bb.Top(), bb.Front());

		if (vdc->GetComponents() > 1 || vdc->GetComponents() < 1) {
			return false;
		}

		if (vdc->GetScalarType() != VolumetricDataCall::ScalarType::FLOATING_POINT) {
			return false;
		}

		auto volPtr = vdc->GetData();
		auto voxelNumber = volumeExtentSmall.width * volumeExtentSmall.height * volumeExtentSmall.depth;
		float * fPtr = reinterpret_cast<float*>(volPtr);

		concMin = FLT_MAX;
		concMax = FLT_MIN;

		for (int i = 0; i < voxelNumber; i++) {
			if (fPtr[i] < concMin) concMin = fPtr[i];
			if (fPtr[i] > concMax) concMax = fPtr[i];
		}
			
		if (volPtr != NULL){
			transferVolumeDirect(volPtr, volumeExtentSmall, concMin, concMax);
			checkCudaErrors(cudaDeviceSynchronize());
		}
		else {
			printf("Volume data was NULL");
			return false;
		}

		if (this->triggerConvertButtonParam.IsDirty()) {
		
			printf("Converting the isosurface of %f to a mesh\n", this->convertedIsoValueParam.Param<param::FloatParam>()->Value());
			convertToMesh(static_cast<float*>(volPtr), volumeExtentSmall, bbMin, bbMax, this->convertedIsoValueParam.Param<param::FloatParam>()->Value(), concMin, concMax);
			this->triggerConvertButtonParam.ResetDirty();
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());

	if (onlyVolumetric)
		vdc->Unlock();

	// read lighting parameters
	float lightPos[4];
	glGetLightfv(GL_LIGHT0, GL_POSITION, lightPos);
	float3 light = make_float3(lightPos[0], lightPos[1], lightPos[2]);

	// opengl parameters
	GLint depthfunc, matrixMode;
	glGetIntegerv(GL_DEPTH_FUNC, &depthfunc);
	auto depthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
	glGetIntegerv(GL_MATRIX_MODE, &matrixMode);

	render_kernel(gridSize, blockSize, cudaImage, cudaDepthImage, viewport.GetWidth(), viewport.GetHeight(), fovx, fovy, camPos, camDir, camUp, camRight, zNear, 
		density, brightness, transferOffset, transferScale, bbMin, bbMax, volumeExtent, light, make_float4(0.3f, 0.5f, 0.4f, 10.0f), plane, clipplaneAvailable);
	
	getLastCudaError("kernel failed");
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef FBO
	this->volume_fbo.Enable();
	GLfloat bk_colour[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bk_colour);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(bk_colour[0], bk_colour[1], bk_colour[2], bk_colour[3]);
	glPushMatrix();
	glLoadIdentity();

	// TODO disable stuff?
#endif

	GLint texID;
	glGetIntegerv(GL_ACTIVE_TEXTURE, &texID);
	glActiveTexture(GL_TEXTURE15);
	glBindTexture(GL_TEXTURE_2D, texHandle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport.GetWidth(), viewport.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, cudaImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glActiveTexture(GL_TEXTURE16);
	glBindTexture(GL_TEXTURE_2D, depthTexHandle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, viewport.GetWidth(), viewport.GetHeight(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, cudaDepthImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	//glDisable(GL_DEPTH_TEST);
	//glDepthFunc(GL_ALWAYS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	textureShader.Enable();
	
	glBindVertexArray(textureVAO);

	glUniform1f(textureShader.ParameterLocation("near"), zNear);
	glUniform1f(textureShader.ParameterLocation("far"), zFar);
	glUniform1i(textureShader.ParameterLocation("useDepth"), true);
	glUniform1i(textureShader.ParameterLocation("showDepth"), this->showDepthTextureParam.Param<param::BoolParam>()->Value());

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	glBindVertexArray(0);
	textureShader.Disable();

	glActiveTexture(texID);

	// restore opengl states
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	
	if (depthTestEnabled) {
		glEnable(GL_DEPTH_TEST);
	}
	glDepthFunc(depthfunc);
	glMatrixMode(matrixMode);

#ifdef FBO
	this->volume_fbo.Disable();
	this->volume_fbo.DrawColourTexture(0, GL_NEAREST, GL_NEAREST, 0.5);
#endif

	// parse selected isovalues if needed
	if (selectedIsovals.IsDirty() || firstTransfer) {
		isoVals.clear();
		vislib::TString valString = selectedIsovals.Param<param::StringParam>()->Value();
		vislib::StringA vala = T2A(valString);

		vislib::StringTokeniserA sta(vala, ',');
		while (sta.HasNext()) {
			vislib::StringA t = sta.Next();
			if (t.IsEmpty()) {
				continue;
			}
			isoVals.push_back((float)vislib::CharTraitsA::ParseDouble(t));
		}

		/*for (float f: isoVals)
		printf("value: %f\n", f);*/

		// sort the isovalues ascending
		std::sort(isoVals.begin(), isoVals.end());

		std::vector<float> adaptedIsoVals = isoVals;
		float div = isoVals[std::min((int)isoVals.size(), 4) - 1] - isoVals[0];

		float bla = 0.5f;

		if (!onlyVolumetric){
			// adapt the isovalues to the filtered values
			for (int i = 0; i < std::min((int)isoVals.size(), 4); i++) {
				adaptedIsoVals[i] = (adaptedIsoVals[i] - isoVals[0]) / div;
			}
		}

		// copy the first four isovalues into a float4
		std::vector<float> help(4, -1.0f);
		for (int i = 0; i < std::min((int)adaptedIsoVals.size(), 4); i++) {
			help[i] = isoVals[i];
		}
		float4 values = make_float4(help[0], help[1], help[2], help[3]);
		printf("isos: %f %f %f %f\n", help[0], help[1], help[2], help[3]);
		transferIsoValues(values, (int)std::min((int)isoVals.size(), 4));

		selectedIsovals.ResetDirty();
		if (firstTransfer) firstTransfer = false;
	}

	curTime++;
	lastTimeVal = myTime;
	recomputeVolume = false;

	return true;
}