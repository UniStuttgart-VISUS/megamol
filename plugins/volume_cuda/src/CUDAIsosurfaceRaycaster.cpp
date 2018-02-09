#include "stdafx.h"
#include "CUDAIsosurfaceRaycaster.h"

#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/CoreInstance.h"

#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

#include "cuda.h"
#include "cuda_gl_interop.h"
#include "helper_cuda.h"

#include <fstream>
#include <sstream>
#include <limits>
#include <array>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::volume_cuda;

/*
 *	CUDAIsosurfaceRaycaster::CUDAIsosurfaceRaycaster
 */
CUDAIsosurfaceRaycaster::CUDAIsosurfaceRaycaster(void) : core::view::Renderer3DModule(),
		volumeDataSlot("getData", "Connects the volume renderer with the volume data storage"),
		inputImageSlot("receiveImage", "Connects the volume renderer with another renderer to overlay both images"),
		brightnessParam("brightness", "Scaling factor for the brightness of the image"),
		densityParam("density", "Scaling factor for the density of the volume"),
		lutFileParam("lut::lutfile", "File path to the file containing the lookup table") {

	this->volumeDataSlot.SetCompatibleCall<misc::VolumeticDataCallDescription>();
	this->MakeSlotAvailable(&this->volumeDataSlot);

	this->inputImageSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
	this->MakeSlotAvailable(&this->inputImageSlot);

	this->brightnessParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->brightnessParam);

	this->densityParam.SetParameter(new param::FloatParam(0.05f, 0.0f));
	this->MakeSlotAvailable(&this->densityParam);

	this->lutFileParam.SetParameter(new param::FilePathParam(""));
	this->MakeSlotAvailable(&this->lutFileParam);

	this->lastViewport.Set(0, 0);
	this->texHandle = 0; 
	this->setCUDAGLDevice = true;
	this->callTime = 0.0f;
	this->volumeExtent = make_cudaExtent(0, 0, 0);
	this->lastDataHash = 0;
	this->cudaImage = NULL;
	this->cudaDepthImage = NULL;

    this->cuda_kernels = std::unique_ptr<CUDAIsosurfaceRaycaster_kernel>(new CUDAIsosurfaceRaycaster_kernel());
}

/*
 *	CUDAIsosurfaceRaycaster::~CUDAIsosurfaceRaycaster
 */
CUDAIsosurfaceRaycaster::~CUDAIsosurfaceRaycaster(void) {
	this->Release();
}

/*
 *	CUDAIsosurfaceRaycaster::create
 */
bool CUDAIsosurfaceRaycaster::create(void) {
	return initOpenGL();
}

/*
 *	CUDAIsosurfaceRaycaster::release
 */
void CUDAIsosurfaceRaycaster::release(void) {
	cuda_kernels->freeCudaBuffers();
}

/*
 *	CUDAIsosurfaceRaycaster::GetCapabilities
 */
bool CUDAIsosurfaceRaycaster::GetCapabilities(megamol::core::Call & call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL) return false;

	cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER
		| view::CallRender3D::CAP_LIGHTING
		| view::CallRender3D::CAP_ANIMATION);

	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::GetExtents
 */
bool CUDAIsosurfaceRaycaster::GetExtents(megamol::core::Call & call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL) return false;

	misc::VolumetricDataCall *vdc = this->volumeDataSlot.CallAs<misc::VolumetricDataCall>();
	if (vdc == NULL) return false;

	view::CallRender3D * incCrd = this->inputImageSlot.CallAs<view::CallRender3D>();

	if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;

	auto volumeBB = vdc->AccessBoundingBoxes().ObjectSpaceBBox();
	auto fcnt = vdc->FrameCount();

	if (incCrd != NULL) {
		if (!(*incCrd)(1)) return false; // get extents
		volumeBB.Union(incCrd->AccessBoundingBoxes().ObjectSpaceBBox());
		fcnt = std::min(fcnt, incCrd->TimeFramesCount());
	}

	float scale;
	if (!vislib::math::IsEqual(volumeBB.LongestEdge(), 0.0f)) {
		scale = 2.0f / volumeBB.LongestEdge();
	} else {
		scale = 1.0f;
	}

	// with two incoming bounding boxes we have to take the already computed one as reference for the scale factor,
	// since we cannot change the rendering done by external renderers
	if (incCrd != NULL) {
		if (!vislib::math::IsEqual(incCrd->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / incCrd->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		} else {
			scale = 1.0f;
		}
	}

	cr3d->AccessBoundingBoxes().Clear();
	cr3d->AccessBoundingBoxes().SetObjectSpaceBBox(volumeBB);
	cr3d->AccessBoundingBoxes().SetObjectSpaceClipBox(volumeBB);
	cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
	cr3d->SetTimeFramesCount(fcnt);

	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::Render
 */
bool CUDAIsosurfaceRaycaster::Render(megamol::core::Call & call) {
	view::CallRender3D * cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL)return false;

	this->callTime = cr3d->Time();
	int myTime = static_cast<int>(callTime);
	
	misc::VolumetricDataCall *vdc = this->volumeDataSlot.CallAs<misc::VolumetricDataCall>();
	if (vdc == NULL) return false;

	view::CallRender3D * incCrd = this->inputImageSlot.CallAs<view::CallRender3D>();

	vdc->SetFrameID(myTime);
	if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;
	if (!(*vdc)(vdc->IDX_GET_METADATA)) return false;
	if (!(*vdc)(vdc->IDX_GET_DATA)) return false;

	float scale = 1.0f;
	auto bb = vdc->AccessBoundingBoxes().ObjectSpaceBBox();

	if (incCrd != NULL) {
		if (!(*incCrd)(1)) return false; // get extents
		bb.Union(incCrd->AccessBoundingBoxes().ObjectSpaceBBox());
	}

	if (!vislib::math::IsEqual(bb.LongestEdge(), 0.0f)) {
		scale = 2.0f / bb.LongestEdge();
	}

	// with two incoming bounding boxes we have to take the already computed one as reference for the scale factor,
	// since we cannot change the rendering done by external renderers
	if (incCrd != NULL) {
		if (!vislib::math::IsEqual(incCrd->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
			scale = 2.0f / incCrd->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
		}
		else {
			scale = 1.0f;
		}
	}

	initCuda(*cr3d);
	initPixelBuffer(*cr3d);

	// get all relevant parameters
	auto viewport = cr3d->GetViewport().GetSize();

	this->renderCallToFBO(incCrd, cr3d, viewport);

	if (incCrd == nullptr) {
		glPushMatrix();
		glScalef(scale, scale, scale);
	}

	GLfloat m[16];
	GLfloat m_proj[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glGetFloatv(GL_PROJECTION_MATRIX, m_proj);
	Mat4f modelMatrix(&m[0]);
	Mat4f projectionMatrix(&m_proj[0]);
	modelMatrix.Invert();
	projectionMatrix.Invert();
	float3 camPos = make_float3(modelMatrix.GetAt(0, 3), modelMatrix.GetAt(1, 3), modelMatrix.GetAt(2, 3));
	float3 camDir = norm(make_float3(-modelMatrix.GetAt(0, 2), -modelMatrix.GetAt(1, 2), -modelMatrix.GetAt(2, 2)));
	float3 camUp = norm(make_float3(modelMatrix.GetAt(0, 1), modelMatrix.GetAt(1, 1), modelMatrix.GetAt(2, 1)));
	float3 camRight = norm(make_float3(modelMatrix.GetAt(0, 0), modelMatrix.GetAt(1, 0), modelMatrix.GetAt(2, 0)));
	// the direction has to be negated because of the right-handed view space of OpenGL

	auto cam = cr3d->GetCameraParameters();
	auto z1 = cam->NearClip();
	auto z2 = cam->FarClip();

	if (incCrd != nullptr) {
		glPushMatrix();
		glScalef(scale, scale, scale);
	}

	float fovy = (float)(cam->ApertureAngle() * M_PI / 180.0f);

	float aspect = (float)viewport.GetWidth() / (float)viewport.GetHeight();
	if (viewport.GetHeight() == 0)
		aspect = 0.0f;

	float fovx = 2.0f * atan(tan(fovy / 2.0f) * aspect);
	float zNear = (2.0f * projectionMatrix.GetAt(2, 3)) / (2.0f * projectionMatrix.GetAt(2, 2) - 2.0f);
	float zFar = 1.0f; // UNKNOWN how to do this

	// new plane computation (hopefully correct)
	zNear = cam->NearClip();
	zFar = cam->FarClip();

	float density = this->densityParam.Param<param::FloatParam>()->Value();
	float brightness = this->brightnessParam.Param<param::FloatParam>()->Value();
	float transferOffset = 0.0f;
	float transferScale = 1.0f;

	dim3 blockSize = dim3(8, 8);
	dim3 gridSize = dim3(iDivUp(viewport.GetWidth(), blockSize.x), iDivUp(viewport.GetHeight(), blockSize.y));

	// get the needed values from the volume
	auto xDir = vdc->GetResolution(0);
	auto yDir = vdc->GetResolution(1);
	auto zDir = vdc->GetResolution(2);
	
	this->volumeExtent = make_cudaExtent(xDir, yDir, zDir);

	auto vbb = vdc->GetBoundingBoxes().ObjectSpaceBBox();
	float3 bbMin = make_float3(vbb.Left(), vbb.Bottom(), vbb.Back());
	float3 bbMax = make_float3(vbb.Right(), vbb.Top(), vbb.Front());

	if (incCrd != nullptr) {
		bbMin *= scale;
		bbMax *= scale;
	}

	if (vdc->GetComponents() > 1 || vdc->GetComponents() < 1) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Only volumes with a single component are currently supported");
		return false;
	}

	if (vdc->GetGridType() != misc::VolumetricDataCall::GridType::CARTESIAN) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Only cartesian grids are currently supported");
		return false;
	}

	if (vdc->DataHash() != this->lastDataHash) {
		this->lastDataHash = vdc->DataHash();
		auto volPtr = loadVolume(vdc);

		if (volPtr != nullptr) {
            cuda_kernels->transferNewVolume(volPtr, volumeExtent);
			checkCudaErrors(cudaDeviceSynchronize());
		}
		else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "The transferred volume is empty");
			return false;
		}
	}

	if (loadLut()) {
        std::vector<float> isoValues(this->lut.size());
        std::vector<float4> colors(this->lut.size());
        
        for (size_t i = 0; i < this->lut.size(); i++) {
            isoValues[i] = this->lut[i].first;
            colors[i] = this->lut[i].second;
        }

        cuda_kernels->copyColorValues(colors.data(), static_cast<int>(colors.size()));
        cuda_kernels->setIsoValues(isoValues, isoValues.size());
	}

	vdc->Unlock();

	glActiveTexture(GL_TEXTURE13);
	if (incCrd != nullptr) {
		// read the depth texture values if available
		this->copyFBO.BindDepthTexture();
		this->copyFBO.GetDepthTexture(this->cudaDepthImage, GL_DEPTH_COMPONENT, GL_FLOAT);
	} else {
		for (size_t i = 0; i < viewport.GetHeight() * viewport.GetWidth(); i++) {
			this->cudaDepthImage[i] = 1.0f;
		}
	}

	// render the stuff
    cuda_kernels->render_kernel(gridSize, blockSize, this->cudaImage, this->cudaDepthImage, viewport.GetWidth(), viewport.GetHeight(), fovx, fovy, camPos, camDir, camUp, camRight, 
        zNear, zFar, density, brightness, transferOffset, transferScale, bbMin, bbMax, this->volumeExtent);
	getLastCudaError("kernel failed");
	checkCudaErrors(cudaDeviceSynchronize());

	glActiveTexture(GL_TEXTURE14);
	if (incCrd != nullptr) {
		this->copyFBO.BindColourTexture();
	} else {
		// generate texture containing the background color
		this->setupBackgroundTexture();
		glBindTexture(GL_TEXTURE_2D, this->bgTexHandle);
	}

	glActiveTexture(GL_TEXTURE15);
	glBindTexture(GL_TEXTURE_2D, this->texHandle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport.GetWidth(), viewport.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, this->cudaImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_ALWAYS);

	textureShader.Enable();
	
	glBindVertexArray(this->textureVAO);
	
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	
	glBindVertexArray(0);
	textureShader.Disable();
	
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::initCuda
 */
bool CUDAIsosurfaceRaycaster::initCuda(megamol::core::view::CallRender3D & cr3d) {
	if (this->setCUDAGLDevice) {
#ifdef _WIN32
		if (cr3d.IsGpuAffinity()) {
			HGPUNV gpuId = cr3d.GpuAffinity<HGPUNV>();
			int devId;
			checkCudaErrors(cudaWGLGetDevice(&devId, gpuId));
			checkCudaErrors(cudaGLSetGLDevice(devId));
		} else {
			checkCudaErrors(cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()));
		}
#else
		checkCudaErrors(cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()));
#endif
		this->setCUDAGLDevice = false;
	}

	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::initOpenGL
 */
bool CUDAIsosurfaceRaycaster::initOpenGL() {
	
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

	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("cudavolumeraycaster::texture::textureVertex", vertSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for texture shader");
		return false;
	}

	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("cudavolumeraycaster::texture::combineFragment", fragSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for texture shader");
		return false;
	}

	this->textureShader.Compile(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());
	this->textureShader.Link();

	if (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) return false;

	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::initPixelBuffer
 */
void CUDAIsosurfaceRaycaster::setupBackgroundTexture(void) {
	
	std::array<float, 4> bgcolor = { 0.0f, 0.0f, 0.0f, 0.0f };
	glGetFloatv(GL_COLOR_CLEAR_VALUE, &bgcolor[0]);
	
	if(!this->bgTexHandle) {
		glGenTextures(1, &this->bgTexHandle);
	}

	glBindTexture(GL_TEXTURE_2D, this->bgTexHandle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_FLOAT, bgcolor.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
}

/*
 *	CUDAIsosurfaceRaycaster::initPixelBuffer
 */
bool CUDAIsosurfaceRaycaster::initPixelBuffer(megamol::core::view::CallRender3D & cr3d) {
	
	auto viewport = cr3d.GetViewport().GetSize();

	if (this->lastViewport == viewport) {
		return true;
	} else {
		this->lastViewport = viewport;
	}

	if (!this->texHandle) {
		glGenTextures(1, &this->texHandle);
		glActiveTexture(GL_TEXTURE15);
		glBindTexture(GL_TEXTURE_2D, this->texHandle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport.GetWidth(), viewport.GetHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	if (this->cudaImage) {
		checkCudaErrors(cudaFreeHost(this->cudaImage));
		cudaImage = NULL;
	}

	if (this->cudaDepthImage) {
		checkCudaErrors(cudaFreeHost(this->cudaDepthImage));
		cudaDepthImage = NULL;
	}

	checkCudaErrors(cudaMallocHost((void **)&this->cudaImage, viewport.GetWidth() * viewport.GetHeight() * sizeof(unsigned int)));
	checkCudaErrors(cudaMallocHost((void **)&this->cudaDepthImage, viewport.GetWidth() * viewport.GetHeight() * sizeof(float)));

	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::loadVolume
 */
void * CUDAIsosurfaceRaycaster::loadVolume(misc::VolumetricDataCall * vdc) {
	auto volPtr = vdc->GetData();
	this->localVolume.clear();
	if (volPtr == nullptr) return nullptr;

	auto numValues = vdc->GetResolution(0) * vdc->GetResolution(1) * vdc->GetResolution(2);
	auto length = vdc->GetScalarLength();

	switch (vdc->GetScalarType()) {
		case misc::VolumetricDataCall::ScalarType::BITS:
			return nullptr;
		case misc::VolumetricDataCall::ScalarType::FLOATING_POINT:
			return volPtr;
		case misc::VolumetricDataCall::ScalarType::SIGNED_INTEGER:
		{
			this->localVolume.resize(numValues, 0.0f);
			switch (length) {
				case 1:
				{
					auto volPtrInt = reinterpret_cast<int8_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrInt[i]);
					}
					return this->localVolume.data();
				}
				case 2:
				{
					auto volPtrInt = reinterpret_cast<int16_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrInt[i]);
					}
					return this->localVolume.data();
				}
				case 4:
				{
					auto volPtrInt = reinterpret_cast<int32_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrInt[i]);
					}
					return this->localVolume.data();
				}
				case 8:
				{
					auto volPtrInt = reinterpret_cast<int64_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrInt[i]);
					}
					return this->localVolume.data();
				}
				default:
					return nullptr;
			}
		}
		case misc::VolumetricDataCall::ScalarType::UNSIGNED_INTEGER:
		{
			this->localVolume.resize(numValues, 0.0f);
			switch (length) {
				case 1:
				{
					auto volPtrUint = reinterpret_cast<uint8_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrUint[i]);
					}
					return this->localVolume.data();
				}
				case 2:
				{
					auto volPtrUint = reinterpret_cast<uint16_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrUint[i]);
					}
					return this->localVolume.data();
				}
				case 4:
				{
					auto volPtrUint = reinterpret_cast<uint32_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrUint[i]);
					}
					return this->localVolume.data();
				}
				case 8:
				{
					auto volPtrUint = reinterpret_cast<uint64_t *>(volPtr);
					for (size_t i = 0; i < numValues; i++) {
						this->localVolume[i] = static_cast<float>(volPtrUint[i]);
					}
					return this->localVolume.data();
				}
				default:
					return nullptr;
			}
		}
		default:
			return nullptr;
	}
}

/*
 *	CUDAIsosurfaceRaycaster::loadLut
 */
bool CUDAIsosurfaceRaycaster::loadLut(void) {
	if (!this->lutFileParam.IsDirty()) return false;
	this->lutFileParam.ResetDirty();
	auto lutSave = this->lut;
	auto path = this->lutFileParam.Param<param::FilePathParam>()->Value();
	if (path.IsEmpty()) return false;
	std::ifstream file;
	file.open(path.PeekBuffer());

	if (file.is_open()) {
		std::string line;
		bool discrete = false;
		if (std::getline(file, line)) {
			if (!line.compare("DISCRETE")) {
				discrete = true;
			} else if (!line.compare("POINTS")) {
				discrete = false;
			} else {
				vislib::sys::Log::DefaultLog.WriteError("Unrecognized lookup file type. No new table loaded");
				return false;
			}
		}

        if (discrete) {
            vislib::sys::Log::DefaultLog.WriteError("Discrete file version not possible for isosurfaces");
            return false;
        }

		std::vector<float> values;
		size_t row = 1;
		// read all the values
		while (std::getline(file, line)) {
			++row;
			if (line.empty()) continue;
			auto splitText = splitStringByCharacter(line, ',');
			if (discrete && splitText.size() < 4) {
				vislib::sys::Log::DefaultLog.WriteError("Error at line %u: A discrete lookup table file needs at least 4 values per row", static_cast<uint>(row));
				return false;
			} else if (!discrete && splitText.size() < 5) {
				vislib::sys::Log::DefaultLog.WriteError("Error at line %u: A point-based lookup table file needs at least 5 values per row", static_cast<uint>(row));
				return false;
			}
			size_t nv = 4;
			if (!discrete) nv = 5;
			for (size_t i = 0; i < nv; i++) {
				values.push_back(static_cast<float>(std::atof(splitText[i].c_str())));
			}
		}
		// process the read values
		this->lut.clear();
		size_t validVals = 0;
		for (size_t i = 0; i < values.size(); i += 5) {
            auto pair = std::make_pair(values[i], make_float4(values[i + 1], values[i + 2], values[i + 3], values[i + 4]));
            this->lut.push_back(pair);
		}
		
		
	} else {
		vislib::sys::Log::DefaultLog.WriteError("The lookup file could not be opened. No new table loaded");
		return false;
	}
	return true;
}

/*
 *	CUDAIsosurfaceRaycaster::splitStringByCharacter
 */
std::vector<std::string> CUDAIsosurfaceRaycaster::splitStringByCharacter(std::string text, char character) {
	std::stringstream stream(text);
	std::string segment;
	std::vector<std::string> result;
	while (std::getline(stream, segment, character)) {
		result.push_back(segment);
	}
	return result;
}

/*
 *	CUDAIsosurfaceRaycaster::renderCallToFBO
 */
bool CUDAIsosurfaceRaycaster::renderCallToFBO(view::CallRender3D * cr3d, view::CallRender3D * incoming, vislib::math::Dimension<float, 2> viewport) {
	if (cr3d == nullptr) return false;
	
	if (!this->copyFBO.IsValid() || this->copyFBO.GetWidth() != viewport.GetWidth() || this->copyFBO.GetHeight() != viewport.GetHeight()) {
		if (!this->copyFBO.Create(static_cast<UINT>(viewport.GetWidth()), static_cast<UINT>(viewport.GetHeight()), GL_RGBA32F, GL_RGBA, GL_FLOAT,
			vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE)) {
			vislib::sys::Log::DefaultLog.WriteError("Unable to create the framebuffer for the copy step");
			return false;
		}
	}
	incoming->DisableOutputBuffer();
	cr3d->operator=(*incoming);
	this->copyFBO.Enable();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
	cr3d->SetOutputBuffer(&this->copyFBO);
	cr3d->EnableOutputBuffer();
	(*cr3d)(); // render call
	cr3d->DisableOutputBuffer();
	glPopMatrix();
	this->copyFBO.Disable();
	incoming->EnableOutputBuffer();
	return true;
}