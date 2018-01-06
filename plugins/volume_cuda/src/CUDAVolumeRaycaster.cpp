#include "stdafx.h"
#include "CUDAVolumeRaycaster.h"

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

using namespace megamol;
using namespace megamol::core;
using namespace megamol::volume_cuda;

/*
 *	CUDAVolumeRaycaster::CUDAVolumeRaycaster
 */
CUDAVolumeRaycaster::CUDAVolumeRaycaster(void) : core::view::Renderer3DModule(),
		volumeDataSlot("getData", "Connects the volume renderer with the volume data storage"),
		brightnessParam("brightness", "Scaling factor for the brightness of the image"),
		densityParam("density", "Scaling factor for the density of the volume"),
		lutFileParam("lut::lutfile", "File path to the file containing the lookup table"),
		lutSizeParam("lut::lutSize", "The number of components the lookup table should have. If a discrete LUT is loaded, this value is ignored.") {

	this->volumeDataSlot.SetCompatibleCall<misc::VolumeticDataCallDescription>();
	this->MakeSlotAvailable(&this->volumeDataSlot);

	this->brightnessParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->brightnessParam);

	this->densityParam.SetParameter(new param::FloatParam(0.05f, 0.0f));
	this->MakeSlotAvailable(&this->densityParam);

	this->lutFileParam.SetParameter(new param::FilePathParam(""));
	this->MakeSlotAvailable(&this->lutFileParam);

	this->lutSizeParam.SetParameter(new param::IntParam(256, 16, 2048));
	this->MakeSlotAvailable(&this->lutSizeParam);

	this->lastViewport.Set(0, 0);
	this->texHandle = 0; 
	this->setCUDAGLDevice = true;
	this->callTime = 0.0f;
	this->volumeExtent = make_cudaExtent(0, 0, 0);
	this->lastDataHash = 0;
	this->cudaImage = NULL;

#ifdef DEBUG_LUT
	const int lutSize = 256;
	float divisor = 255.0f;
	float4 lutMin = make_float4(59.0f, 76.0f, 192.0f, 0.0f);
	float4 lutMax = make_float4(180.0f, 4.0f, 38.0f, 255.0f);
	lutMin /= divisor;
	lutMax /= divisor;

	// lookup table for debugging
	this->lut.resize(lutSize);
	for (int i = 0; i < lutSize; i++) {
		float alpha = static_cast<float>(i) / static_cast<float>(lutSize);
		float4 val = lutMin * (1.0f - alpha) + lutMax * alpha;
		this->lut[i] = val;
	}
#endif // DEBUG_LUT
}

/*
 *	CUDAVolumeRaycaster::~CUDAVolumeRaycaster
 */
CUDAVolumeRaycaster::~CUDAVolumeRaycaster(void) {
	this->Release();
}

/*
 *	CUDAVolumeRaycaster::create
 */
bool CUDAVolumeRaycaster::create(void) {
	return initOpenGL();
}

/*
 *	CUDAVolumeRaycaster::release
 */
void CUDAVolumeRaycaster::release(void) {
	freeCudaBuffers();
}

/*
 *	CUDAVolumeRaycaster::GetCapabilities
 */
bool CUDAVolumeRaycaster::GetCapabilities(megamol::core::Call & call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL) return false;

	cr3d->SetCapabilities(view::CallRender3D::CAP_RENDER
		| view::CallRender3D::CAP_LIGHTING
		| view::CallRender3D::CAP_ANIMATION);

	return true;
}

/*
 *	CUDAVolumeRaycaster::GetExtents
 */
bool CUDAVolumeRaycaster::GetExtents(megamol::core::Call & call) {
	view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL) return false;

	misc::VolumetricDataCall *vdc = this->volumeDataSlot.CallAs<misc::VolumetricDataCall>();
	if (vdc == NULL) return false;

	if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;
	float scale;
	if (!vislib::math::IsEqual(vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
		scale = 2.0f / vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	}
	else {
		scale = 1.0f;
	}

	cr3d->AccessBoundingBoxes() = vdc->AccessBoundingBoxes();
	cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
	cr3d->SetTimeFramesCount(vdc->FrameCount());

	return true;
}

/*
 *	CUDAVolumeRaycaster::Render
 */
bool CUDAVolumeRaycaster::Render(megamol::core::Call & call) {
	view::CallRender3D * cr3d = dynamic_cast<view::CallRender3D *>(&call);
	if (cr3d == NULL)return false;

	this->callTime = cr3d->Time();
	int myTime = static_cast<int>(callTime);
	
	misc::VolumetricDataCall *vdc = this->volumeDataSlot.CallAs<misc::VolumetricDataCall>();
	if (vdc == NULL) return false;

	vdc->SetFrameID(myTime);
	if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;
	if (!(*vdc)(vdc->IDX_GET_METADATA)) return false;
	if (!(*vdc)(vdc->IDX_GET_DATA)) return false;

	glPushMatrix();
	float scale = 1.0f;
	if (!vislib::math::IsEqual(vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
		scale = 2.0f / vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	}
	glScalef(scale, scale, scale);

	initCuda(*cr3d);
	initPixelBuffer(*cr3d);

	// get all relevant parameters
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
	float3 camDir = norm(make_float3(-modelMatrix.GetAt(0, 2), -modelMatrix.GetAt(1, 2), -modelMatrix.GetAt(2, 2)));
	float3 camUp = norm(make_float3(modelMatrix.GetAt(0, 1), modelMatrix.GetAt(1, 1), modelMatrix.GetAt(2, 1)));
	float3 camRight = norm(make_float3(modelMatrix.GetAt(0, 0), modelMatrix.GetAt(1, 0), modelMatrix.GetAt(2, 0)));
	// the direction has to be negated because of the right-handed view space of OpenGL

	auto cam = cr3d->GetCameraParameters();

	float fovy = (float)(cam->ApertureAngle() * M_PI / 180.0f);

	float aspect = (float)viewport.GetWidth() / (float)viewport.GetHeight();
	if (viewport.GetHeight() == 0)
		aspect = 0.0f;

	float fovx = 2.0f * atan(tan(fovy / 2.0f) * aspect);
	float zNear = (2.0f * projectionMatrix.GetAt(2, 3)) / (2.0f * projectionMatrix.GetAt(2, 2) - 2.0f);

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

	auto bb = vdc->GetBoundingBoxes().ObjectSpaceBBox();
	float3 bbMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
	float3 bbMax = make_float3(bb.Right(), bb.Top(), bb.Front());

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
			transferNewVolume(volPtr, volumeExtent);
			checkCudaErrors(cudaDeviceSynchronize());
		}
		else {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "The transferred volume is empty");
			return false;
		}
	}

	if (loadLut()) {
		copyTransferFunction(this->lut.data(), static_cast<int>(this->lut.size()));
	}

	vdc->Unlock();

	// render the stuff
	render_kernel(gridSize, blockSize, this->cudaImage, viewport.GetWidth(), viewport.GetHeight(), fovx, fovy, camPos, camDir, camUp, camRight, zNear, density, brightness,
		transferOffset, transferScale, bbMin, bbMax, this->volumeExtent);
	getLastCudaError("kernel failed");
	checkCudaErrors(cudaDeviceSynchronize());

	/*for (int i = 0; i < viewport.GetWidth() * viewport.GetHeight(); i++) {
		if (i % viewport.GetWidth() == 0)
			printf("\n");

		printf("%u ", this->cudaImage[i]);
	}*/

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
 *	CUDAVolumeRaycaster::initCuda
 */
bool CUDAVolumeRaycaster::initCuda(megamol::core::view::CallRender3D & cr3d) {
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
 *	CUDAVolumeRaycaster::initOpenGL
 */
bool CUDAVolumeRaycaster::initOpenGL() {
	
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

	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("cudavolumeraycaster::texture::textureFragment", fragSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for texture shader");
		return false;
	}

	this->textureShader.Compile(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count());
	this->textureShader.Link();

	return true;
}

/*
 *	CUDAVolumeRaycaster::initPixelBuffer
 */
bool CUDAVolumeRaycaster::initPixelBuffer(megamol::core::view::CallRender3D & cr3d) {
	
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

	checkCudaErrors(cudaMallocHost((void **)&this->cudaImage, viewport.GetWidth() * viewport.GetHeight() * sizeof(unsigned int)));

	return true;
}

/*
 *	CUDAVolumeRaycaster::loadVolume
 */
void * CUDAVolumeRaycaster::loadVolume(misc::VolumetricDataCall * vdc) {
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
 *	CUDAVolumeRaycaster::loadLut
 */
bool CUDAVolumeRaycaster::loadLut(void) {
	if (!this->lutFileParam.IsDirty() && !this->lutSizeParam.IsDirty()) return false;
	this->lutFileParam.ResetDirty();
	this->lutSizeParam.ResetDirty();
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
		if (discrete) {
			this->lut.clear();
			for (size_t i = 0; i < values.size(); i += 4) {
				this->lut.push_back(make_float4(values[i], values[i + 1], values[i + 2], values[i + 3]));
			}
		} else {
			this->lut.clear();
			size_t lutSize = static_cast<size_t>(this->lutSizeParam.Param<param::IntParam>()->Value());
			this->lut.resize(lutSize, make_float4(-1.0f));
			size_t validVals = 0;
			for (size_t i = 0; i < values.size(); i += 5) {
				// determine bin of the current value
				size_t bin = static_cast<size_t>(values[i] * (lutSize - 1));
				if (bin >= this->lut.size()) {
					vislib::sys::Log::DefaultLog.WriteWarn("Lut point of line %u is malformed, ignoring this value.", static_cast<uint>(i + 2));
					continue;
				}
				validVals++;
				this->lut[bin] = make_float4(values[i + 1], values[i + 2], values[i + 3], values[i + 4]);
			}
			// at this point only bins with control points in them have values >= 0
			if (validVals == 0) {
				for (auto& v : this->lut) { // when there are no
					v = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
				}
				return true;
			}
			// extend the first and the last read value to beginning and end
			// find the first one
			size_t index;
			for (index = 0; index < this->lut.size(); index++) {
				if (this->lut[index].w >= 0.0f) break;
			}
			if (index != this->lut.size()) {
				for (size_t i = 0; i < index; i++) {
					this->lut[i] = this->lut[index];
				}
			} else {
				vislib::sys::Log::DefaultLog.WriteError("Something went wrong with the LUT reconstruction (1)");
				this->lut = lutSave;
				return false;
			}
			// find the last one
			for (index = this->lut.size() - 1; index < this->lut.size(); index--) { // the buffer overflow at this point is intentional
				if (this->lut[index].w >= 0.0f) break;
			}
			if (index < this->lut.size()) {
				for (size_t i = index + 1; i < this->lut.size(); i++) {
					this->lut[i] = this->lut[index];
				}
			} else {
				vislib::sys::Log::DefaultLog.WriteError("Something went wrong with the LUT reconstruction (2)");
				this->lut = lutSave;
				return false;
			}
			// at this point the first and the last value should have been extended to the boundaries.
			// now handle the interpolation between the values
			std::vector<std::pair<size_t, size_t>> interpolationRanges;
			std::pair<size_t, size_t> current;
			bool detected = false;
			// determine the ranges in which we want to interpolate
			for (size_t i = 0; i < this->lut.size(); i++) {
				if (this->lut[i].w < 0.0f) {
					detected = true;
				}
				if (detected && this->lut[i].w >= 0.0f) {
					current.second = i;
					detected = false;
					interpolationRanges.push_back(current);
				}
				if (!detected && this->lut[i].w >= 0.0f) {
					current.first = i;
				}
			}
			// perform the interpolation
			for (auto r : interpolationRanges) {
				auto c1 = this->lut[r.first];
				auto c2 = this->lut[r.second];
				vislib::math::Vector<float, 4> colorStart(c1.x, c1.y, c1.z, c1.w);
				vislib::math::Vector<float, 4> colorEnd(c2.x, c2.y, c2.z, c2.w);
				float length = static_cast<float>(r.second - r.first);
				float step = 1.0f / length;
				float val = step;
				for (size_t i = r.first + 1; i < r.second; i++) {
					auto cc = (1.0f - val) * colorStart + val * colorEnd;
					this->lut[i] = make_float4(cc.GetX(), cc.GetY(), cc.GetZ(), cc.GetW());
					val += step;
				}
			}
		}
	} else {
		vislib::sys::Log::DefaultLog.WriteError("The lookup file could not be opened. No new table loaded");
		return false;
	}
	return true;
}

/*
 *	CUDAVolumeRaycaster::splitStringByCharacter
 */
std::vector<std::string> CUDAVolumeRaycaster::splitStringByCharacter(std::string text, char character) {
	std::stringstream stream(text);
	std::string segment;
	std::vector<std::string> result;
	while (std::getline(stream, segment, character)) {
		result.push_back(segment);
	}
	return result;
}