#ifndef VOLUME_CUDA_CUDAVOLUMERAYCASTER_H_INCLUDED
#define VOLUME_CUDA_CUDAVOLUMERAYCASTER_H_INCLUDED

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/misc/VolumetricDataCall.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Matrix.h"
typedef vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> Mat3f;
typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4f;

#include "helper_math.h"

#define DEBUG_LUT

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void freeCudaBuffers(void);
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint * d_output, uint imageW, uint imageH, float fovx, float fovy, float3 camPos, float3 camDir,
	float3 camUp, float3 camRight, float zNear, float density, float brightness, float transferOffset, float transferScale,
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f), const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f), cudaExtent volSize = make_cudaExtent(1, 1, 1));
extern "C" void copyTransferFunction(float4 * transferFunction, int functionSize = 256);
extern "C" void transferNewVolume(void * h_volume, cudaExtent volumeSize);
extern "C" void initCudaDevice(void * h_volume, cudaExtent volumeSize, float4 * transferFunction, int functionSize = 256);

namespace megamol {
namespace volume_cuda {
	
	class CUDAVolumeRaycaster : public megamol::core::view::Renderer3DModule {
	public:
		
		/**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
		static const char *ClassName(void) {
			return "CUDAVolumeRaycaster";
		}

		/**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers raycasting of volumes";
        }

		/**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

		/** Ctor */
		CUDAVolumeRaycaster(void);

		/** Dtor */
		virtual ~CUDAVolumeRaycaster(void);

	protected:

		/**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
		virtual bool create(void);

		/**
         * Implementation of 'release'.
         */
		virtual void release(void);

	private:

		/*
		 *	Struct representing a vertex with a position and a color.
		 */
		struct Vertex
		{
			// The vertex position.
			GLfloat x, y, z, w;

			// The vertex color.
			GLfloat r, g, b, a;

			Vertex(GLfloat x, GLfloat y, GLfloat z, GLfloat r, GLfloat g, GLfloat b, GLfloat a) :
				x(x), y(y), z(z), w(1.0f), r(r), g(g), b(b), a(a) {}

			Vertex(GLfloat x, GLfloat y, GLfloat z, GLfloat w, GLfloat r, GLfloat g, GLfloat b, GLfloat a) :
				x(x), y(y), z(z), w(w), r(r), g(g), b(b), a(a) {}
		};

		VISLIB_FORCEINLINE int iDivUp(int a, int b) {
			return (a % b != 0) ? (a / b + 1) : (a / b);
		}

		/**
		 *	Computes the length of the given vector
		 *
		 *	@param a The given vector
		 *	@return The length of the given vector
		 */
		VISLIB_FORCEINLINE float3 norm(float3 a) {
			float scale = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
			return make_float3(a.x / scale, a.y / scale, a.z / scale);
		}

		/**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return True on success, false otherwise.
         */
        virtual bool GetCapabilities(megamol::core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return True on success, false otherwise.
         */
        virtual bool GetExtents(megamol::core::Call& call);

		/**
         * The Open GL Render callback.
         *
         * @param call The calling call.
         * @return True on success, false otherwise.
         */
        virtual bool Render(megamol::core::Call& call);

		/**
		 *	Initializes all CUDA related stuff
		 *	
		 *	@param cr3d The calling call.
		 *	@return True on success, false otherwise.
		 */
		bool initCuda(megamol::core::view::CallRender3D& cr3d);

		/**
		 *	Initializes all OpenGL related structures
		 *
		 *	@return True on success, false otherwise.
		 */
		bool initOpenGL();

		/**
		 *	Initializes the pixel buffer and other related structures
		 *
		 *	@return True on success, false otherwise.
		 */
		bool initPixelBuffer(megamol::core::view::CallRender3D& cr3d);

		/**
		 * Loads the volume data. If the volume consists of float data, the original pointer is returned.
		 * If the volume has other data types the data is converted to float and a pointer to the converted
		 * data is returned.
		 * 
		 * @param vdc The call containing the volume data.
		 * @return pointer to the volume data.
		 */
		void * loadVolume(megamol::core::misc::VolumetricDataCall* vdc);

		/** caller slot */
		megamol::core::CallerSlot volumeDataSlot;

		/** parameter steering the brightness of the image */
		megamol::core::param::ParamSlot brightnessParam;

		/** parameter scaling the density of the volume */
		megamol::core::param::ParamSlot densityParam;

		/** the viewport dimensions of the last frame */
		vislib::math::Dimension<int, 2U> lastViewport;

		/** the texture handle */
		GLuint texHandle;

		/** the resulting CUDA image */
		unsigned int * cudaImage;

		/** the shader program for texture drawing */
		vislib::graphics::gl::GLSLShader textureShader;

		/** vertex array for the texture drawing vertices */
		GLuint textureVAO;

		/** vertex buffer for the texture drawing vertices */
		GLuint textureVBO;

		/** has the CUDA GL device to be set? */
		bool setCUDAGLDevice;

		/** the curren time */
		float callTime;

		/** the size of the volume */
		cudaExtent volumeExtent;

		/** the data hash of the recently loaded data */
		SIZE_T lastDataHash;

		/** the float volume if the data type of the incoming data is not float */
		std::vector<float> localVolume;

#ifdef DEBUG_LUT
		/** the lookup table */
		std::vector<float4> lut;
#endif // DEBUG_LUT
	};

} /* namespace volume_cuda */
} /* namespace megamol */

#endif // !VOLUME_CUDA_CUDAVOLUMERAYCASTER_H_INCLUDED