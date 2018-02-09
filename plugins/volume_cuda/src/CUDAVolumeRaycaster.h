#ifndef VOLUME_CUDA_CUDAVOLUMERAYCASTER_H_INCLUDED
#define VOLUME_CUDA_CUDAVOLUMERAYCASTER_H_INCLUDED

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/misc/VolumetricDataCall.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/math/Matrix.h"
typedef vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> Mat3f;
typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4f;

#include "helper_math.h"
#include "CUDAVolumeRaycaster_kernel.cuh"

#define DEBUG_LUT

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

		/**
		 * Loads the lookup table from file
		 *
		 * @return True, if a new lookup table is available, false otherwise
		 */
		bool loadLut(void);

		/**
		 * Splits a given string by a character and returns the parts in a vector.
		 *
		 * @param text The text to split.
		 * @param character The character to split by.
		 * @return A list of substrings of the original text.
		 */
		std::vector<std::string> splitStringByCharacter(std::string text, char character = ',');

		/**
		 * Sends out a render call over the givne callRender3D and renders
		 * the result into a fbo.
		 * 
		 * @param cr3d Pointer to the call the rendering request is sent over.
		 * @param incoming Pointer to the incoming render call that will be copied to the outgoing.
		 * @param viewport The viewport that should be rendered
		 * @return True if the rendering was completed succesfully. False if no image is available.
		 */
		bool renderCallToFBO(core::view::CallRender3D * cr3d, core::view::CallRender3D * incoming, vislib::math::Dimension<float, 2> viewport);

		/**
		 * Sets up the background texture
		 */
		void setupBackgroundTexture(void);

		/** caller slot */
		megamol::core::CallerSlot volumeDataSlot;

		/** caller slot for the input image */
		megamol::core::CallerSlot inputImageSlot;

		/** parameter steering the brightness of the image */
		megamol::core::param::ParamSlot brightnessParam;

		/** parameter scaling the density of the volume */
		megamol::core::param::ParamSlot densityParam;

		/** parameter for the file path to the file containing the lookup table */
		megamol::core::param::ParamSlot lutFileParam;

		/** parameter for the number of components of the discretized lookup table */
		megamol::core::param::ParamSlot lutSizeParam;

		/** the viewport dimensions of the last frame */
		vislib::math::Dimension<int, 2U> lastViewport;

		/** fbo holding the old images */
		vislib::graphics::gl::FramebufferObject copyFBO;

		/** the texture handle */
		GLuint texHandle;

		/** the texture handle for the background */
		GLuint bgTexHandle;

		/** the resulting CUDA image */
		unsigned int * cudaImage;

		/** the depth image sent to cuda */
		float * cudaDepthImage;

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

        /** pointer to the object holding the cuda kernels */
        std::unique_ptr<CUDAVolumeRaycaster_kernel> cuda_kernel;

#ifdef DEBUG_LUT
		/** the lookup table */
		std::vector<float4> lut;
#endif // DEBUG_LUT
	};

} /* namespace volume_cuda */
} /* namespace megamol */

#endif // !VOLUME_CUDA_CUDAVOLUMERAYCASTER_H_INCLUDED