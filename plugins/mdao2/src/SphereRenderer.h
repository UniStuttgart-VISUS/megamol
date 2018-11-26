#ifndef AOWT_SPHERERENDERER_H_INCLUDED
#define AOWT_SPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#include "stdafx.h"

#include "vislib/types.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/view/AbstractCallRender3D.h"

#include "VolumeGenerator.h"

#include <vector>


namespace megamol {
namespace mdao {	
	
	class SphereRenderer: public megamol::core::view::Renderer3DModule {
	public:
		
		static const char *ClassName(void) {
			return "MDAO2Renderer";
		}
		
		static const char *Description(void) {
			return "Renderer for sphere glyphs with ambient occlusion";
		}
		
		static bool IsAvailable(void) {
			return (ogl_IsVersionGEQ(3, 3) != 0);
		}
		
		SphereRenderer(void);
		virtual ~SphereRenderer();
		
	protected:
		virtual bool create(void);
		virtual void release(void);

	private:
		struct gpuParticleDataType {
			GLuint vertexVBO, colorVBO, vertexArray;
		};
		
		struct gBufferDataType {
			GLuint color, depth, normals;
			GLuint fbo;
		};
		
		// The sphere shader
        vislib::graphics::gl::GLSLShader sphereShader, lightingShader;
        vislib::graphics::gl::GLSLGeometryShader sphereGeoShader;
		
		// GPU buffers for particle lists
		std::vector<gpuParticleDataType> gpuData;

		// G-Buffer handles for deferred shading
		gBufferDataType gBuffer;
		
		SIZE_T oldHash;
		unsigned int oldFrameID;
		int vpWidth, vpHeight;

		vislib::math::Vector<float, 2> ambConeConstants;
		vislib::math::Vector<float, 4> clipDat, oldClipDat, clipCol;
		// Fallback handle if no transfer function is specified
		GLuint tfFallbackHandle;

		VolumeGenerator *volGen;
		
		// Call for data
		megamol::core::CallerSlot getDataSlot;
        // Call for clipping plane
        megamol::core::CallerSlot getClipPlaneSlot;
		// Call for transfer function
		megamol::core::CallerSlot getTFSlot;
		
		// Enable or disable lighting
		megamol::core::param::ParamSlot enableLightingSlot;
		// Enable Ambient Occlusion
		megamol::core::param::ParamSlot enableAOSlot;
        megamol::core::param::ParamSlot enableGeometryShader;
		// AO texture size 
		megamol::core::param::ParamSlot aoVolSizeSlot;
		// Cone Apex Angle 
		megamol::core::param::ParamSlot aoConeApexSlot;
		// AO offset from surface
		megamol::core::param::ParamSlot aoOffsetSlot;
		// AO strength
		megamol::core::param::ParamSlot aoStrengthSlot;
		// AO cone length
		megamol::core::param::ParamSlot aoConeLengthSlot;
		// High precision textures slot
		megamol::core::param::ParamSlot useHPTexturesSlot;

        // bool parameter to force the time from the data set
        megamol::core::param::ParamSlot forceTimeSlot;

		virtual bool GetExtents(megamol::core::Call& call);
		virtual bool Render(megamol::core::Call& call);
		
		void uploadDataToGPU(const gpuParticleDataType &gpuData, megamol::core::moldyn::MultiParticleDataCall::Particles& particles);
		void renderParticlesGeometry(megamol::core::view::AbstractCallRender3D* renderCall, megamol::core::moldyn::MultiParticleDataCall* dataCall);
		
		bool rebuildShader();
		bool rebuildGBuffer();

		void rebuildWorkingData(megamol::core::view::AbstractCallRender3D* renderCall, megamol::core::moldyn::MultiParticleDataCall* dataCall);

		std::string generateDirectionShaderArrayString(const std::vector< vislib::math::Vector< float, int(4) > >& directions, const std::string& directionsName);
		void generate3ConeDirections(std::vector< vislib::math::Vector< float, int(4) > >& directions, float apex);
		void renderDeferredPass(megamol::core::view::AbstractCallRender3D* renderCall);
		
		void getClipData(vislib::math::Vector<float, 4> &clipDat, vislib::math::Vector<float, 4> &clipCol);
		
		GLuint getTransferFunctionHandle();

        // simple access to the value of forceTimeSlot
        bool isTimeForced() const;

	};
	
}
}

#endif /* AOWT_SPHERERENDERER_H_INCLUDED */
