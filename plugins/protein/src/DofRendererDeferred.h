/*
 * DofRendererDeferred.h
 *
 * Copyright (C) 2012 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_DOFRENDERERDEFERRED_H_INCLUDED
#define MMPROTEINPLUGIN_DOFRENDERERDEFERRED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModuleDS.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace protein {

/**
 * Class providing screen space depth of field effect.
 */
class DofRendererDeferred :
	public megamol::core::view::Renderer3DModuleDS {

public:

	/* Answer the name of this module.
	 *
	 * @return The name of this module.
	 */
	static const char *ClassName(void) {
		return "DofRendererDeferred";
	}

	/**
	 * Answer a human readable description of this module.
	 *
	 * @return A human readable description of this module.
	 */
	static const char *Description(void) {
		return "Offers application of screen space depth of field effect.";
	}

	/**
	 * Answers whether this module is available on the current system.
	 *
	 * @return 'true' if the module is available, 'false' otherwise.
	 */
	static bool IsAvailable(void) {
		if(!vislib::graphics::gl::GLSLShader::AreExtensionsAvailable())
			return false;
		if(!vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable())
			return false;
        if(!isExtAvailable("GL_ARB_texture_rectangle"))
			return false;
		return true;
	}

	/** Ctor. */
	DofRendererDeferred(void);

	/** Dtor. */
	virtual ~DofRendererDeferred(void);

protected:

	/** The depth of field modes */
	enum DepthOfFieldMode {NONE, DOF_SHADERX, DOF_MIPMAP, DOF_LEE};

	/** The renderers render modes */
	enum RenderMode {DOF, FOCAL_DIST};

	/**
	 * Implementation of 'create'.
	 *
	 * @return 'true' on success, 'false' otherwise.
	 */
	bool create(void);

	/**
	 * Implementation of 'release'.
	 */
	void release(void);

	/**
	 * The get extents callback. The module should set the members of
	 * 'call' to tell the caller the extents of its data (bounding boxes
	 * and times).
	 *
	 * @param call The calling call.
	 *
	 * @return The return value of the function.
	 */
	virtual bool GetExtents(megamol::core::Call& call);

	/**
	 * The render callback.
	 *
	 * @param call The calling call.
	 *
	 * @return The return value of the function.
	 */
	virtual bool Render(megamol::core::Call& call);

private:

	/**
	 * TODO
	 *
	 * @param d_focus d_focus
	 * @param a a
	 * @param f f
	 * @return something
	 */
	float calcCocSlope(float d_focus, float a, float f);

	/**
	 * Initialize the frame buffer object.
	 *
	 * @param width The width of the buffer.
	 * @param height The height of the buffer.
	 *
	 * @return 'True' on success, 'false' otherwise
	 */
	bool createFbo(UINT width, UINT height);

	/**
	 * Create a an averaged version of the source texture with a smaller
	 * resolution using mip maps.
	 */
	void createReducedTexMipmap();

	/**
	 * Create a an averaged version of the source texture with a smaller
	 * resolution.
	 */
	void createReducedTexShaderX();

	/**
	 * TODO
	 */
	void drawBlurMipmap();

	/**
	 * Drawe a blurred picture according to shader x algorithm.
	 */
	void drawBlurShaderX();

	/**
	 * TODO
	 */
	void filterLee();

	/**
	 * TODO
	 */
	void filterMipmap();

	/**
	 * Apply gaussian filter to the current framebuffer content
	 */
	void filterShaderX();

	/**
	 * Recalculate parameters for the shader x algorithm
	 */
	void recalcShaderXParams();

	/**
	 * Update parameters if necessary.
	 *
	 * @return 'True' on success, 'false' otherwise
	 */
	bool updateParams();

	/// Caller slot
	megamol::core::CallerSlot rendererSlot;

	/// Param for the renderers rendermode
	megamol::core::param::ParamSlot rModeParam;
	RenderMode rMode;

	/// Param to determine the depth of field algorithm
	megamol::core::param::ParamSlot dofModeParam;
	DepthOfFieldMode dofMode;

	/// Param to toggle gaussian filtering
	megamol::core::param::ParamSlot toggleGaussianParam;
	bool useGaussian;

	/// Param for focal distance
	megamol::core::param::ParamSlot focalDistParam;
	float focalDist;

	/// Param for aperture
	megamol::core::param::ParamSlot apertureParam;
	float aperture;

	/// Param to scale coc radius
	megamol::core::param::ParamSlot cocRadiusScaleParam;
	float cocRadiusScale;

	/// Param to change focal length
	megamol::core::param::ParamSlot focalLengthParam;
	float focalLength;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

	/// The renderers frame buffer object
	GLuint fbo;

	/// The depth buffer
	GLuint depthBuffer;

	/// The source buffer
	GLuint sourceBuffer;

	/// Low res textures (used for mipmap implementation)
	GLuint fboMipMapTexId[2]; // two necessary for filtering

	/// Low res textures (used for shaderX implementation)
	GLuint fboLowResTexId[2]; // two necessary for filtering

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

	/// The fbos width
	int width;

	/// The fbos height
	int height;

	/// The inverse of the fbos width (= pixelsize in texcoords)
	float widthInv;

	/// The inverse of the fbos height (= pixelsize in texcoords)
	float heightInv;

	/// dNear
	float dNear;

	/// dFar
	float dFar;

	/// clampFar
	float clampFar;

	/// filmWidth
	float filmWidth;

	/// maxCoc
	float maxCoC;

	/// Something ...
	bool originalCoC;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

	/// Depth of field shader (reduce, mipmap)
	vislib::graphics::gl::GLSLShader reduceMipMap;

	/// Depth of field shader (blur, mipmap)
	vislib::graphics::gl::GLSLShader blurMipMap;

	/// Depth of field shader (reduce, shaderX)
	vislib::graphics::gl::GLSLShader reduceShaderX;

	/// Depth of field shader (blur, shaderX)
	vislib::graphics::gl::GLSLShader blurShaderX;

	/// Shader for horizontal gaussian filter
	vislib::graphics::gl::GLSLShader gaussianHoriz;

	/// Shader for vertical gaussian filter
	vislib::graphics::gl::GLSLShader gaussianVert;

	/// Shader for gaussian filter (lee)
	vislib::graphics::gl::GLSLShader gaussianLee;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

	/// Camera information
	vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
};


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_DOFRENDERERDEFERRED_H_INCLUDED

