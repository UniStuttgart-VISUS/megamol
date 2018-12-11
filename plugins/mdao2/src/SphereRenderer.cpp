#include "stdafx.h"
#define _USE_MATH_DEFINES 1

#include "SphereRenderer.h"


#include <iostream>

// Vislib headers
#include <GL/glu.h>
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Cuboid.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/CameraParameters.h"
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <deque>

#include <fstream>
#include <signal.h>

// Megamol headers
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/CoreInstance.h"

#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "ShaderUtilities.h"


#define checkGLError { GLenum errCode = glGetError(); if (errCode != GL_NO_ERROR) std::cout<<"Error in line "<<__LINE__<<": "<<gluErrorString(errCode)<<std::endl;}


using namespace megamol;

mdao::SphereRenderer::SphereRenderer(void): 
	Renderer3DModule(),
	getDataSlot("getdata", "Connects to the data source"),
	getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
	getTFSlot("gettransferfunction", "Connects to the transfer function module"),
	enableLightingSlot("enable_lighting", "Lighting"),
	enableAOSlot("enable_ao", "Enable AO"),
    enableGeometryShader("use GS proxies", "enables rendering using triangle strips from the geometry shader"),
	aoVolSizeSlot("ao_volsize", "Longest volume edge"),
	aoConeApexSlot("ao_apex", "Cone Apex Angle"),
	aoOffsetSlot("ao_offset", "AO Offset from Surface"),
	aoStrengthSlot("ao_strength", "AO Strength"),
	aoConeLengthSlot("ao_conelen", "AO Cone length"),
	useHPTexturesSlot("high_prec_tex", "Use high precision textures"),
    forceTimeSlot("forceTime", "Flag to force the time code to the specified value. Set to true when rendering a video.")
{
	// Make the slot connectable to multi particle data and molecular data
    this->getDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);
	
	this->getClipPlaneSlot.SetCompatibleCall<megamol::core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
	
	this->getTFSlot.SetCompatibleCall<megamol::core::view::CallGetTransferFunctionDescription>();

	this->MakeSlotAvailable(&getTFSlot);
	
	this->enableLightingSlot << (new core::param::BoolParam(false));
	this->MakeSlotAvailable(&this->enableLightingSlot);	

	this->enableAOSlot << (new core::param::BoolParam(true));
	this->MakeSlotAvailable(&this->enableAOSlot);

    this->enableGeometryShader << (new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->enableGeometryShader);
	
	this->aoVolSizeSlot << (new core::param::IntParam(128, 1, 1024));
	this->MakeSlotAvailable(&this->aoVolSizeSlot);

	this->aoConeApexSlot << (new core::param::FloatParam(50.0f, 1.0f, 90.0f));
	this->MakeSlotAvailable(&this->aoConeApexSlot);
	
	this->aoOffsetSlot << (new core::param::FloatParam(0.01f, 0.0f, 0.2f));
	this->MakeSlotAvailable(&this->aoOffsetSlot);
	
	this->aoStrengthSlot << (new core::param::FloatParam(1.0f, 0.1f, 20.0f));
	this->MakeSlotAvailable(&this->aoStrengthSlot);
	
	this->aoConeLengthSlot << (new core::param::FloatParam(0.8f, 0.01f, 1.0f));
	this->MakeSlotAvailable(&this->aoConeLengthSlot);
	
	this->useHPTexturesSlot << (new core::param::BoolParam(false));
	this->MakeSlotAvailable(&this->useHPTexturesSlot);

    this->forceTimeSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->forceTimeSlot);

	oldHash = -1;	
	vpWidth = -1;
	vpHeight = -1;	
	volGen = nullptr;
}



mdao::SphereRenderer::~SphereRenderer()
{
	this->Release();
}




bool mdao::SphereRenderer::create(void)
{
	// Try to initialize OPENGL extensions
	if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
		return false;
	}
	
	// Generate texture and frame buffer handles
	glGenTextures(3, reinterpret_cast<GLuint*>(&gBuffer));

	glGenFramebuffers(1, &(gBuffer.fbo));
	
	rebuildGBuffer();
	
	// Build the sphere shader	
	rebuildShader();

	bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();
		
	if (enableAO) {
		volGen = new VolumeGenerator();
		volGen->SetShaderSourceFactory(&instance()->ShaderSourceFactory());
		if (!volGen->Init()) {
			std::cerr<<"Error initializing volume generator!"<<std::endl;
			return false;
		}
	}
	
	glGenTextures(1, &this->tfFallbackHandle);
	unsigned char tex[6] = {0, 0, 0,  255, 255, 255};
	glBindTexture(GL_TEXTURE_1D, this->tfFallbackHandle);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glBindTexture(GL_TEXTURE_1D, 0);

	return true;
}


void mdao::SphereRenderer::getClipData(vislib::math::Vector<float, 4> &clipDat, vislib::math::Vector<float, 4> &clipCol)
{
    core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<core::view::CallClipPlane>();

    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}



bool mdao::SphereRenderer::rebuildGBuffer()
{
    float viewport[4];
    glGetFloatv(GL_VIEWPORT, viewport);
	int width = static_cast<int>(viewport[2]);
	int height = static_cast<int>(viewport[3]);

	if (vpWidth == width && vpHeight == height && !this->useHPTexturesSlot.IsDirty())
		return true;
	
	vpWidth = width;
	vpHeight = height;
	this->useHPTexturesSlot.ResetDirty();

	bool highPrecision = this->useHPTexturesSlot.Param<megamol::core::param::BoolParam>()->Value();
	
	glBindTexture(GL_TEXTURE_2D, gBuffer.color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glBindTexture(GL_TEXTURE_2D, gBuffer.normals);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, highPrecision ? GL_RGBA32F: GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glBindTexture(GL_TEXTURE_2D, gBuffer.depth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr);

	// Configure the framebuffer object
	GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, &prevFBO);
	
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer.fbo);checkGLError;
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gBuffer.color, 0);checkGLError;
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gBuffer.normals, 0);checkGLError;
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gBuffer.depth, 0);checkGLError;

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout<<"Framebuffer NOT complete!"<<std::endl;
	
	glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);

	return true;
}




bool mdao::SphereRenderer::rebuildShader()
{
	vislib::graphics::gl::ShaderSource vert, frag;
	core::utility::ShaderSourceFactory &factory = instance()->ShaderSourceFactory();

	// Create the sphere shader if neccessary
	if (!vislib::graphics::gl::GLSLShader::IsValidHandle(sphereShader) &&
		!megamol::mdao::InitializeShader(&factory, sphereShader, "mdao2::vertex", "mdao2::fragment"))
		return false;
	
    if (!vislib::graphics::gl::GLSLGeometryShader::IsValidHandle(sphereGeoShader) &&
        !megamol::mdao::InitializeShader(&factory, sphereGeoShader, "mdao2::geovert", "mdao2::fragment", "mdao2::geogeo"))
        return false;

	
	// Load the vertex shader
	if (!factory.MakeShaderSource("mdao2::deferred::vertex", vert))
		return false;

	bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();
	bool enableLighting = this->enableLightingSlot.Param<megamol::core::param::BoolParam>()->Value();
	
	frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::main"));
	if (enableLighting)
		frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::lighting"));
	else
		frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::lighting_stub"));
	
	if (enableAO) {
		float apex = this->aoConeApexSlot.Param<megamol::core::param::FloatParam>()->Value();
		
		std::vector<vislib::math::Vector<float, 4> > directions;
		generate3ConeDirections(directions, apex *static_cast<float>(M_PI)/180.0f);
		std::string directionsCode = generateDirectionShaderArrayString(directions, "coneDirs");

		vislib::graphics::gl::ShaderSource::StringSnippet* dirSnippet = new vislib::graphics::gl::ShaderSource::StringSnippet(directionsCode.c_str());
		frag.Append(dirSnippet);

		frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::ambocc"));
	} else {
		frag.Append(factory.MakeShaderSnippet("mdao2::deferred::fragment::ambocc_stub"));
	}
	
	try {
		lightingShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count());
	} catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
			"Unable to compile mdao shader: %s", ce.GetMsg());
		return false;
	}
	
	return true;
}




void mdao::SphereRenderer::release(void)
{
	instance()->ShaderSourceFactory();
	sphereShader.Release();
    sphereGeoShader.Release();
	lightingShader.Release();
	
	glDeleteTextures(3, reinterpret_cast<GLuint*>(&gBuffer));
	
	glDeleteFramebuffers(1, &(gBuffer.fbo));
	
	if (volGen != nullptr) {
		delete volGen;
		volGen = nullptr;
	}
	
}


bool mdao::SphereRenderer::GetExtents(megamol::core::Call& call)
{
    megamol::core::view::AbstractCallRender3D *renderCall = dynamic_cast<megamol::core::view::AbstractCallRender3D*>(&call);
    if (renderCall == NULL) return false;
	
	// Create a caller object
    megamol::core::moldyn::MultiParticleDataCall *extentsCall = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();	
	if (extentsCall == NULL) return false;
	
	extentsCall->SetFrameID(static_cast<unsigned int>(renderCall->Time()), this->isTimeForced());
	
	// Try to call the data storage with calling function 1 (which delivers the extents)
	if (!(*extentsCall)(1)) return false;
	
	// Set values
	renderCall->SetTimeFramesCount(extentsCall->FrameCount());
	renderCall->AccessBoundingBoxes() = extentsCall->AccessBoundingBoxes();
	
	// Scale the box such that the longest edge is 10 units
	float scaling = renderCall->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	if (scaling > 0.0000001f) {
		scaling = 10.0f / scaling;
	} else {
		scaling = 1.0f;
	}
	
	renderCall->AccessBoundingBoxes().MakeScaledWorld(scaling);
	
	return true;
}




bool mdao::SphereRenderer::Render(megamol::core::Call& call)
{
    megamol::core::view::AbstractCallRender3D *renderCall = dynamic_cast<megamol::core::view::AbstractCallRender3D*>(&call);
    if (renderCall == NULL) return false;	

	// Try to get the extents and copy them into the render call
	if (!GetExtents(call)) return false;
	
	// Create a caller object for data retrieval
    megamol::core::moldyn::MultiParticleDataCall *dataCall = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
	if (dataCall == NULL) return false;
	
	dataCall->SetFrameID(static_cast<unsigned int>(renderCall->Time()), this->isTimeForced());

	// Try to get the data
	if (!(*dataCall)(0)) return false;
	dataCall->AccessBoundingBoxes() = renderCall->AccessBoundingBoxes();
	
	// We need to regenerate the shader if certain settings are changed
	if (this->enableAOSlot.IsDirty() || 
		this->enableLightingSlot.IsDirty() || 
		this->aoConeApexSlot.IsDirty()) 
	{
		this->aoConeApexSlot.ResetDirty();
		this->enableLightingSlot.ResetDirty();
		rebuildShader();
	}

	// Rebuild the GBuffer if neccessary
	this->rebuildGBuffer();

	// Rebuild and reupload working data if neccessary
	this->rebuildWorkingData(renderCall, dataCall);

	GLint prevFBO;
    glGetIntegerv( GL_FRAMEBUFFER_BINDING, &prevFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer.fbo);checkGLError;
	GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, bufs);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); checkGLError;

	glBindFragDataLocation(sphereShader.ProgramHandle(), 0, "outColor"); checkGLError;
	glBindFragDataLocation(sphereShader.ProgramHandle(), 1, "outNormal"); checkGLError;
	
	
	// Render the particles' geometry
	this->renderParticlesGeometry(renderCall, dataCall);

	glBindFramebuffer(GL_FRAMEBUFFER, prevFBO);
	
	renderDeferredPass(renderCall);
	
	dataCall->Unlock();

    return true;	
}


void mdao::SphereRenderer::renderDeferredPass(megamol::core::view::AbstractCallRender3D* renderCall)
{
	vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> mv, mvInverse, mvp, mvpInverse, proj, mvpTrans;
	vislib::graphics::gl::CameraOpenGL cam(renderCall->GetCameraParameters()); 
	glGetFloatv(GL_MODELVIEW_MATRIX, mv.PeekComponents());
	glGetFloatv(GL_PROJECTION_MATRIX, proj.PeekComponents());

	bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();
	bool enableLighting = this->enableLightingSlot.Param<megamol::core::param::BoolParam>()->Value();
	bool highPrecision = this->useHPTexturesSlot.Param<megamol::core::param::BoolParam>()->Value();

	
	mvp = proj * mv;
	mvInverse = mv;
	mvInverse.Invert();
	mvpInverse = mvp;
	mvpInverse.Invert();
	
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.depth);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.normals);
	glActiveTexture(GL_TEXTURE0); 
	glBindTexture(GL_TEXTURE_2D, gBuffer.color);

	glPointSize(static_cast<GLfloat>((std::max)(vpWidth, vpHeight)));
	
	lightingShader.Enable();
	lightingShader.SetParameter("inWidth", static_cast<float>(vpWidth));
	lightingShader.SetParameter("inHeight", static_cast<float>(vpHeight));
	glUniformMatrix4fv(lightingShader.ParameterLocation("inMvpInverse"), 1, GL_FALSE, mvpInverse.PeekComponents());
	lightingShader.SetParameter("inColorTex", static_cast<int>(0));
	lightingShader.SetParameter("inNormalsTex", static_cast<int>(1));
	lightingShader.SetParameter("inDepthTex", static_cast<int>(2));
	
	lightingShader.SetParameter("inUseHighPrecision", highPrecision);

	if (enableLighting) {
		vislib::math::Vector<float, 4> lightDir;
		glEnable(GL_LIGHTING);
		glGetLightfv(GL_LIGHT0, GL_POSITION, lightDir.PeekComponents());
		glDisable(GL_LIGHTING);
		lightDir = mvInverse * lightDir;
		lightDir.Normalise();
		lightingShader.SetParameterArray3("inObjLightDir", 1, lightDir.PeekComponents());
		lightingShader.SetParameterArray3("inObjCamPos", 1, mvInverse.GetColumn(3).PeekComponents());
	}
	
	if (enableAO) {
		float aoOffset = this->aoOffsetSlot.Param<megamol::core::param::FloatParam>()->Value();
		float aoStrength = this->aoStrengthSlot.Param<megamol::core::param::FloatParam>()->Value();
		float aoConeLength = this->aoConeLengthSlot.Param<megamol::core::param::FloatParam>()->Value();
		if (volGen != nullptr) {
			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_3D, volGen->GetVolumeTextureHandle());
			glActiveTexture(GL_TEXTURE0);
		}
		lightingShader.SetParameter("inAOOffset", aoOffset);
		lightingShader.SetParameter("inDensityTex", static_cast<int>(3));
		lightingShader.SetParameter("inAOStrength", aoStrength);
		lightingShader.SetParameter("inAOConeLength", aoConeLength);
		lightingShader.SetParameter("inAmbVolShortestEdge", ambConeConstants[0]);
		lightingShader.SetParameter("inAmbVolMaxLod", ambConeConstants[1]);
		lightingShader.SetParameterArray3("inBoundsMin", 1, renderCall->AccessBoundingBoxes().ObjectSpaceClipBox().GetLeftBottomBack().PeekCoordinates());
		lightingShader.SetParameterArray3("inBoundsSize", 1, renderCall->AccessBoundingBoxes().ObjectSpaceClipBox().GetSize().PeekDimension());		
	}
	glBegin(GL_POINTS);
	glVertex2f(0.0f, 0.0f);
	glEnd();
	lightingShader.Disable();
}




void mdao::SphereRenderer::renderParticlesGeometry(megamol::core::view::AbstractCallRender3D* renderCall, megamol::core::moldyn::MultiParticleDataCall* dataCall)
{
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	
	float scaling = 10.0f/renderCall->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	bool highPrecision = this->useHPTexturesSlot.Param<megamol::core::param::BoolParam>()->Value();

	glScalef(scaling, scaling, scaling);
	
    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
	glPointSize(64);
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

	vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> mv, mvInverse, mvp, mvpInverse, proj, mvpTrans;
	vislib::graphics::gl::CameraOpenGL cam(renderCall->GetCameraParameters()); 
	glGetFloatv(GL_MODELVIEW_MATRIX, mv.PeekComponents());
	glGetFloatv(GL_PROJECTION_MATRIX, proj.PeekComponents());

	mvp = proj * mv;
	mvInverse = mv;
	mvInverse.Invert();
	mvpInverse = mvp;
	mvpInverse.Invert();

	vislib::math::Vector<float, 3> mvFront = -mvInverse.GetColumn(2);
	vislib::math::Vector<float, 3> mvRight = mvInverse.GetColumn(0);
	vislib::math::Vector<float, 3> mvUp = mvInverse.GetColumn(1);
	mvFront.Normalise();
	mvRight.Normalise();
	mvUp.Normalise();

    bool useGeo = this->enableGeometryShader.Param<core::param::BoolParam>()->Value();

    vislib::graphics::gl::GLSLShader& theShader = useGeo ? sphereGeoShader : sphereShader;

	theShader.Enable();
	glUniformMatrix4fv(theShader.ParameterLocation("inMvp"), 1, GL_FALSE, mvp.PeekComponents());
	glUniformMatrix4fv(theShader.ParameterLocation("inMvpInverse"), 1, GL_FALSE, mvpInverse.PeekComponents());
	glUniformMatrix4fv(theShader.ParameterLocation("inMvpTrans"), 1, GL_TRUE, mvp.PeekComponents());
	glUniformMatrix4fv(theShader.ParameterLocation("inMv"), 1, GL_FALSE, mv.PeekComponents());

	
    theShader.SetParameterArray4("inViewAttr", 1, viewportStuff);
    theShader.SetParameterArray3("inCamFront", 1, mvFront.PeekComponents());
    theShader.SetParameterArray3("inCamRight", 1, mvRight.PeekComponents());
    theShader.SetParameterArray3("inCamUp", 1, mvUp.PeekComponents());
    theShader.SetParameterArray4("inCamPos", 1, mvInverse.GetColumn(3).PeekComponents());
	
    theShader.SetParameterArray4("inClipDat", 1, clipDat.PeekComponents());
    theShader.SetParameterArray4("inClipCol", 1, clipCol.PeekComponents());
	
    theShader.SetParameter("inUseHighPrecision", highPrecision);
	
	
	for (unsigned int i=0; i<gpuData.size(); ++i) {
		glBindVertexArray(gpuData[i].vertexArray);
		
		core::moldyn::SimpleSphericalParticles &parts = dataCall->AccessParticles(i);
		
		float globalRadius = 0.0f;
		if (parts.GetVertexDataType() != megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
			globalRadius = parts.GetGlobalRadius();
		
        theShader.SetParameter("inGlobalRadius", globalRadius);

		bool useGlobalColor = false;
		if (parts.GetColourDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
			useGlobalColor = true;
			const unsigned char *globalColor = parts.GetGlobalColour();
			float globalColorFlt[4] = {
				static_cast<float>(globalColor[0])/255.0f,
				static_cast<float>(globalColor[1])/255.0f,
				static_cast<float>(globalColor[2])/255.0f,
				1.0f
			};
            theShader.SetParameterArray4("inGlobalColor", 1, globalColorFlt);
		} 
        theShader.SetParameter("inUseGlobalColor", useGlobalColor);
		
		bool useTransferFunction = false;
		if (parts.GetColourDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
			useTransferFunction = true;
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_1D, getTransferFunctionHandle());
            theShader.SetParameter("inTransferFunction", static_cast<int>(0));
			float tfRange[2] = {parts.GetMinColourIndexValue(), parts.GetMaxColourIndexValue()};
            theShader.SetParameterArray2("inIndexRange", 1, tfRange);
		}
        theShader.SetParameter("inUseTransferFunction", useTransferFunction);
		
		glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(dataCall->AccessParticles(i).GetCount()));
	}

	glBindVertexArray(0);
    theShader.Disable();
}


GLuint mdao::SphereRenderer::getTransferFunctionHandle()
{
	core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
	if ((cgtf != nullptr) && (*cgtf)()) 
		return cgtf->OpenGLTexture();
	
	return tfFallbackHandle;
}




void mdao::SphereRenderer::rebuildWorkingData(megamol::core::view::AbstractCallRender3D* renderCall, megamol::core::moldyn::MultiParticleDataCall* dataCall)
{
	SIZE_T hash = dataCall->DataHash();
	unsigned int frameID = dataCall->FrameID();

    this->getClipData(clipDat, clipCol);

	
	// Check if we got a new data set
	bool stateInvalid =  (hash != oldHash || frameID != oldFrameID);

	oldHash = hash;
	oldFrameID = frameID;
	
	// Upload new data if neccessary
	if (stateInvalid) {
		unsigned int partsCount = dataCall->GetParticleListCount();
		
		// Add buffers if neccessary
		for (unsigned int i=static_cast<unsigned int>(gpuData.size()); i<partsCount; ++i) {
			gpuParticleDataType data;

			glGenVertexArrays(1, &(data.vertexArray));
			glGenBuffers(1, &(data.vertexVBO));
			glGenBuffers(1, &(data.colorVBO));

			gpuData.push_back(data);
		}
		
		// Remove buffers if neccessary
		while(gpuData.size() > partsCount) {
			gpuParticleDataType &data = gpuData.back();
			glDeleteVertexArrays(1, &(data.vertexArray));
			glDeleteBuffers(1, &(data.vertexVBO));
			glDeleteBuffers(1, &(data.colorVBO));
			gpuData.pop_back();
		}
		
		// Reupload buffers
		for (unsigned int i=0; i<partsCount; ++i) {
			uploadDataToGPU(gpuData[i], dataCall->AccessParticles(i));
		}
	}

	// Check if voxelization is even needed
	if (this->enableAOSlot.IsDirty()) {
		bool enableAO = this->enableAOSlot.Param<megamol::core::param::BoolParam>()->Value();
		
		if (!enableAO && volGen != nullptr) {
			delete volGen;
			volGen = nullptr;
		}
		if (enableAO && volGen == nullptr) {
			volGen = new VolumeGenerator();
			volGen->SetShaderSourceFactory(&instance()->ShaderSourceFactory());
			volGen->Init();
		}
		
	}
	
	// Recreate the volume if neccessary
	if (volGen != nullptr && (stateInvalid || this->enableAOSlot.IsDirty() || this->aoVolSizeSlot.IsDirty() || clipDat != oldClipDat))  {
		int volSize = this->aoVolSizeSlot.Param<megamol::core::param::IntParam>()->Value();
	
		const vislib::math::Cuboid<float> &cube = renderCall->AccessBoundingBoxes().ObjectSpaceClipBox();
		vislib::math::Dimension<float, 3> dims = cube.GetSize();
	
		// Calculate the extensions of the volume by using the specified number of voxels for the longest edge
		float longestEdge = cube.LongestEdge();
		dims.Scale(static_cast<float>(volSize) / longestEdge);
		
		// The X size must be a multiple of 4, so we might have to correct that a little
		dims.SetWidth(ceil(dims.GetWidth()/4.0f)*4.0f);
		
		dims.SetHeight(ceil(dims.GetHeight()));
		dims.SetDepth(ceil(dims.GetDepth()));
		ambConeConstants[0] = (std::min)(dims.Width(), (std::min)(dims.Height(), dims.Depth()));
		ambConeConstants[1] = ceil(std::log2(static_cast<float>(volSize)))-1.0f;
		
		// Set resolution accordingly
		volGen->SetResolution(ceil(dims.GetWidth()), ceil(dims.GetHeight()), ceil(dims.GetDepth()));
		
		// Insert all particle lists
		volGen->ClearVolume();
		
		volGen->StartInsertion(cube, clipDat);
		for (unsigned int i=0; i<gpuData.size(); ++i) {
			float globalRadius = 0.0f;
			if (dataCall->AccessParticles(i).GetVertexDataType() != megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
				globalRadius = dataCall->AccessParticles(i).GetGlobalRadius();
			volGen->InsertParticles(static_cast<unsigned int>(dataCall->AccessParticles(i).GetCount()), globalRadius, gpuData[i].vertexArray);
		}
		volGen->EndInsertion();
		
		volGen->RecreateMipmap();

	}
	
    // reset shotter for legacy opengl crap
    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

	this->enableAOSlot.ResetDirty();
	this->aoVolSizeSlot.ResetDirty();
	this->oldClipDat = clipDat;
}




void mdao::SphereRenderer::uploadDataToGPU(const mdao::SphereRenderer::gpuParticleDataType &gpuData, megamol::core::moldyn::MultiParticleDataCall::Particles& particles)
{
	glBindVertexArray(gpuData.vertexArray);

	glBindBuffer(GL_ARRAY_BUFFER, gpuData.colorVBO);
	unsigned int partCount = static_cast<unsigned int>(particles.GetCount());
	// colour
	switch (particles.GetColourDataType()) {
		case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE:
			break;
		case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
			glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetColourDataStride(), 3u), particles.GetColourData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, particles.GetColourDataStride(), 0);
			break;
		case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
			glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetColourDataStride(), 4u), particles.GetColourData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, particles.GetColourDataStride(), 0);
			break;
		case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
			glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetColourDataStride(), static_cast<unsigned int>(3*sizeof(float))), particles.GetColourData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, particles.GetColourDataStride(), 0);
			break;
		case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
			glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetColourDataStride(), static_cast<unsigned int>(4*sizeof(float))), particles.GetColourData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, particles.GetColourDataStride(), 0);
			break;
		// Not supported - fall through to the gay version
		// FIXME: this will probably not work!
		case megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: 
			glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetColourDataStride(), static_cast<unsigned int>(1*sizeof(float))), particles.GetColourData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, particles.GetColourDataStride(), 0);
			//std::cout<<"Transfer function"<<std::endl;
			break;
		default:
			glColor4ub(127, 127, 127,255);
			glDisableVertexAttribArray(1);
			break;
	}

	// radius and position
	glBindBuffer(GL_ARRAY_BUFFER, gpuData.vertexVBO);
	switch (particles.GetVertexDataType()) {
		case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
			return;
		case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetVertexDataStride(), static_cast<unsigned int>(3 * sizeof(float))), particles.GetVertexData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, particles.GetVertexDataStride(), 0);
			break;

		case megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            glBufferData(GL_ARRAY_BUFFER, partCount*(std::max)(particles.GetVertexDataStride(), static_cast<unsigned int>(4*sizeof(float))), particles.GetVertexData(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, particles.GetVertexDataStride(), 0);
			break;
		default:
			glDisableVertexAttribArray(0);
			return;
	}

}




void mdao::SphereRenderer::generate3ConeDirections(std::vector< vislib::math::Vector<float, 4> >& directions, float apex)
{
	directions.clear();

	float edge_length = 2.0f * tan(0.5f*apex);
	float height = sqrt(1.0f - edge_length*edge_length/12.0f);
	float radius = sqrt(3.0f)/3.0f * edge_length;		
	
	for (int i=0; i<3; ++i) {
		float angle = static_cast<float>(i)/3.0f*2.0f*static_cast<float>(M_PI);

		vislib::math::Vector<float, 3> center(cos(angle)*radius, height, sin(angle)*radius);
		center.Normalise();
		directions.push_back(vislib::math::Vector<float, 4>(center.X(), center.Y(), center.Z(), edge_length));
	}

}




std::string mdao::SphereRenderer::generateDirectionShaderArrayString(const std::vector< vislib::math::Vector<float, 4> >& directions, const std::string& directionsName)
{
	std::stringstream result;
	
	std::string upperDirName = directionsName;
	std::transform(upperDirName.begin(), upperDirName.end(), upperDirName.begin(), ::toupper);
	
	result<<"#define NUM_"<<upperDirName<<" "<<directions.size()<<std::endl;
	result<<"const vec4 "<<directionsName<<"[NUM_"<<upperDirName<<"] = vec4[NUM_"<<upperDirName<<"]("<<std::endl;
	
	for (auto iter = directions.begin(); iter != directions.end(); ++iter) {
		result<<"\tvec4("<<(*iter)[0]<<", "<<(*iter)[1]<<", "<<(*iter)[2]<<", "<<(*iter)[3]<<")";
		if (iter+1 != directions.end())
			result<<",";
		result<<std::endl;
	}
	result<<");"<<std::endl;

	return result.str();
}


bool mdao::SphereRenderer::isTimeForced() const {
    return this->forceTimeSlot.Param<core::param::BoolParam>()->Value();
}
