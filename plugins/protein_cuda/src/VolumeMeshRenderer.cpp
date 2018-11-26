/*
 * VolumeMeshRenderer.cpp
 *
 * Copyright(C) 2012 by Universitaet Stuttgart(VISUS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#define _USE_MATH_DEFINES 1

//#define TEST
#include "VolumeMeshRenderer.h"
#include "protein_calls/IntSelectionCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "MappableFloatPair.h"
#include "MappableCategoryFloat.h"
#include "mmcore/CoreInstance.h"
#include "MolecularSurfaceFeature.h"
#include "SplitMergeFeature.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/StringConverter.h"
#include "vislib/sys/PerformanceCounter.h"
#include "vislib/math/Matrix.h"
#include <GL/glu.h>
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <cuda_gl_interop.h>
#include <thrust/version.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <omp.h>
#include <ctime>

#include "mmcore/utility/ColourParser.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;

#define CUDA_VERIFY(call) CudaVerify(call, __LINE__)
#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

/*
 * VolumeMeshRenderer::VolumeMeshRenderer(CTOR)
 */
VolumeMeshRenderer::VolumeMeshRenderer(void) : Renderer3DModuleDS(), 
        molDataCallerSlot ("getData", "Connects the molecule rendering with molecule data storage"),
        bsDataCallerSlot ("getBindingSites", "Connects the molecule rendering with binding site data storage"),
        selectionCallerSlot( "getSelection", "Connects the rendering with selection storage." ),
        hiddenCallerSlot( "getHidden", "Connects the rendering with visibility storage." ),
        diagramCalleeSlot ("diagramout", "Provides data for time-based line graph"),
        splitMergeCalleeSlot("splitmergeout", "Provides data for splitmerge graph"),
        centerLineDiagramCalleeSlot("centerlineout", "Provides data for center line graph"),
        resSelectionCallerSlot( "getResSelection", "Connects the sequence diagram rendering with residue selection storage." ),
        polygonModeParam("polygonMode", "Polygon rasterization mode"),
        blendItParam("blendIt", "Enable blending"),
        alphaParam("alphaBlend", "Alpha for blending"),
        showCenterlinesParam("showCenterlines", "Render centerlines"),
        showNormalsParam("showNormals", "Render normals"),
        showCentroidsParam("showCentroids", "Render centroids"),
        minDistCenterLineParam( "minDistCenterLine", "Minimum distance of center line node to molecular surface."),
        aoThresholdParam("aoThreshold", "AO-Threshold for Segmentation"),
        isoValueParam("isoValue", "Isovalue for marching tetrahedrons"),
        maxDistanceParam("maxDistance", "Distance threshold"),
        maxDeltaDistanceParam("maxDistanceDelta", "Distance' threshold"),
        colorTableFileParam("color::colorTableFile", "Color Table Filename"),
        coloringModeParam0( "color::coloringMode0", "The first coloring mode."),
        coloringModeParam1( "color::coloringMode1", "The second coloring mode."),
        cmWeightParam( "color::colorWeighting", "The weighting of the two coloring modes."),
        minGradColorParam("color::minGradColor", "The color for the minimum value for gradient coloring" ),
        midGradColorParam("color::midGradColor", "The color for the middle value for gradient coloring" ),
        maxGradColorParam("color::maxGradColor", "The color for the maximum value for gradient coloring" ),
        interpolParam( "posInterpolation", "Enable positional interpolation between frames" ),
        qualityParam( "quicksurf::quality", "Quality" ),
        radscaleParam( "quicksurf::radscale", "Radius scale" ),
        gridspacingParam( "quicksurf::gridspacing", "Grid spacing" ),
        areaThresholdParam( "areaThreshold", "Area threshold for segment removal"),
		haloEnableParam( "halo::enable", "Enable the halo."),
		haloColorParam( "halo::color", "Color of halo."),
		haloAlphaParam( "halo::alpha", "Alpha of halo."),
        lastTime(-1), polygonMode(FILL), blendIt(false), showNormals(false),
        showCentroids(false), aoThreshold(0.15f), isoValue(0.5f), maxDistance(2.0f), maxDeltaDistance(1.0f),
        centroidColorsIndex(-1), vertexCount(0), cubeCount( 0), cubeCountAllocated(0), cubeCountOld( 0), activeCubeCountOld( 0), 
        activeCubeCountAllocted(0), vertexCountAllocted(0), centroidCountAllocated(0), centroidCountLast( 0), centroidsLast(0), 
        centroidColorsLast(0), centroidLabelsLast(0), centroidAreasLast( 0), featureCounter( 0), featureListIdx( 0),
        modified( 0), segmentsRemoved( 0), featureList(), splitMergeList(), transitionList(), featureSelection(), featureVisibility(),
        resSelectionCall(0), atomSelection(0), atomSelectionCnt(0), setCUDAGLDevice(true)
{
    // set caller slot for different data calls
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable (&this->molDataCallerSlot);
    this->bsDataCallerSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable (&this->bsDataCallerSlot);
    this->selectionCallerSlot.SetCompatibleCall<IntSelectionCallDescription>();
    this->MakeSlotAvailable(&this->selectionCallerSlot);
    this->hiddenCallerSlot.SetCompatibleCall<IntSelectionCallDescription>();
    this->MakeSlotAvailable(&this->hiddenCallerSlot);
    this->resSelectionCallerSlot.SetCompatibleCall<ResidueSelectionCallDescription>();
    this->MakeSlotAvailable(&this->resSelectionCallerSlot);
    
    this->diagramCalleeSlot.SetCallback(DiagramCall::ClassName(), DiagramCall::FunctionName(DiagramCall::CallForGetData), &VolumeMeshRenderer::GetDiagramData);
    this->MakeSlotAvailable(&this->diagramCalleeSlot);
    this->splitMergeCalleeSlot.SetCallback(SplitMergeCall::ClassName(), SplitMergeCall::FunctionName(SplitMergeCall::CallForGetData), &VolumeMeshRenderer::GetSplitMergeData);
    this->MakeSlotAvailable(&this->splitMergeCalleeSlot);
    this->centerLineDiagramCalleeSlot.SetCallback(DiagramCall::ClassName(), DiagramCall::FunctionName(DiagramCall::CallForGetData), &VolumeMeshRenderer::GetCenterLineDiagramData);
    this->MakeSlotAvailable(&this->centerLineDiagramCalleeSlot);

    // parameters
    param::EnumParam* fm = new param::EnumParam(this->polygonMode);
    fm->SetTypePair(POINT, "Point");
    fm->SetTypePair(LINE, "Line");
    fm->SetTypePair(FILL, "Fill");
    this->polygonModeParam.SetParameter(fm);
    this->blendItParam.SetParameter(new param::BoolParam(this->blendIt));
    this->showCenterlinesParam.SetParameter(new param::BoolParam(false));
    this->showNormalsParam.SetParameter(new param::BoolParam(this->showNormals));
    this->showCentroidsParam.SetParameter(new param::BoolParam(this->showCentroids));
    this->aoThresholdParam.SetParameter(new param::FloatParam(this->aoThreshold, vislib::math::FLOAT_EPSILON));
    this->isoValueParam.SetParameter(new param::FloatParam(this->isoValue, vislib::math::FLOAT_EPSILON));
    this->maxDistanceParam.SetParameter(new param::FloatParam(this->maxDistance, vislib::math::FLOAT_EPSILON));
    this->maxDeltaDistanceParam.SetParameter(new param::FloatParam(this->maxDeltaDistance, vislib::math::FLOAT_EPSILON));
    // fill color table with default values and set the filename param
    vislib::StringA filename("colors.txt");
    Color::ReadColorTableFromFile(filename, this->colorTable);
    this->colorTableFileParam.SetParameter(new param::StringParam(A2T(filename)));
    // make all slots available
    this->MakeSlotAvailable(&this->polygonModeParam);
    this->MakeSlotAvailable(&this->blendItParam);
    this->MakeSlotAvailable(&this->showCenterlinesParam);
    this->MakeSlotAvailable(&this->showNormalsParam);
    this->MakeSlotAvailable(&this->showCentroidsParam);
    this->MakeSlotAvailable(&this->aoThresholdParam);
    this->MakeSlotAvailable(&this->isoValueParam);
    this->MakeSlotAvailable(&this->maxDistanceParam);
    this->MakeSlotAvailable(&this->maxDeltaDistanceParam);
    this->MakeSlotAvailable(&this->colorTableFileParam);
    
    this->alphaParam.SetParameter( new param::FloatParam( 0.2f, 0.0f, 1.0f));
    this->MakeSlotAvailable( &this->alphaParam);
    
	this->haloEnableParam.SetParameter( new param::BoolParam(true));
    this->MakeSlotAvailable( &this->haloEnableParam);
	this->haloAlphaParam.SetParameter( new param::FloatParam( 0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable( &this->haloAlphaParam);
	this->haloColorParam.SetParameter( new param::StringParam( "#146496"));
    this->MakeSlotAvailable( &this->haloColorParam);

    // coloring modes
    this->currentColoringMode0 = Color::CHAIN;
    this->currentColoringMode1 = Color::ELEMENT;
    param::EnumParam *cm0 = new param::EnumParam ( int ( this->currentColoringMode0) );
    param::EnumParam *cm1 = new param::EnumParam ( int ( this->currentColoringMode1) );
    MolecularDataCall *mol = new MolecularDataCall();
    BindingSiteCall *bs = new BindingSiteCall();
    unsigned int cCnt;
    Color::ColoringMode cMode;
    for( cCnt = 0; cCnt < Color::GetNumOfColoringModes( mol, bs); ++cCnt) {
        cMode = Color::GetModeByIndex( mol, bs, cCnt);
        cm0->SetTypePair( cMode, Color::GetName( cMode).c_str());
        cm1->SetTypePair( cMode, Color::GetName( cMode).c_str());
    }
    cm0->SetTypePair( -1, "SurfaceFeature");
    cm1->SetTypePair( -1, "SurfaceFeature");
    delete mol;
    delete bs;
    this->coloringModeParam0 << cm0;
    this->coloringModeParam1 << cm1;
    this->MakeSlotAvailable( &this->coloringModeParam0);
    this->MakeSlotAvailable( &this->coloringModeParam1);
    
    // Color weighting parameter
    this->cmWeightParam.SetParameter(new param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->cmWeightParam);

    // make the rainbow color table
    Color::MakeRainbowColorTable( 100, this->rainbowColors);

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::StringParam( "#146496"));
    this->MakeSlotAvailable( &this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::StringParam( "#f0f0f0"));
    this->MakeSlotAvailable( &this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::StringParam( "#ae3b32"));
    this->MakeSlotAvailable( &this->maxGradColorParam);

    // en-/disable positional interpolation
    this->interpolParam.SetParameter(new param::BoolParam( true));
    this->MakeSlotAvailable( &this->interpolParam);
    
    // Quicksurf params
    this->qualityParam.SetParameter( new param::IntParam( 1, 0, 4));
    this->MakeSlotAvailable( &this->qualityParam);

    this->radscaleParam.SetParameter( new param::FloatParam( 1.0f, 0.0f));
    this->MakeSlotAvailable( &this->radscaleParam);

    this->gridspacingParam.SetParameter( new param::FloatParam( 1.0f, vislib::math::FLOAT_EPSILON));
    this->MakeSlotAvailable( &this->gridspacingParam);
    
    // parameter for minimum distance of center line node to molecular surface
    this->minDistCenterLineParam.SetParameter(new param::FloatParam( 1.4f, 0.0f));
    this->MakeSlotAvailable(&this->minDistCenterLineParam);

    numvoxels[0] = 128;
    numvoxels[1] = 128;
    numvoxels[2] = 128;

    origin[0] = 0.0f;
    origin[1] = 0.0f;
    origin[2] = 0.0f;

    xaxis[0] = 1.0f;
    xaxis[1] = 0.0f;
    xaxis[2] = 0.0f;

    yaxis[0] = 0.0f;
    yaxis[1] = 1.0f;
    yaxis[2] = 0.0f;

    zaxis[0] = 0.0f;
    zaxis[1] = 0.0f;
    zaxis[2] = 1.0f;
   
    cudaqsurf = 0;
    
    timer = wkf_timer_create();

    // area threshold param
    this->areaThresholdParam.SetParameter(new param::FloatParam( 1.0f, 0.0f));
    this->MakeSlotAvailable( &this->areaThresholdParam);

    // TODO: fis this --> compute correct bounding box over all time steps
    this->dataBBox.Set(
        36.990002f - 20.0f,
        69.709999f - 20.0f,
        83.580002f - 20.0f,
        36.990002f + 20.0f,
        69.709999f + 20.0f,
        83.580002f + 20.0f);

    // set default capacity and capacity increment for lists
    this->featureList.AssertCapacity(100);
    this->featureList.SetCapacityIncrement(100);
    this->splitMergeList.AssertCapacity(100);
    this->splitMergeList.SetCapacityIncrement(100);
    this->transitionList.AssertCapacity(100);
    this->transitionList.SetCapacityIncrement(100);
    this->featureSelection.AssertCapacity(100);
    this->featureSelection.SetCapacityIncrement(100);
    this->featureVisibility.AssertCapacity(100);
    this->featureVisibility.SetCapacityIncrement(100);
}

/*
 * VolumeMeshRenderer::~VolumeMeshRenderer(DTOR)
 */
VolumeMeshRenderer::~VolumeMeshRenderer(void) {
    this->Release();
}

/*
 * VolumeMeshRenderer::create
 */
bool VolumeMeshRenderer::create(void) {
    using vislib::sys::Log;
    using namespace vislib::graphics::gl;

    if (!ogl_IsVersionGEQ(2,0) || !areExtsAvailable("GL_ARB_vertex_buffer_object GL_EXT_framebuffer_object")) {
        return false;
    }
    if  (!vislib::graphics::gl::GLSLGeometryShader::InitialiseExtensions()) {
        return false;
    }
    if  (!vislib::graphics::gl::FramebufferObject::InitialiseExtensions()) {
        return false;
    }

    ShaderSource vertSrc;
    ShaderSource geomSrc;
    ShaderSource fragSrc;

    // Load normal shader
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("volumemesh::normalVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for normal shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("volumemesh::normalGeometry", geomSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load geometry shader source for normal shader");
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("volumemesh::normalFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load fragment shader source for normal shader");
        return false;
    }
    try {
        if (!this->normalShader.Compile(vertSrc.Code(), vertSrc.Count(), geomSrc.Code(), geomSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
        this->normalShader.SetProgramParameter(GL_GEOMETRY_INPUT_TYPE_EXT , GL_TRIANGLES);
        this->normalShader.SetProgramParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_LINE_STRIP);
        this->normalShader.SetProgramParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 6);
        if( !this->normalShader.Link() ) {
            throw vislib::Exception( "Generic creation failure", __FILE__, __LINE__ );   
        }
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create normal shader: %s\n", e.GetMsgA());
        return false;
    }

    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::perpixellightVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for per pixel lighting shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::std::perpixellightFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for per pixel lighting shader", this->ClassName());
        return false;
    }
    try {
        if (!this->lightShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create per pixel lighting shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::halo::GenerateVertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for halo generation shader", this->ClassName());
        return false;
    }
    if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::halo::GenerateFragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load fragment shader source for halo generation shader", this->ClassName());
        return false;
    }
    try {
        if (!this->haloGenerateShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
        }
    }
    catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create halo generation shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

	// Try to load shader for gaussian filter (horizontal)
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: gaussian filter (horizontal)", this->ClassName());
		return false;
	}
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::halo::fragmentHoriz", fragSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: gaussian filter (horizontal)", this->ClassName());
		return false;
	}
	try {
		if(!this->haloGaussianHoriz.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(vislib::Exception e){
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load shader for gaussian filter (vertical)
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: gaussian filter (vertical)", this->ClassName());
		return false;
	}
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::halo::fragmentVert", fragSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: gaussian filter (vertical)", this->ClassName());
		return false;
	}
	try {
		if(!this->haloGaussianVert.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(vislib::Exception e){
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}
	
	// Try to load shader for substract filter
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: halo substract", this->ClassName());
		return false;
	}
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::halo::SubstractFragment", fragSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: halo substract", this->ClassName());
		return false;
	}
	try {
		if(!this->haloDifferenceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(vislib::Exception e){
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

	// Try to load shader for grow filter
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("proteinDeferred::gaussian::vertex", vertSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load vertex shader source: halo grow filter", this->ClassName());
		return false;
	}
	if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein_cuda::halo::growFragment", fragSrc)) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to load fragment shader source: halo grow filter", this->ClassName());
		return false;
	}
	try {
		if(!this->haloGrowShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
	}
	catch(vislib::Exception e){
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
		return false;
	}

    // Create OpenGL interoperable CUDA device.
    //cudaError_t cuerr = cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId());
    //if( cuerr != cudaError::cudaSuccess ) {
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: cudaGLSetGLDevice: %s\n", this->ClassName(), cudaGetErrorString( cuerr));
    //    return false;
    //}
    
    // log thrust version
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Thrust Version: %d.%d.%d\n", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION);

    // Allocate CUDA memory for labeling.
    CUDA_VERIFY(cudaMalloc(&modified, sizeof(bool)));
    CUDA_VERIFY(cudaMalloc(&segmentsRemoved, sizeof(bool)));

    CUDA_VERIFY(cudaEventCreate(&start));    // TIMING
    CUDA_VERIFY(cudaEventCreate(&stop));     // TIMING

    return true;
}

/*
 * VolumeMeshRenderer::release
 */
void VolumeMeshRenderer::release(void) {
    if (cudaqsurf) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) cudaqsurf;
        delete cqs;
    }
    wkf_timer_destroy(timer);

    using vislib::sys::Log;

    if( centroidColorsLast )
        delete[] centroidColorsLast;
    if( centroidLabelsLast )
        delete[] centroidLabelsLast;
    if( centroidsLast )
        delete[] centroidsLast;
    if( featureListIdx )
        delete[] featureListIdx;

    try {
        ValidateActiveCubeMemory(0);
        ValidateVertexMemory(0);
        ValidateCentroidMemory(0);
        this->volumeSize = make_uint3(0, 0, 0);
        ValidateCubeMemory();
        
        if( modified )
            CUDA_VERIFY(cudaFree(modified));
        if( segmentsRemoved )
            CUDA_VERIFY(cudaFree(segmentsRemoved));
    } catch(vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to release CUDA resources: %s\n", e.GetMsgA());
    }
}

/*
 * ProteinRenderer::GetExtents
 */
bool VolumeMeshRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (!cr3d) {
        return false;
    }

    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if(!mol) {
        return false;
    }
    // Try to call the molecular data
    if(!(*mol)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    float scale;
    if(!vislib::math::IsEqual(mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }  
    
    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(mol->FrameCount());

    return true;
}

/*
 * VolumeMeshRenderer::Render
 */
bool VolumeMeshRenderer::Render(Call& call) {
    using vislib::sys::Log;

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D *>(&call);
    if (!cr3d) {
        return false;
    }
    
    if( setCUDAGLDevice ) {
#ifdef _WIN32
        if( cr3d->IsGpuAffinity() ) {
            HGPUNV gpuId = cr3d->GpuAffinity<HGPUNV>();
            int devId;
            cudaWGLGetDevice( &devId, gpuId);
            cudaGLSetGLDevice( devId);
        } else {
            cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId());
        }
#else
        cudaGLSetGLDevice( cudaUtilGetMaxGflopsDeviceId());
#endif
        printf( "cudaGLSetGLDevice: %s\n", cudaGetErrorString( cudaGetLastError()));
        setCUDAGLDevice = false;
    }

    // get camera information
    this->cameraInfo = cr3d->GetCameraParameters();

    float callTime = cr3d->Time();
    // get pointer to MolecularDataCall
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL) return false;
    
    // get pointer to BindingSiteCall
    BindingSiteCall *bs = this->bsDataCallerSlot.CallAs<BindingSiteCall>();
    if( bs ) {
        (*bs)(BindingSiteCall::CallForGetData);
    }

    int cnt;

    // set call time
    mol->SetCalltime(callTime);
    // set frame ID and call data
    mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) return false;

    // resize selection array, if necessary
    if( this->atomSelectionCnt < mol->AtomCount() ) {
        if( this->atomSelection )
            delete[] this->atomSelection;
        this->atomSelection = new bool[mol->AtomCount()];
        this->atomSelectionCnt = mol->AtomCount();
    }
    // reset selection array
    memset( this->atomSelection, 0, mol->AtomCount() * sizeof(bool));
    // try to get the residue selection
    resSelectionCall = this->resSelectionCallerSlot.CallAs<ResidueSelectionCall>();
    if (resSelectionCall != nullptr) {
        (*resSelectionCall)(ResidueSelectionCall::CallForGetSelection);
        // try to match selection <chainID,resNum> to <id> from MolecularDataCall
        vislib::Array<ResidueSelectionCall::Residue> *resSelPtr = resSelectionCall->GetSelectionPointer();
        if( resSelPtr ) {
            for( unsigned int i = 0; i < resSelPtr->Count(); i++) {
                // do not search for the correct amino acid, if the id was already set (mark atoms directly)
                if( (*resSelPtr)[i].id >= 0 ) {
                    unsigned int firstAtomIdx = mol->Residues()[(*resSelPtr)[i].id]->FirstAtomIndex();
                    unsigned int lastAtomIdx = firstAtomIdx + mol->Residues()[(*resSelPtr)[i].id]->AtomCount();
                    for( unsigned int aCnt = firstAtomIdx; aCnt < lastAtomIdx; aCnt++ ) {
                        this->atomSelection[aCnt] = true;
                    }
                    continue;
                }
                int chainIdx = -1;
                // loop over chains to find chainID
                for( unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++ ) {
                    if( mol->Chains()[cCnt].Name() == (*resSelPtr)[i].chainID ) {
                        chainIdx = cCnt;
                        break;
                    }
                }
                // do nothing if no matching chain was found
                if( chainIdx < 0 ) continue;
                // loop over amino acids to find the correct one
                unsigned int firstResIdx = mol->Molecules()[mol->Chains()[chainIdx].FirstMoleculeIndex()].FirstResidueIndex();
                unsigned int lastResIdx = mol->Molecules()[mol->Chains()[chainIdx].FirstMoleculeIndex()+mol->Chains()[chainIdx].MoleculeCount()-1].FirstResidueIndex()
                    + mol->Molecules()[mol->Chains()[chainIdx].FirstMoleculeIndex()+mol->Chains()[chainIdx].MoleculeCount()-1].ResidueCount();
                for( unsigned int rCnt = firstResIdx; rCnt < lastResIdx; rCnt++ ) {
                    unsigned int firstAtomIdx = mol->Residues()[rCnt]->FirstAtomIndex();
                    unsigned int lastAtomIdx = firstAtomIdx + mol->Residues()[rCnt]->AtomCount();
                    if( mol->Residues()[rCnt]->OriginalResIndex() == (*resSelPtr)[i].resNum ) {
                        (*resSelPtr)[i].id = rCnt;
                        // mark all atoms of the current amino acid
                        for( unsigned int aCnt = firstAtomIdx; aCnt < lastAtomIdx; aCnt++ ) {
                            this->atomSelection[aCnt] = true;
                        }
                    } 
                }
            }
            if( resSelPtr->Count() == 0 ) {
                this->resSelectionCall = nullptr;
            }
        } else {
            this->resSelectionCall = nullptr;
        }
    }


    // check if atom count is zero
    if( mol->AtomCount() == 0 ) return true;
    // get positions of the first frame
    float *pos0 = new float[mol->AtomCount() * 3];
    memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
    // set next frame ID and get positions of the second frame
    if( ( ( static_cast<int>( callTime) + 1) < int( mol->FrameCount()) ) &&
        this->interpolParam.Param<param::BoolParam>()->Value() )
        mol->SetFrameID(static_cast<int>( callTime) + 1);
    else
        mol->SetFrameID(static_cast<int>( callTime));
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        delete[] pos0;
        return false;
    }
    float *pos1 = new float[mol->AtomCount() * 3];
    memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));

    // interpolate atom positions between frames
    float *posInter = new float[mol->AtomCount() * 3];
    float inter = callTime - static_cast<float>(static_cast<int>( callTime));
    float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
        vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
        mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
    for( cnt = 0; cnt < int( mol->AtomCount()); ++cnt ) {
        if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
                std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
                std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
            posInter[3*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
            posInter[3*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
            posInter[3*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
        } else if( inter < 0.5f ) {
            posInter[3*cnt+0] = pos0[3*cnt+0];
            posInter[3*cnt+1] = pos0[3*cnt+1];
            posInter[3*cnt+2] = pos0[3*cnt+2];
        } else {
            posInter[3*cnt+0] = pos1[3*cnt+0];
            posInter[3*cnt+1] = pos1[3*cnt+1];
            posInter[3*cnt+2] = pos1[3*cnt+2];
        }
    }

    // Try to get volume texture.
    /*
    VolumeSliceCall *vol = this->volDataCallerSlot.CallAs<VolumeSliceCall>();
    GLuint volumeTex = 0;
    if (vol) {
        if(!(*vol)(VolumeSliceCall::CallForGetData)) {
            return false;
        }
        volumeTex = vol->getVolumeTex();
    } else {
        return false;
    }
    */

    ParameterRefresh( mol, bs);
    
    // recompute color table, if necessary (i.e. the atom count has changed)
    if( this->atomColorTable.Count()/3 < mol->AtomCount() ) {
        if( this->currentColoringMode0 < 0 ) {
            if( this->currentColoringMode1 < 0 ) {
                // Color by surface feature -> set all colors to white
                this->atomColorTable.SetCount( mol->AtomCount() * 3);
                for( unsigned int i = 0; i < mol->AtomCount() * 3; i++ ) {
                    this->atomColorTable[i] = 1.0f;
                }
            } else {
                // only color by color mode 1
                Color::MakeColorTable( mol,
                    static_cast<Color::ColoringMode>(this->currentColoringMode1),
                    this->atomColorTable, this->colorTable, this->rainbowColors,
                    this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(),
                    true, bs);
            }
        } else {
            if( this->currentColoringMode1 < 0 ) {
                // only color by color mode 0
                Color::MakeColorTable( mol,
                    static_cast<Color::ColoringMode>(this->currentColoringMode0),
                    this->atomColorTable, this->colorTable, this->rainbowColors,
                    this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(),
                    true, bs);
            } else {
            // Mix two coloring modes
            Color::MakeColorTable( mol,
                static_cast<Color::ColoringMode>(this->currentColoringMode0),
                static_cast<Color::ColoringMode>(this->currentColoringMode1),
                cmWeightParam.Param<param::FloatParam>()->Value(),       // weight for the first cm
                1.0f - cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
                this->atomColorTable, this->colorTable, this->rainbowColors,
                this->minGradColorParam.Param<param::StringParam>()->Value(),
                this->midGradColorParam.Param<param::StringParam>()->Value(),
                this->maxGradColorParam.Param<param::StringParam>()->Value(),
                true, bs);
            }
        }
    }
    
    // TEST
#ifdef TEST
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0);
#endif

    // brushing linking selection fake
    static int lastSelection = -1;
    
    //SplitMergeCall *smc = dynamic_cast<SplitMergeCall*>(&call);
    //DiagramCall *dc = dynamic_cast<DiagramCall*>(&call);

    //if (

    // calculate surface
    if( !cudaqsurf ) {
        cudaqsurf = new CUDAQuickSurf();
    }
    this->calcMap( mol, posInter, 
        this->qualityParam.Param<param::IntParam>()->Value(),
        this->radscaleParam.Param<param::FloatParam>()->Value(),
        this->gridspacingParam.Param<param::FloatParam>()->Value(),
        this->isoValueParam.Param<param::FloatParam>()->Value(),
        false);

    cudaDeviceSynchronize(); // Paranoia

    // store CUDA density map size
    CUDAQuickSurf *cqs = (CUDAQuickSurf *) cudaqsurf;
    uint3 hVolSize;
    hVolSize.x = numvoxels[0];
    hVolSize.y = numvoxels[1];
    hVolSize.z = numvoxels[2];
    this->volumeSize = hVolSize;
    CUDA_VERIFY( copyVolSizeToDevice( hVolSize));
    cudaDeviceSynchronize();
    // TEST: read back dVolumeSize to verify correct copy!
    hVolSize = make_uint3( 0, 0, 0);
    CUDA_VERIFY( copyVolSizeFromDevice( hVolSize));
    cudaDeviceSynchronize();
    if( hVolSize.x != numvoxels[0] || hVolSize.y != numvoxels[1] || hVolSize.z != numvoxels[2] ) {
        Log::DefaultLog.WriteError( "%s: cudaMemcpyFromSymbol failed!\n", this->Name().PeekBuffer());
        delete[] pos0;
        delete[] pos1;
        delete[] posInter;
        mol->Unlock();
        return false;
    }
    
    // -------- START create mesh -----
    try {
        float time = cr3d->Time();
        if (time != lastTime) {
            // Compute scaling and translation to object space.
            vislib::math::Vector<float, 3> translation(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
            vislib::math::Vector<float, 3> scale(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension());
            scale *= vislib::math::Vector<float, 3>(1.0f / this->volumeSize.x, 1.0f / this->volumeSize.y, 1.0f / this->volumeSize.z);
            
            //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Scale: %f %f %f\n", scale.X(), scale.Y(), scale.Z());
            // Compute AO volume and update mesh.
            this->aoShader.setVolumeSize( static_cast<unsigned int>(mol->AccessBoundingBoxes().ObjectSpaceBBox().Width() / 10.0f),
                static_cast<unsigned int>(mol->AccessBoundingBoxes().ObjectSpaceBBox().Height() / 10.0f),
                static_cast<unsigned int>(mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth() / 10.0f));
            float* aoVolumeHost = this->aoShader.createVolume(*mol);
            if (!UpdateMesh( cqs->getMap(), translation, scale, aoVolumeHost, mol, cqs->getNeighborMap())) {
                delete[] aoVolumeHost;
                return false;
            }
            delete[] aoVolumeHost;
        }
        lastTime = time;
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "UpdateMesh failed: %s, %i \n", e.GetMsgA(), e.GetLine());
        return false;
    }
    // -------- END create mesh -----
    
    // TEST
#ifdef TEST
    cudaDeviceSynchronize();
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for UpdateMesh: %f ms\n", time);
#endif
    
    // Push attributes and model matrix.
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushMatrix();
    
    float spec[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, spec);
    glEnable( GL_COLOR_MATERIAL);

    // Scale model properly.
    float scale;
    if(!vislib::math::IsEqual(mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) { 
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }
    
    
    //
    //vislib::math::Vector<float, 3> translation(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
    //vislib::math::Vector<float, 3> scale2(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension());
    //glTranslatef(translation.X(), translation.Y(), translation.Z());
    //glScalef(scale2.X(), scale2.Y(), scale2.Z());

    glScalef(scale, scale, scale);
    
    // Mesh placement fix
    vislib::math::Vector<float, 3> tmpOrig(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack().PeekCoordinates());
    glTranslatef(tmpOrig.X(), tmpOrig.Y(), tmpOrig.Z());
    vislib::math::Vector<float, 3> tmpDim(mol->AccessBoundingBoxes().ObjectSpaceBBox().GetSize().PeekDimension());
    tmpDim *= vislib::math::Vector<float, 3>(1.0f / this->volumeSize.x, 1.0f / this->volumeSize.y, 1.0f / this->volumeSize.z);
    glScalef(tmpDim.X(), tmpDim.Y(), tmpDim.Z());

    //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Scale: %f\n", scale);
    
    glEnable(GL_DEPTH_TEST);

//#define DRAW_FEATURE_TRIANGLES
#ifdef DRAW_FEATURE_TRIANGLES
    // TEST feature triangle drawing ...
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glLineWidth(1.0f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f( 1.0f, 1.0f, 1.0f);
    glBegin( GL_TRIANGLES);
    for(unsigned int tCnt = 0; tCnt < this->featureTrianglesCount; tCnt++ ) {
        glVertex3f( this->featureTriangleVerticesHost[3*tCnt].x,  this->featureTriangleVerticesHost[3*tCnt].y,  this->featureTriangleVerticesHost[3*tCnt].z);
        glVertex3f( this->featureTriangleVerticesHost[3*tCnt+1].x,  this->featureTriangleVerticesHost[3*tCnt+1].y,  this->featureTriangleVerticesHost[3*tCnt+1].z);
        glVertex3f( this->featureTriangleVerticesHost[3*tCnt+2].x,  this->featureTriangleVerticesHost[3*tCnt+2].y,  this->featureTriangleVerticesHost[3*tCnt+2].z);
    }
    glEnd();
    // ... TEST feature triangle drawing
#endif // DRAW_FEATURE_TRIANGLES

//#define DRAW_FEATURE_VERTICES
#ifdef DRAW_FEATURE_VERTICES
    // TEST feature vertex drawing ...
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glPointSize(5.0f);
    glColor3f( 1.0f, 1.0f, 1.0f);
    glBegin( GL_POINTS);
    for(unsigned int tCnt = 0; tCnt < this->featureVertexCntNew; tCnt++ ) {
        glVertex3f( this->featureTriangleVerticesHost[tCnt].x,  this->featureTriangleVerticesHost[tCnt].y,  this->featureTriangleVerticesHost[tCnt].z);
    }
    glEnd();
    // draw edges
    glLineWidth(1.0f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f( 1.0f, 1.0f, 0.0f);
    //glEnableClientState(GL_VERTEX_ARRAY);
    //glVertexPointer(4, GL_FLOAT, 0, (float*)this->featureTriangleVerticesHost);
    //glDrawElements(GL_LINES, this->edgeCount * 2, GL_UNSIGNED_INT, (unsigned int*)this->featureTriangleEdgesHost);
    //glDisableClientState(GL_VERTEX_ARRAY);
    glBegin(GL_LINES);
    for(unsigned int eCnt = 0; eCnt < this->edgeCount; eCnt++ ) {
        if( this->featureTriangleEdgeCountHost[eCnt] < 2 )
            glColor3f( 1.0f, 0.5f, 0.0f);
        else
            glColor3f( 1.0f, 1.0f, 0.0f);
        glVertex3f( this->featureTriangleVerticesHost[this->featureTriangleEdgesHost[eCnt].x].x,
                    this->featureTriangleVerticesHost[this->featureTriangleEdgesHost[eCnt].x].y,
                    this->featureTriangleVerticesHost[this->featureTriangleEdgesHost[eCnt].x].z);
        glVertex3f( this->featureTriangleVerticesHost[this->featureTriangleEdgesHost[eCnt].y].x,
                    this->featureTriangleVerticesHost[this->featureTriangleEdgesHost[eCnt].y].y,
                    this->featureTriangleVerticesHost[this->featureTriangleEdgesHost[eCnt].y].z);
    }
    // ... TEST feature vertex drawing
    glEnd(); // GL_LINES
#endif // DRAW_FEATURE_VERTICES

    // TEST center line drawing ...
#define DRAW_CENTERLINE
#ifdef DRAW_CENTERLINE
    if( this->showCenterlinesParam.Param<param::BoolParam>()->Value() ) {
        glDisable(GL_CULL_FACE);
        glDisable(GL_LIGHTING);
        glPointSize( 10.0f);
        glEnable(GL_POINT_SMOOTH);
        glLineWidth( 5.0f);
        for( unsigned int fCnt = 0; fCnt < this->clNodes.Count(); fCnt++ ) {
            unsigned int clnCnt = static_cast<unsigned int>(this->clNodes[fCnt].size());
            unsigned int curClnCnt = 0;
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glBegin( GL_POINTS);
            for( auto nodes : this->clNodes[fCnt]) {
                // draw only center line nodes that were created by a full ring
                //if( !nodes->isRing ) continue;
                if( this->clg[fCnt]->fType == CenterLineGenerator::CHANNEL )
                    //glColor3f( ( 1.0f / clnCnt) * curClnCnt, 1.0f, 0.0f);
                    glColor3f( 0.0f, 1.0f, 0.0f);
                else if( this->clg[fCnt]->fType == CenterLineGenerator::POCKET )
                    //glColor3f( 1.0f, 0.0f, ( 1.0f / clnCnt) * curClnCnt);
                    glColor3f( 1.0f, 0.0f, 1.0f);
                else // this->clg[fCnt]->fType == CenterLineGenerator::CAVITY
                    //glColor3f( 0.0f, 1.0f, ( 1.0f / clnCnt) * curClnCnt);
                    glColor3f( 0.0f, 1.0f, 1.0f);
                if( nodes->isStartNode )
                    glColor3f( 1.0f, 1.0f, 1.0f);
                else continue; // DRAW ONLY START AND END NODES OF THE CENTERLINE
                glVertex3fv( nodes->p.PeekComponents());
                curClnCnt++;
            }
            glEnd();
            glBegin( GL_LINES);
            clnCnt = static_cast<unsigned int>(this->clEdges[fCnt].size());
            curClnCnt = 0;
            for( auto ed : this->clEdges[fCnt]) {
                if( this->clg[fCnt]->fType == CenterLineGenerator::CHANNEL )
                    glColor3f( 0.0f, 1.0f, 0.0f);
                else if( this->clg[fCnt]->fType == CenterLineGenerator::POCKET )
                    glColor3f( 1.0f, 0.0f, 1.0f);
                else // this->clg[fCnt]->fType == CenterLineGenerator::CAVITY
                    glColor3f( 0.0f, 1.0f, 1.0f);
                glVertex3fv( ed->node1->p.PeekComponents());
                //if( this->clg[fCnt]->fType == CenterLineGenerator::CHANNEL )
                //    glColor3f( ( 1.0f / clnCnt) * (curClnCnt+1), 1.0f, 0.0f);
                //else if( this->clg[fCnt]->fType == CenterLineGenerator::POCKET )
                //    glColor3f( 1.0f, 0.0f, ( 1.0f / clnCnt) * (curClnCnt+1));
                //else // this->clg[fCnt]->fType == CenterLineGenerator::CAVITY
                //    glColor3f( 0.0f, 1.0f, ( 1.0f / clnCnt) * (curClnCnt+1));
                glVertex3fv( ed->node2->p.PeekComponents());
                curClnCnt++;
            }
            glEnd();
        }
        glDisable(GL_POINT_SMOOTH);
        // ... TEST center line drawing
    }
#endif // DRAW_CENTERLINE

//#define DRAW_CENTER_LINE_RINGS
#ifdef DRAW_CENTER_LINE_RINGS
    // TEST center line branches ...
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glLineWidth(3.0f);
    glColor3f( 0.0f, 1.0f, 1.0f);
    glBegin( GL_LINES);
    for( unsigned int fCnt = 0; fCnt < this->clNodes.Count(); fCnt++ ) {
        unsigned int clnCnt = this->clNodes[fCnt].size();
        unsigned int curClnCnt = 0;
        for( auto branch : clg[fCnt]->allBranches) {
            if(curClnCnt % 2)
                glColor3f( 0.0f, ( 1.0f / clnCnt) * curClnCnt, 1.0f);
            else
                glColor3f( 1.0f, ( 1.0f / clnCnt) * curClnCnt, 0.0f);
            for(auto edge : branch->edges ) {
                glVertex3fv( edge->getNode1()->p.PeekComponents());
                glVertex3fv( edge->getNode2()->p.PeekComponents());
            }
            curClnCnt++;
        }
    }
    glEnd();
    // ... TEST center line branches
#endif // DRAW_CENTER_LINE_RINGS
    
//#define DRAW_FIRST_RING
#ifdef DRAW_FIRST_RING
    // TEST center line first ring ...
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glLineWidth(5.0f);
    glColor3f( 1.0f, 0.0f, 1.0f);
    glBegin( GL_LINES);
    for( unsigned int fCnt = 0; fCnt < this->clNodes.Count(); fCnt++ ) {
        for( auto edge : clg[fCnt]->freeEdgeRing) {
            glVertex3fv( edge->getNode1()->p.PeekComponents());
            glVertex3fv( edge->getNode2()->p.PeekComponents());
        }
    }
    glEnd();
    // ... TEST center line first ring
#endif // DRAW_FIRST_RING

    // sort the mesh for transparent rendering
    if( this->blendIt ) {
        this->SortTriangleMesh();
    }

    glPointSize( 1.0f);

    // Bind VBOs.
    glBindBufferARB(GL_ARRAY_BUFFER, this->positionVbo);
    glVertexPointer(4, GL_FLOAT, sizeof(GLfloat) * 4, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER, this->normalVbo);
    glNormalPointer(GL_FLOAT, sizeof(GLfloat) * 4, 0);
    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER, this->colorVbo);
    glColorPointer(4, GL_FLOAT, sizeof(GLfloat) * 4, 0);
    glEnableClientState(GL_COLOR_ARRAY);

    // Render mesh.
    if (this->blendIt) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //glBlendFunc(GL_ONE, GL_ONE);
        glEnable(GL_BLEND);
        //glDisable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_BLEND);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0.5f);
    }
    glDisable(GL_CULL_FACE);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    this->lightShader.Enable();
    switch (polygonMode) {
    case POINT:
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        break;
    case LINE:
        glLineWidth(1.0f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        break;
    case FILL:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        break;
    }
    glDrawArrays(GL_TRIANGLES, 0, this->vertexCount);
    this->lightShader.Disable();

	// Render normals, if requested.
    if (this->showNormals) {
        glLineWidth(1.0f);
        this->normalShader.Enable();
        glUniform1fARB(this->normalShader.ParameterLocation("normalsLength"), 0.25f);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        this->normalShader.Disable();
    }

    // Render centroids, if requested.
    if (this->showCentroids) {
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glEnable(GL_POINT_SMOOTH);
        // Background
        glPointSize(10.0f);
        glBegin(GL_POINTS);
        for (uint i = 0; i < this->centroidCountLast; ++i) { 
            glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
            glVertex4fv(reinterpret_cast<const GLfloat*>(&this->centroidsLast[i]));
        }
        glEnd();
        // Foreground centroid color)
        glPointSize(7.0f);
        glBegin(GL_POINTS);
        for (uint i = 0; i < this->centroidCountLast; ++i) { 
            glColor4fv(reinterpret_cast<const GLfloat*>(&this->centroidColorsLast[i]));
            glVertex4fv(reinterpret_cast<const GLfloat*>(&this->centroidsLast[i]));
        }
        glEnd();
    }

	if( this->haloEnableParam.Param<param::BoolParam>()->Value() )
	{
		// =============== Query Camera View Dimensions ===============
		if( static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth()) != this->width ||
			static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight()) != this->height ) {
			this->width = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetWidth());
			this->height = static_cast<unsigned int>(cameraInfo->VirtualViewSize().GetHeight());
		}
	 
		// create the fbo, if necessary
		if( !this->haloFBO.IsValid() ) {
			this->haloFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
		}
		if( !this->haloBlurFBO.IsValid() ) {
			this->haloBlurFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
		}
		if( !this->haloBlurFBO2.IsValid() ) {
			this->haloBlurFBO2.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
		}
		// resize the fbo, if necessary
		if( this->haloFBO.GetWidth() != this->width || this->haloFBO.GetHeight() != this->height ) {
			this->haloFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
			this->haloBlurFBO.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
			this->haloBlurFBO2.Create( this->width, this->height, GL_RGBA16F, GL_RGBA, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
		}

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

		vislib::math::Vector<float, 4> haloColor;
		megamol::core::utility::ColourParser::FromString(
			this->haloColorParam.Param<param::StringParam>()->Value(),
            haloColor.PeekComponents()[0],
            haloColor.PeekComponents()[1],
            haloColor.PeekComponents()[2]);

		haloColor.SetW( this->haloAlphaParam.Param<param::FloatParam>()->Value() );
		
		this->haloFBO.Enable();
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
			glDisable( GL_DEPTH_TEST );

			this->haloGenerateShader.Enable();
			glUniform4fv(this->haloGenerateShader.ParameterLocation("haloColor"), 1, haloColor.PeekComponents() );
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawArrays(GL_TRIANGLES, 0, this->vertexCount);
			this->haloGenerateShader.Disable();
		this->haloFBO.Disable();


		this->haloBlurFBO2.Enable();
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
			glClear( GL_COLOR_BUFFER_BIT );
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, this->haloFBO.GetColourTextureID());
			
			this->haloGrowShader.Enable();
			glUniform1i(this->haloGrowShader.ParameterLocation("sourceTex"), 0);
			glUniform2f(this->haloGaussianHoriz.ParameterLocation("screenResInv"), 1.0f/this->width, 1.0f/this->height);

			glRecti(-1, -1, 1, 1); // Draw screen quad
			this->haloGrowShader.Disable();
			glBindTexture(GL_TEXTURE_2D, 0);
		this->haloBlurFBO2.Disable();

		this->haloBlurFBO.Enable();
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
			glClear( GL_COLOR_BUFFER_BIT );
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, this->haloBlurFBO2.GetColourTextureID());
			
			this->haloGaussianHoriz.Enable();
			glUniform1i(this->haloGaussianHoriz.ParameterLocation("sourceTex"), 0);
			glUniform2f(this->haloGaussianHoriz.ParameterLocation("screenResInv"), 1.0f/this->width, 1.0f/this->height);

			glRecti(-1, -1, 1, 1); // Draw screen quad
			this->haloGaussianHoriz.Disable();
			glBindTexture(GL_TEXTURE_2D, 0);
		this->haloBlurFBO.Disable();


		this->haloBlurFBO2.Enable();
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f);
			glClear( GL_COLOR_BUFFER_BIT );
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, this->haloBlurFBO.GetColourTextureID());
			
			this->haloGaussianVert.Enable();
			glUniform1i(this->haloGaussianVert.ParameterLocation("sourceTex"), 0);
			glUniform2f(this->haloGaussianVert.ParameterLocation("screenResInv"), 1.0f/this->width, 1.0f/this->height);

			glRecti(-1, -1, 1, 1); // Draw screen quad
			this->haloGaussianVert.Disable();
			glBindTexture(GL_TEXTURE_2D, 0);
		this->haloBlurFBO2.Disable();
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, this->haloFBO.GetColourTextureID());
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, this->haloBlurFBO2.GetColourTextureID());

		this->haloDifferenceShader.Enable();
		glUniform1i(this->haloDifferenceShader.ParameterLocation("originalTex"), 0);
		glUniform1i(this->haloDifferenceShader.ParameterLocation("blurredTex"), 1);
		glRecti(-1, -1, 1, 1); // Draw screen quad
		this->haloDifferenceShader.Disable();
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);
		
		
		//this->haloBlurFBO2.DrawColourTexture();

		glEnable( GL_DEPTH_TEST );
        glDisable(GL_BLEND);
		/*

		parameter für an/aus
		selectionHalo

		framebuffer->attach()
		clear(black)
		texture1 = render_features (alpha == 1)
		vislib::graphics::gl::GLSLShader gaussianHoriz;
		vislib::graphics::gl::GLSLShader gaussianVert;
		texture 2 = blur/guass (texture1)
		draw(texture2 - texture1)



		this->lightShader.Enable();
		switch (polygonMode) {
		case POINT:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
			break;
		case LINE:
			glLineWidth(1.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			break;
		case FILL:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		}
		glDrawArrays(GL_TRIANGLES, 0, this->vertexCount);
		this->lightShader.Disable();

		*/
	}

    // Unbind VBOs.
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);

    // Pop model matrix and attributes.
    glPopMatrix();
    glPopAttrib();

    glDisable(GL_ALPHA_TEST);

    //CHECK_FOR_OGL_ERROR();
    
    delete[] pos0;
    delete[] pos1;
    delete[] posInter;

    // unlock the current frame
    mol->Unlock();

    return true;
}

void VolumeMeshRenderer::SortTriangleMesh() {
    // Calculate cam pos using last column of inverse modelview matrix
    float3 camPos;
    GLfloat m[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> modelMatrix(&m[0]);
    modelMatrix.Invert();
    camPos.x = modelMatrix.GetAt(0, 3);
    camPos.y = modelMatrix.GetAt(1, 3);
    camPos.z = modelMatrix.GetAt(2, 3);

    copyCamPosToDevice( camPos);

    // Map VBOs.
    size_t resourceSize;
    float4* vertices;
    CUDA_VERIFY(cudaGraphicsMapResources(1, &positionResource, 0));
    CUDA_VERIFY(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vertices), &resourceSize, positionResource));
    float4* normals;
    CUDA_VERIFY(cudaGraphicsMapResources(1, &normalResource, 0));
    CUDA_VERIFY(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&normals), &resourceSize, normalResource));
    float4* colors;
    CUDA_VERIFY(cudaGraphicsMapResources(1, &colorResource, 0));
    CUDA_VERIFY(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&colors), &resourceSize, colorResource));
    cudaDeviceSynchronize(); // Paranoia

    // make copy of vertices to sort normals
    CUDA_VERIFY(cudaMemcpy(verticesCopy, vertices, this->vertexCount * sizeof(float4), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize(); // Paranoia

    // sort everything
    SortTrianglesDevice( this->vertexCount / 3, (float4x3*)vertices, (float4x3*)verticesCopy, (float4x3*)colors, (float4x3*)normals);
    
    // Unmap VBOs.
    CUDA_VERIFY(cudaGraphicsUnmapResources(1, &positionResource, 0));
    CUDA_VERIFY(cudaGraphicsUnmapResources(1, &normalResource, 0));
    CUDA_VERIFY(cudaGraphicsUnmapResources(1, &colorResource, 0));
    cudaDeviceSynchronize(); // Paranoia
}

/*
 * VolumeMeshRenderer::UpdateMesh
 */
bool VolumeMeshRenderer::UpdateMesh(float* densityMap, vislib::math::Vector<float, 3> translation, 
    vislib::math::Vector<float, 3> scale, const float* aoVolumeHost, MolecularDataCall *mol, int* neighborMap) {
    using vislib::sys::Log;
    vislib::sys::PerformanceCounter perf(true);
    
    // allocate buffers for copies from previous step
    this->ValidateOldCubeMemory();

    // copy data of previous time step (if available)
    if( this->cubeCountAllocated > 0 ) {
        CUDA_VERIFY( cudaMemcpy( cubeStatesOld, cubeStates, sizeof(uint) * this->cubeCount, cudaMemcpyDeviceToDevice));
        CUDA_VERIFY( cudaMemcpy( cubeOffsetsOld, cubeOffsets, sizeof(uint) * this->cubeCount, cudaMemcpyDeviceToDevice));
        CUDA_VERIFY( cudaMemcpy( cubeMapOld, cubeMap, sizeof(uint) * this->cubeCount, cudaMemcpyDeviceToDevice));
        CUDA_VERIFY( cudaMemcpy( verticesPerTetrahedronOld, verticesPerTetrahedron, sizeof(uint) * this->activeCubeCount * 6, cudaMemcpyDeviceToDevice));
        CUDA_VERIFY( cudaMemcpy( eqListOld, eqList, sizeof(uint) * this->activeCubeCount * 6, cudaMemcpyDeviceToDevice));
    }

    // allocate buffers for cube data
    ValidateCubeMemory();

    // Validate texture.
    if( !densityMap ) {
        return false;
    }
    CUDA_VERIFY(BindVolumeTexture( densityMap));
    CUDA_VERIFY(BindNeighborAtomTexture( neighborMap));

    cudaDeviceSynchronize(); // Paranoia

    // Copy AO Volume to device.
    cudaExtent aoVolumeHostExtent = make_cudaExtent(this->aoShader.getVolumeSizeX() - 2, 
            this->aoShader.getVolumeSizeY() - 2, 
            this->aoShader.getVolumeSizeZ() - 2);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.dstArray = this->aoVolume;
    copyParams.dstPos = make_cudaPos(1, 1, 1);
    copyParams.srcPtr   = make_cudaPitchedPtr(
        (void*)aoVolumeHost, aoVolumeHostExtent.width * sizeof(float),
        aoVolumeHostExtent.width, aoVolumeHostExtent.height);
    copyParams.extent   = aoVolumeHostExtent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_VERIFY(cudaMemcpy3D(&copyParams));
    CUDA_VERIFY(BindAOVolumeTexture(this->aoVolume));

    cudaDeviceSynchronize(); // Paranoia

    // ========================================================================
    // Phase 1: classify cubes.
    // ========================================================================
    this->cubeCount = this->volumeSize.x * this->volumeSize.y * this->volumeSize.z;
    CUDA_VERIFY(ClassifyCubes(cubeStates, this->isoValue, this->cubeCount));
    CUDA_VERIFY(ScanCubes(cubeOffsets, cubeStates, this->cubeCount));
    this->activeCubeCount = CudaGetLast(cubeStates, cubeOffsets, this->cubeCount);
    ValidateActiveCubeMemory(this->activeCubeCount);
    if (this->activeCubeCount == 0) {
        // Unbind, unmap and unregister volume texture.
        CUDA_VERIFY(UnbindAOVolumeTexture());
        CUDA_VERIFY(UnbindVolumeTexture());
        return false;
    }
    //Log::DefaultLog.WriteInfo("Active cubes: %d (%g%%)", this->activeCubeCount,
    //    static_cast<float>(this->activeCubeCount) / cubeCount * 100);

    cudaDeviceSynchronize(); // Paranoia
    
    // ========================================================================
    // Phase 2: compact active cubes.
    // ========================================================================
    CUDA_VERIFY(CompactCubes(cubeMap, cubeOffsets, cubeStates, this->cubeCount));
    
    // ========================================================================
    // Phase 3: classify tetrahedrons (active cubes).
    // ========================================================================
    const uint tetrahedronCount = activeCubeCount * 6;
    CUDA_VERIFY(ClassifyTetrahedronsInACube(verticesPerTetrahedron, cubeMap, this->isoValue,
        activeCubeCount));
    //CUDA_VERIFY(ScanTetrahedrons(vertexOffsets, verticesPerTetrahedron, tetrahedronCount));
    CUDA_VERIFY(ScanTetrahedrons(vertexOffsets, verticesPerTetrahedron, tetrahedronCount+1));
    this->vertexCount = CudaGetLast(verticesPerTetrahedron, vertexOffsets, tetrahedronCount);
    ValidateVertexMemory(this->vertexCount);
    uint triangleCount = this->vertexCount / 3;
    //Log::DefaultLog.WriteInfo("Vertices: # %d", this->vertexCount);

    cudaDeviceSynchronize(); // Paranoia

    // Map VBOs.
    size_t resourceSize;
    float4* vertices;
    CUDA_VERIFY(cudaGraphicsMapResources(1, &positionResource, 0));
    CUDA_VERIFY(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vertices), &resourceSize, positionResource));
    float4* normals;
    CUDA_VERIFY(cudaGraphicsMapResources(1, &normalResource, 0));
    CUDA_VERIFY(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&normals), &resourceSize, normalResource));
    float4* colors;
    CUDA_VERIFY(cudaGraphicsMapResources(1, &colorResource, 0));
    CUDA_VERIFY(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&colors), &resourceSize, colorResource));

    cudaDeviceSynchronize(); // Paranoia
    
    // ========================================================================
    // Phase 4: generate triangles.
    // ========================================================================
    CUDA_VERIFY(GenerateTriangles(vertices, neighborAtomOfVertex_d, normals, translation.X(), translation.Y(), translation.Z(),
        scale.X(), scale.Y(), scale.Z(), vertexOffsets, cubeMap, this->isoValue, tetrahedronCount));
    cudaDeviceSynchronize(); // Paranoia
    // TODO use cudaMemcpyAsync here?
    CUDA_VERIFY(cudaMemcpy(this->neighborAtomOfVertex, this->neighborAtomOfVertex_d, this->vertexCount*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_VERIFY(ComputeTriangleAO( vertices, normals, triangleAO, triangleCount));
    cudaDeviceSynchronize(); // Paranoia
    
    // ========================================================================
    // Phase 5: label mesh using tetrahedral neighbors.
    // ========================================================================
    bool hostModified;
    CUDA_VERIFY(MeshReset(eqList, refList, tetrahedronCount, vertexOffsets, triangleAO));
    cudaDeviceSynchronize(); // Paranoia
    while (true) {
        CUDA_VERIFY(cudaMemset(modified, false, sizeof(bool)));
        cudaDeviceSynchronize(); // Paranoia
        // TODO check MeshScan for unnecessary parameters/variables
        CUDA_VERIFY(MeshScan(vertices, normals, vertexOffsets, triangleAO, this->aoThreshold, eqList, refList, modified, cubeStates, cubeOffsets, cubeMap, this->isoValue, tetrahedronCount));        
        CUDA_VERIFY(cudaDeviceSynchronize()); // Paranoia
        CUDA_VERIFY(cudaMemcpy(&hostModified, modified, sizeof(bool), cudaMemcpyDeviceToHost));
        if (!hostModified) {
            break;
        }
        CUDA_VERIFY(MeshAnalysis(eqList, refList, tetrahedronCount));
        CUDA_VERIFY(MeshLabeling(eqList, refList, tetrahedronCount));
    }

    cudaDeviceSynchronize(); // Paranoia
    
    // ========================================================================
    // Phase 6: map, reduce and finalize mesh segments to centroids.
    // ========================================================================
    uint centroidCount;
    CUDA_VERIFY(CentroidMap(vertexLabels, vertexOffsets, verticesPerTetrahedron, eqList, tetrahedronCount));
    CUDA_VERIFY(cudaMemcpy(verticesCopy, vertices, this->vertexCount * sizeof(float4), cudaMemcpyDeviceToDevice));
    CUDA_VERIFY(cudaMemcpy(vertexLabelsCopy, vertexLabels, this->vertexCount * sizeof(uint), cudaMemcpyDeviceToDevice));
    CUDA_VERIFY(CentroidReduce(&centroidCount, centroidLabels, centroidSums, centroidCounts, vertexLabelsCopy, verticesCopy, this->vertexCount));
    ValidateCentroidMemory(centroidCount);
    CUDA_VERIFY(CentroidFinalize(centroids, centroidSums, centroidCounts, centroidCount));
    //Log::DefaultLog.WriteInfo("Centroids: # %d", centroidCount);

    cudaDeviceSynchronize(); // Paranoia

    // Compute surface areas.
    CUDA_VERIFY(ComputeSurfaceArea(verticesCopy, triangleAreas, triangleCount));
    cudaDeviceSynchronize(); // Paranoia
    // Sum areas by label (note: vertexCopy and vertexLabelsCopy was sorted by CendroidReduce()).
    //TODO: map vertices to triangle areas somehow.
    //uint* vertexLabelsCopyEnd = vertexLabelsCopy + triangleCount;
    //thrust::reduce_by_key(thrust::device_ptr<uint>(vertexLabelsCopy), thrust::device_ptr<uint>(vertexLabelsCopyEnd),
    //    thrust::device_ptr<float4>(verticesCopy), thrust::device_ptr<uint>(centroidLabels), thrust::device_ptr<float>(centroidAreas));

    //cudaEvent_t start, stop;
    //float time;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord( start, 0);
    CUDA_VERIFY(ComputeCentroidArea(centroidAreas, featureStartEnd, triangleAreas, vertexLabelsCopy, centroidLabels, triangleCount, centroidCount));
    cudaDeviceSynchronize(); // Paranoia
    //cudaDeviceSynchronize();
    //cudaEventRecord( stop, 0);
    //cudaEventSynchronize( stop);
    //cudaEventElapsedTime(&time, start, stop);
    //printf ("Time for the ComputeCentroidArea kernel: %f ms\n", time);
    
    // DEBUG ...
    /*
    float *centroidAreasHost = new float[centroidCount];
    CUDA_VERIFY(cudaMemcpy(centroidAreasHost, centroidAreas, centroidCount * sizeof(float), cudaMemcpyDeviceToHost));
    float centroidAreaSum = 0;
    for( unsigned int t = 0; t < centroidCount; t++ ) {
        printf( "centroid %4i area: %10.4f\n", t, centroidAreasHost[t]);
        centroidAreaSum += centroidAreasHost[t];
    }
    float *triangleAreasHost = new float[triangleCount];
    CUDA_VERIFY(cudaMemcpy(triangleAreasHost, triangleAreas, triangleCount * sizeof(float), cudaMemcpyDeviceToHost));
    float triangleAreasSum = 0;
    for( unsigned int t = 0; t < triangleCount; t++ ) {
        triangleAreasSum += triangleAreasHost[t];
    }
    printf( "c area: %10.3f; t area: %10.3f; diff: %10.4f\n", centroidAreaSum, triangleAreasSum, centroidAreaSum-triangleAreasSum);
    delete[] centroidAreasHost;
    delete[] triangleAreasHost;
    */
    // ... DEBUG

    
    // Remove all segments with a surface area below the given threshold
    CUDA_VERIFY(cudaMemset(segmentsRemoved, false, sizeof(bool)));
    cudaDeviceSynchronize(); // Paranoia
    float areaThreshold = this->areaThresholdParam.Param<param::FloatParam>()->Value();
    CUDA_VERIFY( RemoveSmallSegments( centroidAreas, triangleAO, vertexLabels, centroidLabels, areaThreshold, this->aoThreshold, triangleCount, centroidCount, segmentsRemoved));
    cudaDeviceSynchronize(); // Paranoia
    bool segmentsRemovedHost = false;
    CUDA_VERIFY(cudaMemcpy(&segmentsRemovedHost, segmentsRemoved, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); // Paranoia
    
    // search for small segments and mark them for removal
    if (segmentsRemovedHost) {
        // ====================================================================
        // Phase 5: label mesh using tetrahedral neighbors.
        // ====================================================================
        bool hostModified;
        CUDA_VERIFY(MeshReset(eqList, refList, tetrahedronCount, vertexOffsets, triangleAO));
        cudaDeviceSynchronize(); // Paranoia
        while (true) {
            CUDA_VERIFY(cudaMemset(modified, false, sizeof(bool)));
            cudaDeviceSynchronize(); // Paranoia
            CUDA_VERIFY(MeshScan(vertices, normals, vertexOffsets, triangleAO, this->aoThreshold, eqList, refList, modified, cubeStates, cubeOffsets, cubeMap, this->isoValue, tetrahedronCount));
            CUDA_VERIFY(cudaDeviceSynchronize()); // Paranoia
            CUDA_VERIFY(cudaMemcpy(&hostModified, modified, sizeof(bool), cudaMemcpyDeviceToHost));
            if (!hostModified) {
                break;
            }
            CUDA_VERIFY(MeshAnalysis(eqList, refList, tetrahedronCount));
            CUDA_VERIFY(MeshLabeling(eqList, refList, tetrahedronCount));
        }
        cudaDeviceSynchronize(); // Paranoia
        
        // ====================================================================
        // Phase 6: map, reduce and finalize mesh segments to centroids.
        // ====================================================================
        CUDA_VERIFY(CentroidMap(vertexLabels, vertexOffsets, verticesPerTetrahedron, eqList, tetrahedronCount));
        CUDA_VERIFY(cudaMemcpy(verticesCopy, vertices, this->vertexCount * sizeof(float4), cudaMemcpyDeviceToDevice));
        CUDA_VERIFY(cudaMemcpy(vertexLabelsCopy, vertexLabels, this->vertexCount * sizeof(uint), cudaMemcpyDeviceToDevice));
        CUDA_VERIFY(CentroidReduce(&centroidCount, centroidLabels, centroidSums, centroidCounts, vertexLabelsCopy, verticesCopy, this->vertexCount));
        ValidateCentroidMemory(centroidCount);
        CUDA_VERIFY(CentroidFinalize(centroids, centroidSums, centroidCounts, centroidCount));
        cudaDeviceSynchronize(); // Paranoia

        // Compute surface areas.
        CUDA_VERIFY(ComputeSurfaceArea(verticesCopy, triangleAreas, triangleCount));
        cudaDeviceSynchronize(); // Paranoia
        CUDA_VERIFY(ComputeCentroidArea(centroidAreas, featureStartEnd, triangleAreas, vertexLabelsCopy, centroidLabels, triangleCount, centroidCount));
        cudaDeviceSynchronize(); // Paranoia
    }

    // List the beginning and end of each surface feature
    //uint *vl = new uint[this->vertexCount];
    //uint2 *se = new uint2[centroidCount];
    //CUDA_VERIFY(cudaMemcpy(vl, vertexLabelsCopy, this->vertexCount * sizeof(uint), cudaMemcpyDeviceToHost));
    //CUDA_VERIFY(cudaMemcpy(se, featureStartEnd, centroidCount * sizeof(uint2), cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();
    //for( unsigned int i = 0; i < centroidCount; i++ ) {
    //    printf( "feature %2i start = %8i; end = %8i\n", i, se[i].x, se[i].y);
    //    printf( "           start = %8i; end = %8i\n", vl[3*se[i].x], vl[3*se[i].y]);
    //} 
    //delete[] vl;
    //delete[] se;

    // ========================================================================
    // Phase 7: correlate centroids
    // ========================================================================
    vislib::StringA featureName;
    uint* centroidLabelsHost = new uint[centroidCount];
    uint* centroidCountsHost = new uint[centroidCount];
    uint* prevFeatureDataLastIdx = new uint[centroidCountLast];
    float4* centroidsHost = new float4[centroidCount];
    float4* centroidColorsHost = new float4[centroidCount];
    float* centroidAreasHost = new float[centroidCount];
    unsigned int* tmpFeatureListIdx = new unsigned int[centroidCount];
    DiagramCall::DiagramSeries *ds = 0;
    MolecularSurfaceFeature *ms = 0;
    SplitMergeCall::SplitMergeSeries *sms = 0;
    SplitMergeFeature *smf = 0;

    CUDA_VERIFY(cudaMemcpy(centroidLabelsHost, centroidLabels, centroidCount * sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_VERIFY(cudaMemcpy(centroidCountsHost, centroidCounts, centroidCount * sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_VERIFY(cudaMemcpy(centroidsHost, centroids, centroidCount * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_VERIFY(cudaMemcpy(centroidAreasHost, centroidAreas, centroidCount * sizeof(float), cudaMemcpyDeviceToHost));
    
    // search for the largest feature (i.e. the outer hull)
    uint largestFeatureIdx = 0;
    uint largestFeatureLabel = 0;
    float largestFeatureLArea = 0.0f;
    for( unsigned int i = 0; i < centroidCount; i++ ) {
        if( centroidAreasHost[i] > largestFeatureLArea ) {
            largestFeatureIdx = i;
            largestFeatureLabel = centroidLabelsHost[i];
            largestFeatureLArea = centroidAreasHost[i];
        }
    }
    
    if ( centroidsLast != 0 && centroidCount > 0 && this->cubeCountOld > 0 ) {
        // Compare tetraeder label with previous time step
        CUDA_VERIFY( WritePrevTetraLabel( tetrahedronLabelPair, cubeStatesOld, cubeOffsetsOld, //cubeMapOld, 
            verticesPerTetrahedronOld, eqListOld, cubeMap, verticesPerTetrahedron, eqList, tetrahedronCount));
        cudaDeviceSynchronize();
        int length = 0;
        CUDA_VERIFY( SortPrevTetraLabel( tetrahedronLabelPair, tetrahedronCount, length));
        cudaDeviceSynchronize();
        
        int2 *tetrahedronLabelPairHost = new int2[length];
        CUDA_VERIFY( cudaMemcpy( tetrahedronLabelPairHost, tetrahedronLabelPair,  sizeof(int2) * length, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        //for( unsigned int i = 0; i < length; i++ ) {
        //    if( tetrahedronLabelPairHost[i].x > 0 && tetrahedronLabelPairHost[i].y > 0 ) {
        //        printf( "tetrahedon %8i: [%8i - %8i]\n", i, tetrahedronLabelPairHost[i].x, tetrahedronLabelPairHost[i].y);
        //    }
        //}
        
        // match the features from current to previous time step
        vislib::Array<vislib::Array<int> > cur2prevMatching;
        cur2prevMatching.SetCount( centroidCount);
        vislib::Array<vislib::Array<int> > prev2curMatching;
        prev2curMatching.SetCount( centroidCountLast);
        for( unsigned int fcnt = 0; fcnt < centroidCount; fcnt++ ) {
            // ignore the largest feature (i.e. the outer hull)
            if( centroidLabelsHost[fcnt] == largestFeatureLabel ) {
                cur2prevMatching[fcnt].Add( largestFeatureLabelOld);
                continue;
            }
            for( unsigned int i = 0; static_cast<int>(i) < length; i++ ) {
                if( centroidLabelsHost[fcnt] == tetrahedronLabelPairHost[i].x ) {
                    // ignore the previously largest feature (i.e. the outer hull)
                    if( tetrahedronLabelPairHost[i].y == largestFeatureLabelOld || tetrahedronLabelPairHost[i].y == -1) {
                        continue;
                    }
                    cur2prevMatching[fcnt].Add( tetrahedronLabelPairHost[i].y);
                }
            }
        }
        
        // match the features from previous to current time step
        for( unsigned int fcnt = 0; fcnt < centroidCountLast; fcnt++ ) {
            // ignore the largest feature (i.e. the outer hull)
            if( centroidLabelsLast[fcnt] == largestFeatureLabelOld ) {
                prev2curMatching[fcnt].Add( largestFeatureLabel);
                continue;
            }
            for( unsigned int i = 0; static_cast<int>(i) < length; i++ ) {
                if( centroidLabelsLast[fcnt] == tetrahedronLabelPairHost[i].y ) {
                    // ignore the current largest feature (i.e. the outer hull)
                    if( tetrahedronLabelPairHost[i].x == largestFeatureLabel || tetrahedronLabelPairHost[i].x == -1) {
                        continue;
                    }
                    prev2curMatching[fcnt].Add( tetrahedronLabelPairHost[i].x);
                }
            }
            // write index of last data point in the feature series
            prevFeatureDataLastIdx[fcnt] = this->featureList[this->featureListIdx[fcnt]]->GetMappable()->GetDataCount() - 1;
        }

        vislib::Array<int> unmatchedFeatures;
        unmatchedFeatures.AssertCapacity( centroidCount);
        // match features
        for( unsigned int i = 0; i < centroidCount; i++ ) {
            if( cur2prevMatching[i].Count() == 0 ) {
                // No match --> new feature
                unmatchedFeatures.Add( i);
            } else if( cur2prevMatching[i].Count() == 1 ) {
                // One match --> continuity or split
                unsigned int j = 0;
                while( cur2prevMatching[i][0] != centroidLabelsLast[j] && j < centroidCountLast ) {
                    j++;
                }
                ASSERT( j < centroidCountLast);
                // check for split
                if( prev2curMatching[j].Count() > 1 ) {
                    // --> split
                    unsigned int k = 0;
                    unsigned int bestK;
                    float diff = std::numeric_limits<float>::max();
                    float tmpDiff;
                    // loop over all split childs to find the best fit (by area)
                    for( uint l = 0; l < prev2curMatching[j].Count(); l++ ) {
                        k = 0;
                        while( prev2curMatching[j][l] != centroidLabelsHost[k] && k < centroidCount ) {
                            k++;
                        }
                        tmpDiff = fabs(centroidAreasLast[j] - centroidAreasHost[k]);
                        if( tmpDiff < diff ) {
                            diff = tmpDiff;
                            bestK = k;
                        }
                    }
                    // if the current feature is the best fit --> continuity
                    if( bestK == i ) {
                        // Diagram
                        ms = static_cast<MolecularSurfaceFeature*>(this->featureList[this->featureListIdx[j]]->GetMappable());
                        ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                        ms->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
                        tmpFeatureListIdx[i] = this->featureListIdx[j];
                        centroidColorsHost[i] = centroidColorsLast[j];
                        //DiagramCall::DiagramSeries *ds = this->featureList[this->featureListIdx[j]];
                        ds = this->featureList[this->featureListIdx[j]];
                        //MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                        ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                        ds->AddMarker(new DiagramCall::DiagramMarker( ms->GetDataCount() - 1, DiagramCall::DIAGRAM_MARKER_SPLIT));
                        // SplitMerge
                        sms = this->splitMergeList[this->featureListIdx[j]];
                        smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
                        smf->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                        smf->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
                        // Add no transition! (feature is the best fit)
                    } else {
                        // if the current feature is NOT the best fit --> create new feature
                        // TODO should this also try to map to a previously extinct feature?
                        centroidColorsHost[i] = GetNextColor();
                        featureName.Format( "Feature %i", this->featureCounter);
                        ds = new DiagramCall::DiagramSeries( featureName, 
                            new MolecularSurfaceFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                            centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
                        ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
                        ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                        ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                        ms->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
                        this->featureList.Append( ds);
                        this->featureSelection.Append(true);
                        this->featureVisibility.Append(true);
                        tmpFeatureListIdx[i] = static_cast<unsigned int>(this->featureList.Count()) - 1;
                        this->featureCounter++;
                        vislib::Array<int> *partners = new vislib::Array<int>();
                        partners->Append(this->featureListIdx[j]);
                        ds->AddMarker(new DiagramCall::DiagramMarker( ms->GetDataCount() - 1, DiagramCall::DIAGRAM_MARKER_SPLIT, partners));
                        // SplitMerge
                        sms = new SplitMergeCall::SplitMergeSeries( featureName, 
                            new SplitMergeFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                            centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
                        sms->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
                        smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
                        smf->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                        smf->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
                        this->splitMergeList.Append( sms);
                        // Add transition
                        int test = smf->GetDataCount() - 1;    
                        if( tmpFeatureListIdx[i] == this->featureListIdx[j] ) {
                            Log::DefaultLog.WriteError( 1, "%s, SplitMerge split partner list corrupted!", this->ClassName());
                        } else {
                            this->transitionList.Add( new SplitMergeCall::SplitMergeTransition(
                                tmpFeatureListIdx[i],
                                0,
                                centroidAreasHost[i],
                                this->featureListIdx[j],
                                prevFeatureDataLastIdx[j],
                                centroidAreasLast[j]));
                        }
                    }
                } else {
                    // --> continuity
                    ms = static_cast<MolecularSurfaceFeature*>(this->featureList[this->featureListIdx[j]]->GetMappable());
                    ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                    tmpFeatureListIdx[i] = this->featureListIdx[j];
                    centroidColorsHost[i] = centroidColorsLast[j];
                    // SplitMerge
                    smf = static_cast<SplitMergeFeature*>(this->splitMergeList[this->featureListIdx[j]]->GetMappable());
                    smf->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                }
            } else if( cur2prevMatching[i].Count() > 1 ) {
                // More than one match --> merge
                unsigned int j = 0;
                int bestJ = -1;
                float diff = std::numeric_limits<float>::max();
                float tmpDiff;
                // loop over all merge partners to find the best fit (by area)
                vislib::Array<int> *partners = new vislib::Array<int>();
                for( uint k = 0; k < cur2prevMatching[i].Count(); k++ ) {    
                    j = 0;
                    // find the matching centroid label from the last time step (index j)
                    while( cur2prevMatching[i][k] != centroidLabelsLast[j] && j < centroidCountLast ) {
                        j++;
                    }
                    tmpDiff = fabs(centroidAreasHost[i] - centroidAreasLast[j]);
                    if( tmpDiff < diff ) {
                        diff = tmpDiff;
                        // add merge marker to previous best fit
                        if( bestJ >= 0 ) {
                            DiagramCall::DiagramSeries *ds = this->featureList[this->featureListIdx[bestJ]];
                            MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                            ds->AddMarker(new DiagramCall::DiagramMarker( ms->GetDataCount() - 1, DiagramCall::DIAGRAM_MARKER_MERGE));
                            partners->Append(this->featureListIdx[bestJ]);
                        }
                        // new best fit
                        bestJ = j;
                    } else {
                        // TODO add merge marker to other features
                        DiagramCall::DiagramSeries *ds = this->featureList[this->featureListIdx[j]];
                        MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                        ds->AddMarker(new DiagramCall::DiagramMarker( ms->GetDataCount() - 1, DiagramCall::DIAGRAM_MARKER_MERGE));
                        partners->Append(this->featureListIdx[j]);
                    }
                }
                ds = this->featureList[this->featureListIdx[bestJ]];
                ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                ms->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
                tmpFeatureListIdx[i] = this->featureListIdx[bestJ];
                centroidColorsHost[i] = centroidColorsLast[bestJ];
                // add merge marker to best fit
                ds->AddMarker(new DiagramCall::DiagramMarker( ms->GetDataCount() - 1, DiagramCall::DIAGRAM_MARKER_MERGE, partners));
                // SplitMerge
                sms = this->splitMergeList[this->featureListIdx[bestJ]];
                smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
                smf->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                smf->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
                // Add transition for all merge partners
                for( unsigned int mpIdx = 0; mpIdx < (*partners).Count(); mpIdx++ ) {
                    if( tmpFeatureListIdx[i] == (*partners)[mpIdx] ) {
                        Log::DefaultLog.WriteError( 1, "%s, SplitMerge merge partner list corrupted!", this->ClassName());
                        continue;
                    }
                    // Add transition
                    smf = static_cast<SplitMergeFeature*>(this->splitMergeList[(*partners)[mpIdx]]->GetMappable());
                    unsigned int lastDataIdx = smf->GetDataCount() - 1;
                    this->transitionList.Add( new SplitMergeCall::SplitMergeTransition(
                        tmpFeatureListIdx[i],
                        ms->GetDataCount() - 1,
                        centroidAreasHost[i],
                        (*partners)[mpIdx],
                        lastDataIdx,
                        smf->GetOrdinateValue(lastDataIdx)));
                }
            }
        }

        //try to match all unmatched features to extinct features
        unsigned int featureIdx;
        float xVal;

        // TEST start
        // find all extinct features
        vislib::Array<int> extinctFeatureIdx;
        extinctFeatureIdx.AssertCapacity( this->featureList.Count());
        for( unsigned int j = 0; j < this->featureList.Count(); j++ ) {
            ds = this->featureList[j];
            ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
            if( ms->GetAbscissaValue( ms->GetDataCount() - 1, 0, &xVal) ) {
                sms = this->splitMergeList[j];
                smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
                // no new value for calltime --> extinct feature
                if( mol->Calltime() != xVal ) {
                    extinctFeatureIdx.Add( j);
                    smf->AppendHole();
                    ms->AppendHole();
                }
            } else {
                extinctFeatureIdx.Add( j);
            }
        }
        float *featureDistMatrix = new float[unmatchedFeatures.Count() * extinctFeatureIdx.Count()];
        int2 *featureIdxMatrix = new int2[unmatchedFeatures.Count() * extinctFeatureIdx.Count()];
        // compute the distance between all unmatched and extinct features
        for( unsigned int i = 0; i < unmatchedFeatures.Count(); i++ ) {
            featureIdx = unmatchedFeatures[i];
            for( unsigned int j = 0; j < extinctFeatureIdx.Count(); j++ ) {
                ds = this->featureList[extinctFeatureIdx[j]];
                ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                featureDistMatrix[i * extinctFeatureIdx.Count() + j] = 
                    ( vislib::math::Vector<float, 3>( centroidsHost[featureIdx].x,  centroidsHost[featureIdx].y,  centroidsHost[featureIdx].z) - ms->GetPosition()).Length();
                featureIdxMatrix[i * extinctFeatureIdx.Count() + j] = make_int2( i, j);
            }
        }
        // sort feature distances
        thrust::sort_by_key( featureDistMatrix, featureDistMatrix + (unmatchedFeatures.Count() * extinctFeatureIdx.Count()), featureIdxMatrix);
        // match features
        int centroidIdx;
        int extinctIdx;
        for( unsigned int i = 0; i < (unmatchedFeatures.Count() * extinctFeatureIdx.Count()); i++ ) {
            // features are not yet matched and distance is within user-defined range
            centroidIdx = unmatchedFeatures[featureIdxMatrix[i].x];
            extinctIdx = extinctFeatureIdx[featureIdxMatrix[i].y];
            if( centroidIdx >= 0 && extinctIdx >= 0 && featureDistMatrix[i] < this->maxDistance ) {
                ds = this->featureList[extinctIdx];
                ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                ms->AppendValue( mol->Calltime(), centroidAreasHost[centroidIdx]);
                ms->SetPosition( centroidsHost[centroidIdx].x, centroidsHost[centroidIdx].y, centroidsHost[centroidIdx].z);
                tmpFeatureListIdx[centroidIdx] = extinctIdx;
                centroidColorsHost[centroidIdx] = make_float4( ds->GetColor().X(), ds->GetColor().Y(), ds->GetColor().Z(),  ds->GetColor().W());
                // mark unmatched feature as matched
                unmatchedFeatures[featureIdxMatrix[i].x] = -1;
                extinctFeatureIdx[featureIdxMatrix[i].y] = -1;
                // SplitMerge
                sms = this->splitMergeList[extinctIdx];
                smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
                smf->AppendValue( mol->Calltime(), centroidAreasHost[centroidIdx]);
                smf->SetPosition( centroidsHost[centroidIdx].x, centroidsHost[centroidIdx].y, centroidsHost[centroidIdx].z);
            }
        }
        delete[] featureDistMatrix;
        delete[] featureIdxMatrix;
        // TEST end

        /*
        for( unsigned int i = 0; i < unmatchedFeatures.Count(); i++ ) {
            featureIdx = unmatchedFeatures[i];
            int bestMatchIdx = -1;
            float tmpDist;
            float dist = std::numeric_limits<float>::max();
            // try to find the best fit for the new feathre from all extinct features (by distance)
            for( unsigned int j = 0; j < this->featureList.Count(); j++ ) {
                ds = this->featureList[j];
                ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                ms->GetAbscissaValue( ms->GetDataCount() - 1, 0, &xVal);
                // no new value for calltime --> extinct feature
                if( mol->Calltime() != xVal ) {
                    tmpDist = ( vislib::math::Vector<float, 3>( centroidsHost[featureIdx].x,  centroidsHost[featureIdx].y,  centroidsHost[featureIdx].z) - ms->GetPosition()).Length();
                    if( tmpDist < dist ) {
                        dist = tmpDist;
                        bestMatchIdx = j;
                    }
                }
            }
            // the best match is only valid if it is within a user-defined distance
            if( bestMatchIdx >= 0 && dist < this->maxDistance ) {
                ds = this->featureList[bestMatchIdx];
                ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                ms->AppendValue( mol->Calltime(), centroidAreasHost[featureIdx]);
                ms->SetPosition( centroidsHost[featureIdx].x, centroidsHost[featureIdx].y, centroidsHost[featureIdx].z);
                tmpFeatureListIdx[featureIdx] = bestMatchIdx;
                centroidColorsHost[featureIdx] = make_float4( ds->GetColor().X(), ds->GetColor().Y(), ds->GetColor().Z(),  ds->GetColor().W());
                // mark unmatched feature as matched
                unmatchedFeatures[i] = -1;
            }
        }
        */

        // create new diagram series for all unmatched features
        for( unsigned int j = 0; j < unmatchedFeatures.Count(); j++ ) {
            const int i = unmatchedFeatures[j];
            // do nothing for previously matched features (-1)
            if( i < 0 ) continue;
            centroidColorsHost[i] = GetNextColor();
            featureName.Format( "Feature %i", this->featureCounter);
            ds = new DiagramCall::DiagramSeries( featureName, 
                new MolecularSurfaceFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
            ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
            ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
            ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
            ms->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
            this->featureList.Append( ds);
            this->featureSelection.Append(true);
            this->featureVisibility.Append(true);
            tmpFeatureListIdx[i] = static_cast<unsigned int>(this->featureList.Count()) - 1;
            this->featureCounter++;
            // SplitMerge
            sms = new SplitMergeCall::SplitMergeSeries( featureName, 
                new SplitMergeFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
            sms->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
            smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
            smf->AppendValue( mol->Calltime(), centroidAreasHost[i]);
            smf->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
            this->splitMergeList.Append( sms);
        }

        // TODO
        // No match from previous to current --> extinction
        
        
        delete[] tetrahedronLabelPairHost;
        // delete old values
        delete[] centroidCountsHost;
        delete[] centroidsLast;
        delete[] centroidColorsLast;
        delete[] centroidLabelsLast;
        delete[] centroidAreasLast;
        delete[] featureListIdx;
    } else {
        // first frame --> color everything!
        for (uint i = 0; i < centroidCount; ++i) {
            centroidColorsHost[i] = GetNextColor();
            // add all new features to the diagram
            featureName.Format( "Feature %i", this->featureCounter);
            ds = new DiagramCall::DiagramSeries( featureName, 
                new MolecularSurfaceFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
            ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
            ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
            ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
            ms->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
            this->featureList.Append( ds);
            this->featureSelection.Append(true);
            this->featureVisibility.Append(true);
            tmpFeatureListIdx[i] = static_cast<unsigned int>(this->featureList.Count()) - 1;
            this->featureCounter++;
            // SplitMerge
            sms = new SplitMergeCall::SplitMergeSeries( featureName, 
                new SplitMergeFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
            sms->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
            smf = static_cast<SplitMergeFeature*>(sms->GetMappable());
            smf->AppendValue( mol->Calltime(), centroidAreasHost[i]);
            smf->SetPosition( centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z);
            this->splitMergeList.Append( sms);
        }
    }

    /*
    // preliminary solution (kroneml)
    if ( centroidsLast != 0 && centroidCount > 0 ) {
        // list of potential matches for each feature
        vislib::Array<vislib::Array<uint> > potentialMatch;
        potentialMatch.SetCount( centroidCount);
        vislib::Array<vislib::Array<float> > potentialMatchDistance;
        potentialMatchDistance.SetCount( centroidCount);
        
        // loop over all centroids
        for ( uint i = 0; i < centroidCount; i++) {
            potentialMatchDistance[i].Clear();
            potentialMatchDistance[i].AssertCapacity(centroidCountLast);
            potentialMatch[i].Clear();
            potentialMatch[i].AssertCapacity( centroidCountLast);
            // compute distance to all neighbors
            for ( uint j = 0; j < centroidCountLast; j++) {
                float dist = length(centroidsHost[i] - centroidsLast[j]);
                // mark centroids within distance threshold as potential matches
                if ( dist <= this->maxDistance) {
                    potentialMatchDistance[i].Add( dist);
                    potentialMatch[i].Add( j);
                }
            }
            
            // sort match list by distance (ascending order)
            if ( potentialMatchDistance[i].Count() > 0 ) {
                thrust::sort_by_key( &potentialMatchDistance[i][0], 
                    &potentialMatchDistance[i][0] + potentialMatchDistance[i].Count(), 
                    &potentialMatch[i][0]);
            }
        }
        // Preliminary results: 
        //      No match            --> extinction
        //      One match           --> continuity
        //      More than one match --> merge

        // TODO: set extinction markers
        // TODO: check merge probability (compare areas)

        // generate split list
        // TODO: this is kind of ugly (4 nested for-loops), fix this!
        bool split = false;
        vislib::Array< vislib::Array<uint> > splitList;
        splitList.SetCount( centroidCountLast);
        // loop over all centroids
        for( unsigned int i = 0; i < centroidCount - 1; i++) {
            // loop over all remaining centroids
            for( unsigned int j = i + 1; j < centroidCount; j++) {
                // loop over all matches for the current centroid
                for (unsigned int k = 0; k < potentialMatch[i].Count(); k++ ) {
                    // loop over all matches for the second centroid
                    for ( unsigned int l = 0; l < potentialMatch[j].Count(); l++) {
                        // check for splits
                        if( potentialMatch[i][k] == potentialMatch[j][l]) {
                            splitList[potentialMatch[i][k]].Add( i);
                            splitList[potentialMatch[i][k]].Add( j);
                            split = true;
                        }
                    }
                }
            }
        }

        // DEBUG
        //for( unsigned int i = 0; i < splitList.Count(); i++) {
        //    if( splitList[i].Count() > 0 ) {
        //        printf( "Old feature %i [%6.2f] split into: ", i, centroidAreasLast[i]);
        //        for( unsigned int j = 0; j < splitList[i].Count(); j++) {
        //            printf( "%i [%6.2f] (", splitList[i][j], centroidAreasHost[splitList[i][j]]);
        //            for( unsigned int k = 0; k < potentialMatch[splitList[i][j]].Count(); k++) {
        //                printf( "%i [%6.2f %6.2f]", potentialMatch[splitList[i][j]][k], centroidAreasLast[potentialMatch[splitList[i][j]][k]], potentialMatchDistance[splitList[i][j]][k]);
        //                if( k < potentialMatch[splitList[i][j]].Count() - 1 ) {
        //                    printf( ", ");
        //                }
        //            }
        //            printf("), ");
        //        }
        //        printf("\n");
        //    }
        //}

        // match features
        for( unsigned int i = 0; i < centroidCount; i++ ) {
            if( potentialMatch[i].Count() == 1 && splitList[potentialMatch[i][0]].Count() == 0 ) {
                // only one potential match and no potential splits --> continuity
                unsigned int j = potentialMatch[i][0];
                MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(this->featureList[this->featureListIdx[j]]->GetMappable());
                ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                tmpFeatureListIdx[i] = this->featureListIdx[j];
                centroidColorsHost[i] = centroidColorsLast[j];
            } else if( potentialMatch[i].Count() > 1 && splitList[potentialMatch[i][0]].Count() == 0 ) {
                // more than one potential match and no potential splits --> merge
                unsigned int j = potentialMatch[i][0];
                MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(this->featureList[this->featureListIdx[j]]->GetMappable());
                ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                tmpFeatureListIdx[i] = this->featureListIdx[j];
                centroidColorsHost[i] = centroidColorsLast[j];
            } else {
                // no (clear) match --> add new feature to the diagram
                centroidColorsHost[i] = GetNextColor();
                featureName.Format( "Feature %i", this->featureCounter);
                DiagramCall::DiagramSeries *ds = new DiagramCall::DiagramSeries( featureName, 
                    new MolecularSurfaceFeature( static_cast<float>(mol->FrameCount()), vislib::math::Vector<float, 3>( 
                    centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
                ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
                MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
                ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                this->featureList.Append( ds);
                tmpFeatureListIdx[i] = this->featureList.Count() - 1;
                this->featureCounter++;
            }
        }

        // delete old values
        delete[] centroidCountsHost;
        delete[] centroidsLast;
        delete[] centroidColorsLast;
        delete[] centroidLabelsLast;
        delete[] centroidAreasLast;
        delete[] featureListIdx;
    } else {
        // first frame --> color everything!
        for (uint i = 0; i < centroidCount; ++i) {
            centroidColorsHost[i] = GetNextColor();
            // add all new features to the diagram
            featureName.Format( "Feature %i", this->featureCounter);
            DiagramCall::DiagramSeries *ds = new DiagramCall::DiagramSeries( featureName, 
                new MolecularSurfaceFeature( mol->FrameCount(), vislib::math::Vector<float, 3>( 
                centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
            ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
            MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
            ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
            this->featureList.Append( ds);
            tmpFeatureListIdx[i] = this->featureList.Count() - 1;
            this->featureCounter++;
        }
    }
    */

    /*
    // Christoph Schulz' solution
    if (centroidsLast != 0) {
        std::multimap<float, uint> distances;
        for (uint i = 0; i < centroidCount; ++i) {
            distances.clear();
            // Build a distance map from non-hull centroids.
            for (uint j = 0; j < centroidCountLast; ++j) {
                float distance = length(centroidsHost[i] - centroidsLast[j]);
                if (distance <= this->maxDistance) {
                    distances.insert(std::pair<float, uint>(distance, j));
                }
            }
            // Scan for tracking candidates.
            float lastDistance = 0;
            std::multimap<float, uint>::iterator it;
            for(it = distances.begin(); it != distances.end(); ++it) {
                float deltaDistance = abs(lastDistance - it->first);
                if (lastDistance != 0 && deltaDistance > this->maxDeltaDistance) {
                    // Truncate rest of the distance map.
                    std::multimap<float, uint>::iterator eraseIt = it;
                    while (eraseIt != distances.end()) {
                        distances.erase(eraseIt++);
                    }
                    break;
                }
                lastDistance = it->first;
            }
            if (distances.size() == 0) {
                // N == 0: New centroid.
                centroidColorsHost[i] = GetNextColor();
            } else if (distances.size() == 1) {
                // N == 1: Split canidate.
                it = distances.begin();
                bool splittedCentroid = false;
                for (uint k = 0; k < i; ++k) {
                    if (centroidColorsHost[k].x == centroidColorsLast[it->second].x &&
                        centroidColorsHost[k].y == centroidColorsLast[it->second].y &&
                        centroidColorsHost[k].z == centroidColorsLast[it->second].z) {
                        //Log::DefaultLog.WriteInfo("Split %d:%d and %d (distance: %f)", i, k, it->second, it->first);
                        splittedCentroid = true;
                        // search for the feature entry in the diagram and add the value
                        vislib::math::Vector<float, 3> col0( centroidColorsHost[k].x, centroidColorsHost[k].y, centroidColorsHost[k].z);
                        vislib::math::Vector<float, 3> col1;
                        for( unsigned int fIdx = 0; fIdx < this->featureList.Count(); fIdx++ ) {
                            // check if colors are the same
                            // TODO this is bad - use explicit index for this!
                            col1 = this->featureList[fIdx]->GetColorRGB();
                            if( (col0 - col1).Length() < vislib::math::FLOAT_EPSILON ) {
                                MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(this->featureList[fIdx]->GetMappable());
                                ms->AppendValue( mol->Calltime(), centroidAreasHost[k]);
                                break;
                            }
                        }
                        break;
                    }
                }
                if (splittedCentroid) {
                    // Pick a new color for a splitted centroid.
                    centroidColorsHost[i] = GetNextColor();
                    // add all new entries to the diagram
                    featureName.Format( "Feature %i", this->featureCounter);
                    this->featureCounter++;
                    DiagramCall::DiagramSeries *ds = new DiagramCall::DiagramSeries( featureName, 
                        new MolecularSurfaceFeature( mol->FrameCount(), vislib::math::Vector<float, 3>( 
                        centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
                    ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
                    this->featureList.Append( ds);
                } else {
                    // Reuse old color.
                    centroidColorsHost[i] = centroidColorsLast[it->second];
                    // search for the feature entry in the diagram and add the value
                    vislib::math::Vector<float, 3> col0( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z);
                    vislib::math::Vector<float, 3> col1;
                    for( unsigned int fIdx = 0; fIdx < this->featureList.Count(); fIdx++ ) {
                        // check if colors are the same
                        // TODO this is bad - use explicit index for this!
                        col1 = this->featureList[fIdx]->GetColorRGB();
                        if( (col0 - col1).Length() < vislib::math::FLOAT_EPSILON ) {
                            MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(this->featureList[fIdx]->GetMappable());
                            ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
                            break;
                        }
                    }
                }
            } else {
                // N > 1: Merge.
                for(it = distances.begin(); it != distances.end(); ++it) {
                    //Log::DefaultLog.WriteInfo("Merge %d:%d (distance: %f)", i, it->second, it->first);
                }
            }
        }
        delete[] centroidCountsHost;
        delete[] centroidLabelsLast;
        delete[] centroidsLast;
        delete[] centroidColorsLast;
    } else {
        // first frame -> color everything!
        for (uint i = 0; i < centroidCount; ++i) {
            centroidColorsHost[i] = GetNextColor();
            // add all new entries to the diagram
            featureName.Format( "Feature %i", this->featureCounter);
            this->featureCounter++;
            DiagramCall::DiagramSeries *ds = new DiagramCall::DiagramSeries( featureName, 
                new MolecularSurfaceFeature( mol->FrameCount(), vislib::math::Vector<float, 3>( 
                centroidsHost[i].x, centroidsHost[i].y, centroidsHost[i].z)));
            ds->SetColor( centroidColorsHost[i].x, centroidColorsHost[i].y, centroidColorsHost[i].z, centroidColorsHost[i].w);
            MolecularSurfaceFeature *ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
            ms->AppendValue( mol->Calltime(), centroidAreasHost[i]);
            this->featureList.Append( ds);
        }
    }
    */

    centroidCountLast = centroidCount;
    centroidsLast = centroidsHost;
    centroidColorsLast = centroidColorsHost;
    centroidLabelsLast = centroidLabelsHost;
    centroidAreasLast = centroidAreasHost;
    featureListIdx = tmpFeatureListIdx;
    largestFeatureLabelOld = largestFeatureLabel;

    cudaDeviceSynchronize(); // Paranoia

    // TODO: für alle feature jede 0.5 T (MegaMol-time) wertepaar anhängen
    // TODO: hash implementieren (fuer diagramcall)

    // Get feature selection for coloring
    IntSelectionCall *selectionCall = this->selectionCallerSlot.CallAs<IntSelectionCall>();
    if (selectionCall != NULL) {
        (*selectionCall)(IntSelectionCall::CallForGetSelection);
        if (selectionCall->GetSelectionPointer() != NULL) {
            if (selectionCall->GetSelectionPointer()->Count() > 0) {
                for (SIZE_T x = 0; x < this->featureSelection.Count(); x++) {
                    this->featureSelection[x] = false;
                }
                for (SIZE_T x = 0; x < selectionCall->GetSelectionPointer()->Count(); x++) {
                    this->featureSelection[(*selectionCall->GetSelectionPointer())[x]] = true;
                }
            } else {
                for (SIZE_T x = 0; x < this->featureSelection.Count(); x++) {
                    this->featureSelection[x] = true;
                }
            }
        }
    }
    
    // Get feature selection for visibility of features
    IntSelectionCall *visibilityCall = this->hiddenCallerSlot.CallAs<IntSelectionCall>();
    if (visibilityCall != NULL) {
        (*visibilityCall)(IntSelectionCall::CallForGetSelection);
        if (visibilityCall->GetSelectionPointer() != NULL) {
            if (visibilityCall->GetSelectionPointer()->Count() > 0) {
                for (SIZE_T x = 0; x < this->featureSelection.Count(); x++) {
                    this->featureVisibility[x] = true;
                }
                for (SIZE_T x = 0; x < visibilityCall->GetSelectionPointer()->Count(); x++) {
                    this->featureVisibility[(*visibilityCall->GetSelectionPointer())[x]] = false;
                }
            } else {
                for (SIZE_T x = 0; x < this->featureSelection.Count(); x++) {
                    this->featureVisibility[x] = true;
                }
            }
        }
    }

    // Set features visible/invisible (alpha value of centroid color)
    for( unsigned int i = 0; i < centroidCount; i++ ) {
        float w;
        if (this->featureVisibility[this->featureListIdx[i]]) {
            w = 1.0f;
        } else {
            w = 0.0f;
        }
        if (!this->featureSelection[this->featureListIdx[i]]) {
            centroidColorsHost[i] = make_float4( 0.5f, 0.5f, 0.5f, w);
        } else {
            centroidColorsHost[i] = make_float4( 
                this->featureList[this->featureListIdx[i]]->GetColor().X(),
                this->featureList[this->featureListIdx[i]]->GetColor().Y(), 
                this->featureList[this->featureListIdx[i]]->GetColor().Z(), 
                w);
        }
    }
    
    
    // ========================================================================
    // Phase 8: Compute center lines for all features
    // ========================================================================
#if 1
    // List the beginning and end of each surface feature
    //uint *vl = new uint[this->vertexCount];
    //uint2 *se = new uint2[centroidCount];
    //CUDA_VERIFY(cudaMemcpy(vl, vertexLabelsCopy, this->vertexCount * sizeof(uint), cudaMemcpyDeviceToHost));
    //CUDA_VERIFY(cudaMemcpy(se, featureStartEnd, centroidCount * sizeof(uint2), cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();
    //for( unsigned int i = 0; i < centroidCount; i++ ) {
    //    printf( "feature %2i start = %8i; end = %8i\n", i, se[i].x, se[i].y);
    //    printf( "           start = %8i; end = %8i\n", vl[3*se[i].x], vl[3*se[i].y]);
    //} 
    //delete[] vl;
    //delete[] se;

    if( this->clg.Capacity() < centroidCount ) {
        this->clg.AssertCapacity( centroidCount * 2);
        this->clEdges.AssertCapacity( centroidCount * 2);
        this->clNodes.AssertCapacity( centroidCount * 2);
    }
    for( unsigned int i = 0; i < this->clg.Count(); i++ ) {
        if( clg[i] )
            delete clg[i];
        clg[i] = 0;
    }
    this->clg.SetCount( centroidCount-1);
    this->clEdges.SetCount( centroidCount-1);
    this->clNodes.SetCount( centroidCount-1);

    // get the feature start and end indices to compute center
    CUDA_VERIFY(cudaMemcpy( this->featureStartEndHost, this->featureStartEnd, centroidCount * sizeof(uint2), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    this->featureTrianglesCount = 0;
        
    // loop over all features (skip first feature)
    for( unsigned int fCnt = 1; fCnt < centroidCount; fCnt++ ) {
        cudaEventRecord( start, 0); // TIMING
        unsigned int fStart = this->featureStartEndHost[fCnt].x;
        unsigned int fEnd = this->featureStartEndHost[fCnt].y;
        unsigned int fLength = fEnd - fStart + 1;
        CUDA_VERIFY(cudaMemcpy( this->featureVertices, &(verticesCopy[3*fStart]), fLength * 3 * sizeof(float4), cudaMemcpyDeviceToDevice));
        this->featureTrianglesCount = fLength;
        cudaDeviceSynchronize();
        // reset feature vertex count
        this->featureVertexCntNew = 0;
        CUDA_VERIFY( TriangleVerticesToIndexList( this->featureVertices, this->featureVerticesOut, this->featureVertexIdx, this->featureVertexCnt, 
            this->featureVertexCntOut, this->featureVertexStartIdx, this->featureVertexIdxOut, fLength * 3, this->featureVertexCntNew));
        cudaDeviceSynchronize();
        
        // reset feature edge count
        this->edgeCount = 0;
        TriangleEdgeList( this->featureVertexIdxOut, this->featureEdgeCnt, this->featureEdgeCntOut, fLength, this->featureEdges, this->featureEdgesOut, this->edgeCount);
        
        // download compacted feature vertices
        CUDA_VERIFY(cudaMemcpy( this->featureTriangleVerticesHost, this->featureVerticesOut, this->featureVertexCntNew * sizeof(float4), cudaMemcpyDeviceToHost));
        // download feature vertex indices
        //CUDA_VERIFY(cudaMemcpy( this->featureTriangleVertexIndicesHost, this->featureVertexIdxOut,  fLength * 3 * sizeof(uint), cudaMemcpyDeviceToHost));
        // download triangle edges
        CUDA_VERIFY(cudaMemcpy( this->featureTriangleEdgesHost, this->featureEdgesOut, this->edgeCount * sizeof(uint2), cudaMemcpyDeviceToHost));
        // download triangle edge count
        CUDA_VERIFY(cudaMemcpy( this->featureTriangleEdgeCountHost, this->featureEdgeCntOut, this->edgeCount * sizeof(uint), cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize();        // TIMING
        cudaEventRecord( stop, 0);      // TIMING
        cudaEventSynchronize( stop);    // TIMING
        cudaEventElapsedTime(&time, start, stop);   // TIMING
        //printf( "CUDA Prepare center line data for feature %3i (%5i tria):    %.5f\n", fCnt, fLength, time / 1000.0f); // TIMING
        
        // TODO compute the center line of this feature
        perf.SetMark();
        clg[fCnt-1] = new CenterLineGenerator();
        clg[fCnt-1]->SetTriangleMesh( this->featureVertexCntNew, (float*)this->featureTriangleVerticesHost, 
            this->edgeCount, (unsigned int*)this->featureTriangleEdgesHost, (unsigned int*)this->featureTriangleEdgeCountHost);
        //printf( "Time to prepare center line data for feature %3i (%5i tria): %.5f\n", fCnt, fLength, ( double( clock() - t) / double( CLOCKS_PER_SEC) ));
        if( !clg[fCnt-1]->freeEdgeRing.empty() && !clg[fCnt-1]->freeEdgeRing[0].empty() ) {
            for ( auto edge : clEdges[fCnt-1] )  {
                if( edge ) delete edge;
            }
            for ( auto node : clNodes[fCnt-1] )  {
                if( node ) delete node;
            }
            clEdges[fCnt-1].clear();
            clNodes[fCnt-1].clear();
            clg[fCnt-1]->CenterLine( clg[fCnt-1]->freeEdgeRing[0], clEdges[fCnt-1], clNodes[fCnt-1], this->minDistCenterLineParam.Param<param::FloatParam>()->Value());
            //printf( "Time to compute center line for feature %3i (%5i tria):      %.5f\n\n", fCnt, fLength, ( double( clock() - t) / double( CLOCKS_PER_SEC) ));
        }
        INT64 t1 = perf.Difference();
        Log::DefaultLog.WriteInfo( 1, "Time to compute centerline for feature %3i (%5i tria): %.5f", fCnt, fLength, (vislib::sys::PerformanceCounter::ToMillis(t1) + time) / 1000.0);
    }
#endif
    // ========================================================================
    // Phase 9: Colorize vertices
    // ========================================================================
    CUDA_VERIFY(cudaMemcpy(centroidColors, centroidColorsHost, centroidCount * sizeof(float4), cudaMemcpyHostToDevice));
    ColorizeByCentroid(colors, centroidColors, centroidLabels, centroidCount, vertexLabels, this->vertexCount);
    //ColorizeByAO(colors, triangleAO, aoThreshold, this->vertexCount);
    
    cudaDeviceSynchronize(); // Paranoia
    
    // coloring by nearest atom TEST ...
    cudaMemcpy( this->vertexColors, colors, this->vertexCount * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    float ifac = this->cmWeightParam.Param<param::FloatParam>()->Value();
#pragma omp parallel for
	for (int i = 0; i < (int)this->vertexCount; i++) {
        int atomIdx = this->neighborAtomOfVertex[i];
        if( atomIdx < 0 ) {
            // ERROR no nearest atom found (color magenta)
            this->vertexColors[4*i+0] = 1.0f;
            this->vertexColors[4*i+1] = 0.0f;
            this->vertexColors[4*i+2] = 1.0f;
        } else if( atomIdx >= (this->atomColorTable.Count() / 3)) {
            // ERROR nearest atom has too large index (color cyan)
            this->vertexColors[4*i+0] = 0.0f;
            this->vertexColors[4*i+1] = 1.0f;
            this->vertexColors[4*i+2] = 1.0f;
        } else if ( !this->resSelectionCall || this->atomSelection[atomIdx] ) {
            // color triangles
            if( this->currentColoringMode0 < 0 ) {
                if( this->currentColoringMode1 >= 0 ) {
                    // mix between surface feature color and color mode 1
                    this->vertexColors[4*i+0] = ( this->vertexColors[4*i+0] * ifac + this->atomColorTable[3*atomIdx+0] * (1.0f - ifac));
                    this->vertexColors[4*i+1] = ( this->vertexColors[4*i+1] * ifac + this->atomColorTable[3*atomIdx+1] * (1.0f - ifac));
                    this->vertexColors[4*i+2] = ( this->vertexColors[4*i+2] * ifac + this->atomColorTable[3*atomIdx+2] * (1.0f - ifac));
                }
                // else - color by surface feature (do nothing)
            } else {
                if( this->currentColoringMode1 < 0 ) {
                    // mix between color mode 0 and  surface feature color
                    this->vertexColors[4*i+0] = ( this->atomColorTable[3*atomIdx+0] * ifac + this->vertexColors[4*i+0] * (1.0f - ifac));
                    this->vertexColors[4*i+1] = ( this->atomColorTable[3*atomIdx+1] * ifac + this->vertexColors[4*i+1] * (1.0f - ifac));
                    this->vertexColors[4*i+2] = ( this->atomColorTable[3*atomIdx+2] * ifac + this->vertexColors[4*i+2] * (1.0f - ifac));
                } else {
                    // use only atom colors
                    this->vertexColors[4*i+0] = this->atomColorTable[3*atomIdx+0];
                    this->vertexColors[4*i+1] = this->atomColorTable[3*atomIdx+1];
                    this->vertexColors[4*i+2] = this->atomColorTable[3*atomIdx+2];
                }
            }
        } else {
            // atom not selected: color grey
            this->vertexColors[4*i+0] = 0.5f;
            this->vertexColors[4*i+1] = 0.5f;
            this->vertexColors[4*i+2] = 0.5f;
        }
        if( this->blendIt ) {
            if ( this->resSelectionCall && this->atomSelection[atomIdx] ) {
                // is this correct? if alpha was set to zero (invisible) keep it that way.
                this->vertexColors[4*i+3] = vislib::math::Min( 1.0f, this->vertexColors[4*i+3]);
            } else {
                this->vertexColors[4*i+3] = vislib::math::Min( 
                    this->alphaParam.Param<param::FloatParam>()->Value(),
                    this->vertexColors[4*i+3]);
            }
        }
    }
    cudaMemcpy( colors, this->vertexColors, this->vertexCount * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // Paranoia
    // ... coloring by nearest atom TEST

    // Unmap VBOs.
    CUDA_VERIFY(cudaGraphicsUnmapResources(1, &positionResource, 0));
    CUDA_VERIFY(cudaGraphicsUnmapResources(1, &normalResource, 0));
    CUDA_VERIFY(cudaGraphicsUnmapResources(1, &colorResource, 0));

    // Unbind, unmap and unregister volume texture.
    CUDA_VERIFY(UnbindAOVolumeTexture());
    CUDA_VERIFY(UnbindVolumeTexture());
    
    return true;
}

float4 VolumeMeshRenderer::GetNextColor() {
    using vislib::sys::Log;
    bool repick = true;
    int nextColorIndex = centroidColorsIndex;
    while (repick) {
        nextColorIndex = (nextColorIndex + 1) % this->colorTable.Count();
        if (nextColorIndex == centroidColorsIndex) {
            Log::DefaultLog.WriteError("Out of colors");
            break;
        }
        float4 nextColor = make_float4(this->colorTable[nextColorIndex][0], 
            this->colorTable[nextColorIndex][1], this->colorTable[nextColorIndex][2], 1.0f);
        repick = false;
        if (centroidsLast != 0) {
            for (uint j = 0; j < centroidCountLast; ++j) {
                if (nextColor.x == centroidColorsLast[j].x &&
                    nextColor.y == centroidColorsLast[j].y &&
                    nextColor.z == centroidColorsLast[j].z) {
                    repick = true;
                    break;
                }
            }
        }
    }
    centroidColorsIndex = nextColorIndex;
    return make_float4(this->colorTable[centroidColorsIndex][0], 
        this->colorTable[centroidColorsIndex][1], this->colorTable[centroidColorsIndex][2], 1.0f);
}

/*
 * refresh parameters
 */
void VolumeMeshRenderer::ParameterRefresh( const MolecularDataCall *mol, const BindingSiteCall *bs) {
    if (this->polygonModeParam.IsDirty()) {
        this->polygonMode = static_cast<PolygonMode>(this->polygonModeParam.Param<param::EnumParam>()->Value());
        this->polygonModeParam.ResetDirty();
    }
    if (this->blendItParam.IsDirty()) {
        this->blendIt = this->blendItParam.Param<param::BoolParam>()->Value();
        this->blendItParam.ResetDirty();
    }
    if (this->showNormalsParam.IsDirty()) {
        this->showNormals = this->showNormalsParam.Param<param::BoolParam>()->Value();
        this->showNormalsParam.ResetDirty();
    }
    if (this->showCentroidsParam.IsDirty()) {
        this->showCentroids = this->showCentroidsParam.Param<param::BoolParam>()->Value();
        this->showCentroidsParam.ResetDirty();
    }
    if (this->aoThresholdParam.IsDirty()) {
        this->aoThreshold = this->aoThresholdParam.Param<param::FloatParam>()->Value();
        this->aoThresholdParam.ResetDirty();
    }
    if (this->isoValueParam.IsDirty()) {
        this->isoValue = this->isoValueParam.Param<param::FloatParam>()->Value();
        this->isoValueParam.ResetDirty();
    }
    if (this->maxDistanceParam.IsDirty()) {
        this->maxDistance = this->maxDistanceParam.Param<param::FloatParam>()->Value();
        this->maxDistanceParam.ResetDirty();
    }
    if (this->maxDeltaDistanceParam.IsDirty()) {
        this->maxDeltaDistance = this->maxDeltaDistanceParam.Param<param::FloatParam>()->Value();
        this->maxDeltaDistanceParam.ResetDirty();
    }

    // color table param
    if( this->colorTableFileParam.IsDirty() ) {
        Color::ReadColorTableFromFile(
            this->colorTableFileParam.Param<param::StringParam>()->Value(),
            this->colorTable);
        this->colorTableFileParam.ResetDirty();
    }
    // Recompute color table
    if( this->coloringModeParam0.IsDirty() || this->coloringModeParam1.IsDirty() || this->cmWeightParam.IsDirty()) {
        
        this->currentColoringMode0 = static_cast<Color::ColoringMode>(int(
            this->coloringModeParam0.Param<param::EnumParam>()->Value()));
        this->currentColoringMode1 = static_cast<Color::ColoringMode>(int(
            this->coloringModeParam1.Param<param::EnumParam>()->Value()));
        
        if( this->currentColoringMode0 < 0 ) {
            if( this->currentColoringMode1 < 0 ) {
                // Color by surface feature -> set all colors to white
                this->atomColorTable.SetCount( mol->AtomCount() * 3);
                for( unsigned int i = 0; i < mol->AtomCount() * 3; i++ ) {
                    this->atomColorTable[i] = 1.0f;
                }
            } else {
                // only color by color mode 1
                Color::MakeColorTable( mol,
                    static_cast<Color::ColoringMode>(this->currentColoringMode1),
                    this->atomColorTable, this->colorTable, this->rainbowColors,
                    this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(),
                    true, bs);
            }
        } else {
            if( this->currentColoringMode1 < 0 ) {
                // only color by color mode 0
                Color::MakeColorTable( mol,
                    static_cast<Color::ColoringMode>(this->currentColoringMode0),
                    this->atomColorTable, this->colorTable, this->rainbowColors,
                    this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(),
                    true, bs);
            } else {
            // Mix two coloring modes
            Color::MakeColorTable( mol,
                static_cast<Color::ColoringMode>(this->currentColoringMode0),
                static_cast<Color::ColoringMode>(this->currentColoringMode1),
                cmWeightParam.Param<param::FloatParam>()->Value(),       // weight for the first cm
                1.0f - cmWeightParam.Param<param::FloatParam>()->Value(), // weight for the second cm
                this->atomColorTable, this->colorTable, this->rainbowColors,
                this->minGradColorParam.Param<param::StringParam>()->Value(),
                this->midGradColorParam.Param<param::StringParam>()->Value(),
                this->maxGradColorParam.Param<param::StringParam>()->Value(),
                true, bs);
            }
        }
        
        this->coloringModeParam0.ResetDirty();
        this->coloringModeParam1.ResetDirty();
        this->cmWeightParam.ResetDirty();
    }
}

/*
 * VolumeMeshRenderer::ValidateCubeMemory
 */
void VolumeMeshRenderer::ValidateOldCubeMemory() {
    // sufficient memory already allocated --> do nothing
    if (this->cubeCount <= this->cubeCountOld && this->cubeCountAllocated != 0 && 
        this->activeCubeCount <= this->activeCubeCountOld ) {
        // store old cube count and old activeCubeCount
        this->cubeCountOld = this->cubeCount;
        this->activeCubeCountOld = this->activeCubeCount;
        return;
    }
    if( this->cubeCountAllocated > 0 ) {
        // cubeCountAllocated > 0 --> not the first frame
        if( this->cubeCountOld > 0 ) {
            CUDA_VERIFY(cudaFree(cubeMapOld));
            CUDA_VERIFY(cudaFree(cubeOffsetsOld));
            CUDA_VERIFY(cudaFree(cubeStatesOld));
        }
        if( this->activeCubeCountOld > 0 ) {
            CUDA_VERIFY(cudaFree(verticesPerTetrahedronOld));
            CUDA_VERIFY(cudaFree(eqListOld));
        }
        // Allocate CUDA memory for Marching Tetrahedrons.
        CUDA_VERIFY(cudaMalloc(&cubeStatesOld, sizeof(uint) * this->cubeCount));
        CUDA_VERIFY(cudaMalloc(&cubeOffsetsOld, sizeof(uint) * this->cubeCount));
        CUDA_VERIFY(cudaMalloc(&cubeMapOld, sizeof(uint) * this->cubeCount));
        CUDA_VERIFY(cudaMalloc(&verticesPerTetrahedronOld, sizeof(uint) * this->activeCubeCount * 6));
        CUDA_VERIFY(cudaMalloc(&eqListOld, sizeof(uint) * this->activeCubeCount * 6));
        // store old cube count and old activeCubeCount
        this->cubeCountOld = this->cubeCount;
        this->activeCubeCountOld = this->activeCubeCount;
    } else {
        // cubeCountAllocated <= 0 --> first frame (do not copy)
        this->cubeCountOld = 0;
        this->activeCubeCountOld = 0;
    }
}

/*
 * VolumeMeshRenderer::ValidateCubeMemory
 */
void VolumeMeshRenderer::ValidateCubeMemory() {
    this->cubeCount = this->volumeSize.x * this->volumeSize.y * this->volumeSize.z;
    if ( this->cubeCount <= this->cubeCountAllocated && cubeCountAllocated != 0) {
        return;
    }
    if (this->cubeCountAllocated > 0) {
        // Free CUDA memory for AO.
        CUDA_VERIFY(cudaFreeArray(aoVolume));

        // Free CUDA memory for labling.
        //CUDA_VERIFY(cudaFree(modified));

        // Free CUDA memory for Marching Tetrahedrons.
        CUDA_VERIFY(cudaFree(cubeMap));
        CUDA_VERIFY(cudaFree(cubeOffsets));
        CUDA_VERIFY(cudaFree(cubeStates));
    }
    if (cubeCount > 0) {
        // Allocate CUDA memory for Marching Tetrahedrons.
        this->cubeCountAllocated = this->cubeCount;
        CUDA_VERIFY(cudaMalloc(&cubeStates, sizeof(uint) * this->cubeCountAllocated));
        CUDA_VERIFY(cudaMalloc(&cubeOffsets, sizeof(uint) * this->cubeCountAllocated));
        CUDA_VERIFY(cudaMalloc(&cubeMap, sizeof(uint) * this->cubeCountAllocated));

        // Allocate CUDA memory for labling.
        //CUDA_VERIFY(cudaMalloc(&modified, sizeof(bool)));

        // Allocate CUDA memory for AO.
        cudaExtent aoVolumeExtent = make_cudaExtent(this->aoShader.getVolumeSizeX(), 
            this->aoShader.getVolumeSizeY(), 
            this->aoShader.getVolumeSizeZ());
        cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
        CUDA_VERIFY(cudaMalloc3DArray(&this->aoVolume, &cd, aoVolumeExtent));
        // Emulate cudaMemset3D for cudaArrays. 
        float* zeroArray = (float*) calloc(aoVolumeExtent.width * aoVolumeExtent.height * aoVolumeExtent.depth, sizeof(float));
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr(
            (void*)zeroArray, aoVolumeExtent.width * sizeof(float),
            aoVolumeExtent.width, aoVolumeExtent.height);
        copyParams.dstArray = this->aoVolume;
        copyParams.extent   = aoVolumeExtent;
        copyParams.kind     = cudaMemcpyHostToDevice;
        CUDA_VERIFY(cudaMemcpy3D(&copyParams));
        free(zeroArray);
    }
}

/*
 * VolumeMeshRenderer::ValidateVertexMemory
 */
void VolumeMeshRenderer::ValidateActiveCubeMemory(uint activeCubeCount) {
    if (activeCubeCount <= this->activeCubeCountAllocted && activeCubeCount != 0) {
        return;
    }
    if (this->activeCubeCountAllocted > 0) {
        // Free CUDA memory for Marching Tetrahedrons.
        CUDA_VERIFY(cudaFree(verticesPerTetrahedron));
        CUDA_VERIFY(cudaFree(vertexOffsets));
        // Free CUDA memory for Marching Tetrahedrons.
        CUDA_VERIFY(cudaFree(eqList));
        CUDA_VERIFY(cudaFree(refList));
        CUDA_VERIFY(cudaFree(tetrahedronLabelPair));
    }
    if (this->activeCubeCount > 0) {
        this->activeCubeCountAllocted = activeCubeCount * 2;
        const size_t uintBufferSize = sizeof(uint) * this->activeCubeCountAllocted * 6;
        // Allocate CUDA memory for Marching Tetrahedrons.
        CUDA_VERIFY(cudaMalloc(&verticesPerTetrahedron, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&vertexOffsets, uintBufferSize));
        // Allocate CUDA memory for labling.
        CUDA_VERIFY(cudaMalloc(&eqList, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&refList, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&tetrahedronLabelPair, sizeof(int2) * this->activeCubeCountAllocted * 6));
    }
}

/*
 * VolumeMeshRenderer::ValidateVertexMemory
 */
void VolumeMeshRenderer::ValidateVertexMemory(uint vertexCount) {
    if (vertexCount <= this->vertexCountAllocted && vertexCount != 0) {
        return;
    }
    if (this->vertexCountAllocted > 0) {
        // Free CUDA memory for centroid map-reduce.
        CUDA_VERIFY(cudaFree(vertexLabels));
        CUDA_VERIFY(cudaFree(vertexLabelsCopy));
        CUDA_VERIFY(cudaFree(verticesCopy));
        CUDA_VERIFY(cudaFree(featureVertices));
        CUDA_VERIFY(cudaFree(featureVerticesOut));
        CUDA_VERIFY(cudaFree(featureVertexIdx));
        CUDA_VERIFY(cudaFree(featureVertexCnt));
        CUDA_VERIFY(cudaFree(featureVertexCntOut));
        CUDA_VERIFY(cudaFree(featureVertexStartIdx));
        CUDA_VERIFY(cudaFree(featureVertexIdxOut));
        CUDA_VERIFY(cudaFree(triangleAO));
        CUDA_VERIFY(cudaFree(triangleAreas));
        CUDA_VERIFY(cudaFree(centroidLabels));
        CUDA_VERIFY(cudaFree(centroidAreas));
        CUDA_VERIFY(cudaFree(centroidSums));
        CUDA_VERIFY(cudaFree(centroidCounts));
        CUDA_VERIFY(cudaFree(featureEdgeCnt));
        CUDA_VERIFY(cudaFree(featureEdgeCntOut));
        CUDA_VERIFY(cudaFree(featureEdges));
        CUDA_VERIFY(cudaFree(featureEdgesOut));
        // Free VBOs for Marching Tetrahedrons.
        DestroyVbo(&positionVbo, &positionResource);
        DestroyVbo(&normalVbo, &normalResource);
        DestroyVbo(&colorVbo, &colorResource);
        CUDA_VERIFY(cudaFreeHost(neighborAtomOfVertex));
        CUDA_VERIFY(cudaFree(neighborAtomOfVertex_d));
        CUDA_VERIFY(cudaFreeHost(vertexColors));
        CUDA_VERIFY(cudaFreeHost(featureTriangleVerticesHost));
        CUDA_VERIFY(cudaFreeHost(featureTriangleVertexIndicesHost));
        CUDA_VERIFY(cudaFreeHost(featureTriangleEdgesHost));
        CUDA_VERIFY(cudaFreeHost(featureTriangleEdgeCountHost));
    }
    if (vertexCount > 0) {
        this->vertexCountAllocted = vertexCount * 2;
        const size_t floatSize = sizeof(GLfloat) * this->vertexCountAllocted;
        const size_t float4Size = sizeof(GLfloat) * 4 * this->vertexCountAllocted;
        const size_t uintBufferSize =  sizeof(uint) * this->vertexCountAllocted;
        const size_t intBufferSize =  sizeof(int) * this->vertexCountAllocted;
        // Allocate VBOs for Marching Tetrahedrons.
        CreateVbo(&positionVbo, float4Size, &positionResource);
        CreateVbo(&normalVbo, float4Size, &normalResource);
        CreateVbo(&colorVbo, float4Size, &colorResource);
        CUDA_VERIFY(cudaMallocHost(&neighborAtomOfVertex, intBufferSize));
        CUDA_VERIFY(cudaMalloc(&neighborAtomOfVertex_d, intBufferSize));
        CUDA_VERIFY(cudaMallocHost(&vertexColors, float4Size));
        CUDA_VERIFY(cudaMallocHost(&featureTriangleVerticesHost, float4Size));
        CUDA_VERIFY(cudaMallocHost(&featureTriangleVertexIndicesHost, uintBufferSize));
        CUDA_VERIFY(cudaMallocHost(&featureTriangleEdgesHost, uintBufferSize * 2));
        CUDA_VERIFY(cudaMallocHost(&featureTriangleEdgeCountHost, uintBufferSize));
        // Allocate CUDA memory for centroid map-reduce.
        CUDA_VERIFY(cudaMalloc(&vertexLabels, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&vertexLabelsCopy, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&verticesCopy, float4Size));
        CUDA_VERIFY(cudaMalloc(&featureVertices, float4Size));
        CUDA_VERIFY(cudaMalloc(&featureVerticesOut, float4Size));
        CUDA_VERIFY(cudaMalloc(&featureVertexIdx, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureVertexCnt, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureVertexCntOut, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureVertexStartIdx, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureVertexIdxOut, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&triangleAO, floatSize / 3));
        CUDA_VERIFY(cudaMalloc(&triangleAreas, floatSize / 3));
        CUDA_VERIFY(cudaMalloc(&centroidLabels, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&centroidAreas, floatSize));
        CUDA_VERIFY(cudaMalloc(&centroidSums, float4Size));
        CUDA_VERIFY(cudaMalloc(&centroidCounts, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureEdgeCnt, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureEdgeCntOut, uintBufferSize));
        CUDA_VERIFY(cudaMalloc(&featureEdges, uintBufferSize * 2));
        CUDA_VERIFY(cudaMalloc(&featureEdgesOut, uintBufferSize * 2));
    }
}

/*
 * VolumeMeshRenderer::ValidateCentroidMemory
 */
void VolumeMeshRenderer::ValidateCentroidMemory(uint centroidCount) 
{
    if (centroidCount <= this->centroidCountAllocated && centroidCount != 0) {
        return;
    }
    if (this->centroidCountAllocated > 0) {
        // Free CUDA memory for centroids.
        CUDA_VERIFY(cudaFree(centroids));
        CUDA_VERIFY(cudaFree(centroidColors));
        CUDA_VERIFY(cudaFree(featureStartEnd));
        CUDA_VERIFY(cudaFreeHost(featureStartEndHost));
    }
    if (vertexCount > 0) {
        this->centroidCountAllocated = centroidCount * 2;        
        const size_t float4Size = sizeof(float4) * this->centroidCountAllocated;
        // Allocate CUDA memory for centroids.
        CUDA_VERIFY(cudaMalloc(&centroids, float4Size));
        CUDA_VERIFY(cudaMalloc(&centroidColors, float4Size));
        CUDA_VERIFY(cudaMalloc(&featureStartEnd, centroidCountAllocated * sizeof(uint2)));
        CUDA_VERIFY(cudaMallocHost(&featureStartEndHost, centroidCountAllocated * sizeof(uint2)));
    }
}

/*
 * VolumeMeshRenderer::CreateVbo
 */
void VolumeMeshRenderer::CreateVbo(GLuint* vbo, size_t size, struct cudaGraphicsResource** cudaResource)
{
    glGenBuffersARB(1, vbo);
    glBindBufferARB(GL_ARRAY_BUFFER, *vbo);
    glBufferDataARB(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBufferARB(GL_ARRAY_BUFFER, 0);
    CHECK_FOR_OGL_ERROR();
    CUDA_VERIFY(cudaGraphicsGLRegisterBuffer(cudaResource, *vbo, cudaGraphicsMapFlagsWriteDiscard));
}

/*
 * VolumeMeshRenderer::DestroyVbo
 */
void VolumeMeshRenderer::DestroyVbo(GLuint* vbo, struct cudaGraphicsResource** cudaResource)
{
    cudaGraphicsUnregisterResource(*cudaResource);
    glBindBufferARB(1, *vbo);
    glDeleteBuffersARB(1, vbo);
    *vbo = 0;
}

/*
 * VolumeMeshRenderer::CudaGetLast
 */
uint VolumeMeshRenderer::CudaGetLast(uint* buffer, uint* scanBuffer, size_t size)
{
    uint lastElement;
    uint lastScanElement;
    CUDA_VERIFY(cudaMemcpy(&lastElement, (buffer + size - 1), 
        sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_VERIFY(cudaMemcpy(&lastScanElement, (scanBuffer + size - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));
    return lastElement + lastScanElement;
}

/*
 * VolumeMeshRenderer::CudaVerify
 */
void VolumeMeshRenderer::CudaVerify(cudaError error, const int line)
{
    if (error != cudaSuccess) {
        const char* errorMsg = cudaGetErrorString(error);
        throw vislib::Exception(errorMsg, __FILE__, line);
    }
}

int VolumeMeshRenderer::calcMap(MolecularDataCall *mol, float *posInter,
                         int quality, float radscale, float gridspacing,
                         float isoval, bool useCol) {
    wkf_timer_start(timer);

    // If no volumetric texture will be computed we will use the cmap
    // parameter to pass in the solid color to be applied to all vertices
    //vec_copy(solidcolor, cmap);

    // compute min/max atom radius, build list of selected atom radii,
    // and compute bounding box for the selected atoms
    unsigned int i;
    float mincoord[3], maxcoord[3];
    float maxrad;
    /*
    float minx, miny, minz, maxx, maxy, maxz;
    float minrad;

    minx = maxx = posInter[0];
    miny = maxy = posInter[1];
    minz = maxz = posInter[2];
    minrad = maxrad = mol->AtomTypes()[mol->AtomTypeIndices()[0]].Radius();
    for ( i = 0; i < mol->AtomCount(); i++) {
#ifdef COMPUTE_BBOX
        int ind = i * 3;
        float tmpx = posInter[ind  ];
        float tmpy = posInter[ind+1];
        float tmpz = posInter[ind+2];

        minx = (tmpx < minx) ? tmpx : minx;
        maxx = (tmpx > maxx) ? tmpx : maxx;

        miny = (tmpy < miny) ? tmpy : miny;
        maxy = (tmpy > maxy) ? tmpy : maxy;

        minz = (tmpz < minz) ? tmpz : minz;
        maxz = (tmpz > maxz) ? tmpz : maxz;
#endif
  
        float r = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
        minrad = (r < minrad) ? r : minrad;
        maxrad = (r > maxrad) ? r : maxrad;
    }

    mincoord[0] = minx;
    mincoord[1] = miny;
    mincoord[2] = minz;
    maxcoord[0] = maxx;
    maxcoord[1] = maxy;
    maxcoord[2] = maxz;
    */
    mincoord[0] = this->dataBBox.Left();
    mincoord[1] = this->dataBBox.Bottom();
    mincoord[2] = this->dataBBox.Back();
    maxcoord[0] = this->dataBBox.Right();
    maxcoord[1] = this->dataBBox.Top();
    maxcoord[2] = this->dataBBox.Front();
    maxrad = 2.0f;

    // crude estimate of the grid padding we require to prevent the
    // resulting isosurface from being clipped
    float gridpadding = radscale * maxrad * 1.5f;
    float padrad = gridpadding;
    padrad = 0.4f * sqrtf(4.0f/3.0f*static_cast<float>(M_PI)*padrad*padrad*padrad);
    gridpadding = vislib::math::Max(gridpadding, padrad);

#if VERBOSE
    printf("  Padding radius: %.3f  (minrad: %.3f maxrad: %.3f)\n", 
        gridpadding, minrad, maxrad);
#endif

    mincoord[0] -= gridpadding;
    mincoord[1] -= gridpadding;
    mincoord[2] -= gridpadding;
    maxcoord[0] += gridpadding;
    maxcoord[1] += gridpadding;
    maxcoord[2] += gridpadding;

    // kroneml
    mincoord[0] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    mincoord[1] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    mincoord[2] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    maxcoord[0] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Right();
    maxcoord[1] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Top();
    maxcoord[2] = mol->AccessBoundingBoxes().ObjectSpaceBBox().Front();

    // compute the real grid dimensions from the selected atoms
    xaxis[0] = maxcoord[0]-mincoord[0];
    yaxis[1] = maxcoord[1]-mincoord[1];
    zaxis[2] = maxcoord[2]-mincoord[2];
    numvoxels[0] = (int) ceil(xaxis[0] / gridspacing);
    numvoxels[1] = (int) ceil(yaxis[1] / gridspacing);
    numvoxels[2] = (int) ceil(zaxis[2] / gridspacing);

    // recalc the grid dimensions from rounded/padded voxel counts
    xaxis[0] = (numvoxels[0]-1) * gridspacing;
    yaxis[1] = (numvoxels[1]-1) * gridspacing;
    zaxis[2] = (numvoxels[2]-1) * gridspacing;
    maxcoord[0] = mincoord[0] + xaxis[0];
    maxcoord[1] = mincoord[1] + yaxis[1];
    maxcoord[2] = mincoord[2] + zaxis[2];

#if VERBOSE
    printf("  Final bounding box: (%.1f %.1f %.1f) -> (%.1f %.1f %.1f)\n",
        mincoord[0], mincoord[1], mincoord[2],
        maxcoord[0], maxcoord[1], maxcoord[2]);

    printf("  Grid size: (%d %d %d)\n",
        numvoxels[0], numvoxels[1], numvoxels[2]);
#endif

    //vec_copy(origin, mincoord);
    origin[0] = mincoord[0];
    origin[1] = mincoord[1];
    origin[2] = mincoord[2];

    // build compacted lists of bead coordinates, radii, and colors
    float *xyzr = NULL;
    float *colors = NULL;

    int ind = 0;
    int ind4 = 0; 
    xyzr = (float *) malloc( mol->AtomCount() * sizeof(float) * 4);
    if (useCol) {
        colors = (float *) malloc( mol->AtomCount() * sizeof(float) * 4);

        // build compacted lists of atom coordinates, radii, and colors
        for (i = 0; i < mol->AtomCount(); i++) {
            const float *fp = posInter + ind;
            xyzr[ind4    ] = fp[0]-origin[0];
            xyzr[ind4 + 1] = fp[1]-origin[1];
            xyzr[ind4 + 2] = fp[2]-origin[2];
            xyzr[ind4 + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
 
            //const float *cp = &cmap[colidx[i] * 3];
            const float *cp = &this->atomColorTable[ind];
            colors[ind4    ] = cp[0];
            colors[ind4 + 1] = cp[1];
            colors[ind4 + 2] = cp[2];
            colors[ind4 + 3] = 1.0f;

            ind4 += 4;
            ind += 3;
        }
    } else {
        // build compacted lists of atom coordinates and radii only
        for (i = 0; i < mol->AtomCount(); i++) {
            const float *fp = posInter + ind;
            xyzr[ind4    ] = fp[0]-origin[0];
            xyzr[ind4 + 1] = fp[1]-origin[1];
            xyzr[ind4 + 2] = fp[2]-origin[2];
            xyzr[ind4 + 3] = mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius();
            ind4 += 4;
            ind += 3;
        }
    }

    // set gaussian window size based on user-specified quality parameter
    float gausslim = 2.0f;
    switch (quality) {
    case 3: gausslim = 4.0f; break; // max quality

    case 2: gausslim = 3.0f; break; // high quality

    case 1: gausslim = 2.5f; break; // medium quality

    case 0: 
    default: gausslim = 2.0f; // low quality
        break;
    }

    pretime = wkf_timer_timenow(timer);

    CUDAQuickSurf *cqs = (CUDAQuickSurf *) cudaqsurf;

    // compute both density map and floating point color texture map
    //int rc = cqs->calc_surf( mol->AtomCount(), &xyzr[0],
    //    (useCol) ? &colors[0] : &this->atomColorTable[0],
    //    useCol, origin, numvoxels, maxrad,
    //    radscale, gridspacing, gausslim,
    //    gpunumverts, gv, gn, gc, gpunumfacets, gf);
    int rc = cqs->calc_map( mol->AtomCount(), &xyzr[0],
        (useCol) ? &colors[0] : NULL,
        useCol, origin, numvoxels, maxrad,
        radscale, gridspacing, isoval, gausslim, true);

    if (rc == 0) {
        free(xyzr);
        if (colors) free(colors);
        voltime = wkf_timer_timenow(timer);
        return 0;
    } else {
        free(xyzr);
        if (colors) free(colors);  
        voltime = wkf_timer_timenow(timer);
        return 1;
    }
}

bool VolumeMeshRenderer::GetSplitMergeData(core::Call& call) {

    SplitMergeCall *smc = dynamic_cast<SplitMergeCall*>(&call);
    if (smc == NULL) return false;   

#if 1
    for( SIZE_T sIdx = smc->GetSeriesCount(); sIdx < this->splitMergeList.Count(); sIdx++ ) {
        smc->AddSeries( this->splitMergeList[sIdx]);
    }
    smc->SetTransitions( &this->transitionList);
#else
    SplitMergeCall::SplitMergeSeries *sms;
    if (smc->GetSeriesCount() == 0) {
        sms = new SplitMergeCall::SplitMergeSeries(vislib::StringA("heinz"), new MappableWibble(0));
        sms->SetColor(0.9f, 0.1f, 0.1f);
        smc->AddSeries(sms);
        sms = new SplitMergeCall::SplitMergeSeries(vislib::StringA("horscht"), new MappableWibble(1));
        sms->SetColor(0.5f, 0.3f, 0.1f);
        smc->AddSeries(sms);
        sms = new SplitMergeCall::SplitMergeSeries(vislib::StringA("hugo"), new MappableWibble(5));
        sms->SetColor(0.5f, 0.3f, 0.9f);
        smc->AddSeries(sms);
        //SplitMergeTransition *smt;
        //smt = new SplitMergeTransition(
    } else {
        // boh
    }
#endif
    // set current call time for sliding guide line
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol != NULL) {
        if (smc->GetGuideCount() == 0) {
            smc->AddGuide(mol->Calltime(), SplitMergeCall::SPLITMERGE_GUIDE_VERTICAL);
        } else {
            smc->GetGuide(0)->SetPosition(mol->Calltime());
        }
    }

    return true;
}

bool VolumeMeshRenderer::GetDiagramData(core::Call& call) {

    DiagramCall *dc = dynamic_cast<DiagramCall*>(&call);
    if (dc == NULL) return false;    
    
#if 1
    for( SIZE_T sIdx = dc->GetSeriesCount(); sIdx < this->featureList.Count(); sIdx++ ) {
        dc->AddSeries( this->featureList[sIdx]);
    }
#else
    DiagramCall::DiagramSeries *ds;
    if (dc->GetSeriesCount() == 0) {
        ds = new DiagramCall::DiagramSeries(vislib::StringA("horscht"), new MappableFloatPair(0.0f, 1.0f, false, 7));
        ds->SetColor(0.9f, 0.1f, 0.1f);
        ds->AddMarker(new DiagramCall::DiagramMarker(1, DiagramCall::DIAGRAM_MARKER_DISAPPEAR, "kaboom!"));
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("hugo"), new MappableFloatPair());
        ds->SetColor(0.2f, 1.0f, 0.2f);
        ds->AddMarker(new DiagramCall::DiagramMarker(4, DiagramCall::DIAGRAM_MARKER_SPLIT));
        ds->AddMarker(new DiagramCall::DiagramMarker(7, DiagramCall::DIAGRAM_MARKER_BOOKMARK));
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("heinz"), new MappableFloatPair(0.0f, 0.5f, true, 2));
        ds->SetColor(0.1f, 0.3f, 0.7f);
        ds->AddMarker(new DiagramCall::DiagramMarker(4, DiagramCall::DIAGRAM_MARKER_MERGE, "universe joining something\n or other"));
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("helge"), new MappableCategoryFloat(0));
        ds->SetColor(1.0f, 0.4f, 0.7f);
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("hägar"), new MappableCategoryFloat(1));
        ds->SetColor(0.6f, 0.8f, 0.7f);
        dc->AddSeries(ds);
    } else {
        ds = dc->GetSeries(0);
    }

#endif
    // set current call time for sliding guide line
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol != NULL) {
        if (dc->GetGuideCount() == 0) {
            dc->AddGuide(mol->Calltime(), DiagramCall::DIAGRAM_GUIDE_VERTICAL);
        } else {
            dc->GetGuide(0)->SetPosition(mol->Calltime());
        }
    }

    return true;
}

bool VolumeMeshRenderer::GetCenterLineDiagramData(core::Call& call) {

    DiagramCall *dc = dynamic_cast<DiagramCall*>(&call);
    if (dc == NULL) return false;
    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL) return false;
    
#if 1
    vislib::StringA featureName;
    DiagramCall::DiagramSeries *ds;
    MolecularSurfaceFeature *ms, *prevMS;
	prevMS = nullptr;
    for( SIZE_T fIdx = this->featureCenterLines.Count(); fIdx < (this->clg.Count() + 1); fIdx++ ) {
        if( fIdx == 0 ) {
            // add first feature (outer surface)
            featureName.Format( "Feature %i (Surface)", fIdx);
        } else {
            if( this->clg[fIdx-1]->fType == CenterLineGenerator::CAVITY )
                featureName.Format( "Feature %i (Cavity)", fIdx);
            else if( this->clg[fIdx-1]->fType == CenterLineGenerator::CHANNEL )
                featureName.Format( "Feature %i (Channel)", fIdx);
            else
                featureName.Format( "Feature %i (Pocket)", fIdx);
        }
        ds = new DiagramCall::DiagramSeries( featureName, 
            new MolecularSurfaceFeature( 1.0f));
        ds->SetColor( this->featureList[fIdx]->GetColor());
        this->featureCenterLines.Append( ds);
    }
    
    float length = 0.0f;
    float maxLength = 0.0f;
    for( SIZE_T fIdx = 1; fIdx < this->featureCenterLines.Count(); fIdx++ ) {
        if( this->clEdges[fIdx-1].empty() ) continue;
        // sum up edge length and reset edge as not visited
        length = 0.0f;
        for( auto edge : this->clEdges[fIdx-1] ) {
            //length += (edge->node1->p - edge->node2->p).Length();
            length++;
            //length++;
            edge->visited = false;
        }
        maxLength = vislib::math::Max( maxLength, length);
    }
    for( SIZE_T fIdx = 1; fIdx < this->featureCenterLines.Count(); fIdx++ ) {
        if( this->clEdges[fIdx-1].empty() ) continue;
        ds = this->featureCenterLines[fIdx];
        length = 0.0f;
        /*
        // sum up edge length and reset edge as not visited
        for( auto edge : this->clEdges[fIdx] ) {
            length += (edge->node1->p - edge->node2->p).Length();
            //length++;
            edge->visited = false;
        }
        */
        //ms = new MolecularSurfaceFeature( length);
        ms = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
        ms->ClearValues();
        ms->SetMaxTime( maxLength);
        auto edge = this->clEdges[fIdx-1].front();
        edge->visited = true;
        auto node1 = edge->node1;
        auto node2 = edge->node2;
        if( node2->isStartNode ) {
            node1 = edge->node2;
            node2 = edge->node1;
        }
        length = 0;
        ms->AppendValue( length, node1->minimumDistance);
        //length += (node1->p - node2->p).Length();
        length++;
        //length++;
        ms->AppendValue( length, node2->minimumDistance);
        while( node2 != nullptr && !node2->isStartNode ) {
            node1 = node2;
            node2 = nullptr;
            for( auto nextEdge : this->clEdges[fIdx-1] ) {
                if( !nextEdge->visited ) {
                    // TODO support branching! (recursion)
                    if( nextEdge->node1 == node1 ) {
                        node2 = nextEdge->node2;
                        //length += (node1->p - node2->p).Length();
                        length++;
                        ms->AppendValue( length, node2->minimumDistance);
                        nextEdge->visited = true;
                        break;
                    } else if( nextEdge->node2 == node1 ) {
                        node2 = nextEdge->node1;
                        //length += (node1->p - node2->p).Length();
                        length++;
                        ms->AppendValue( length, node2->minimumDistance);
                        nextEdge->visited = true;
                        break;
                    }
                }
            }
            // set marker if the last node was a start node
            if( node2 != nullptr && node2->isStartNode ) {
                ds->AddMarker(new DiagramCall::DiagramMarker(ms->GetDataCount() - 1, DiagramCall::DIAGRAM_MARKER_EXIT, "Channel exit!"));
            }
        }
        //prevMS = static_cast<MolecularSurfaceFeature*>(ds->GetMappable());
        //delete prevMS;
        //ds->SetMappable( ms);
    }

    for( SIZE_T sIdx = dc->GetSeriesCount(); sIdx < this->featureCenterLines.Count(); sIdx++ ) {
        dc->AddSeries( this->featureCenterLines[sIdx]);
    }
#else
    DiagramCall::DiagramSeries *ds;
    if (dc->GetSeriesCount() == 0) {
        ds = new DiagramCall::DiagramSeries(vislib::StringA("horscht"), new MappableFloatPair(0.0f, 1.0f, false, 7));
        ds->SetColor(0.9f, 0.1f, 0.1f);
        ds->AddMarker(new DiagramCall::DiagramMarker(1, DiagramCall::DIAGRAM_MARKER_DISAPPEAR, "kaboom!"));
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("hugo"), new MappableFloatPair());
        ds->SetColor(0.2f, 1.0f, 0.2f);
        ds->AddMarker(new DiagramCall::DiagramMarker(4, DiagramCall::DIAGRAM_MARKER_SPLIT));
        ds->AddMarker(new DiagramCall::DiagramMarker(7, DiagramCall::DIAGRAM_MARKER_BOOKMARK));
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("heinz"), new MappableFloatPair(0.1f, 0.5f, true, 2));
        ds->SetColor(0.1f, 0.3f, 0.7f);
        ds->AddMarker(new DiagramCall::DiagramMarker(4, DiagramCall::DIAGRAM_MARKER_MERGE, "universe joining something\n or other"));
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("helge"), new MappableCategoryFloat(0));
        ds->SetColor(1.0f, 0.4f, 0.7f);
        dc->AddSeries(ds);
        ds = new DiagramCall::DiagramSeries(vislib::StringA("hägar"), new MappableCategoryFloat(1));
        ds->SetColor(0.6f, 0.8f, 0.7f);
        dc->AddSeries(ds);
    } else {
        ds = dc->GetSeries(0);
    }
#endif

    return true;
}
