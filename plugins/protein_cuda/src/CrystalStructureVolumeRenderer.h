/*
 * CrystalStructureVolumeRenderer.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINCUDAPLUGIN_CRYSTALSTRUCTUREVOLUMERENDERER_H
#define MMPROTEINCUDAPLUGIN_CRYSTALSTRUCTUREVOLUMERENDERER_H

#include "protein_calls/CrystalStructureDataCall.h"
#include "CUDACurl.cuh"
#include "UniGrid3D.h"
#include "CUDAMarchingCubes.h"

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/CallRender3D.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "mmcore/BoundingBoxes.h"

namespace megamol {
namespace protein_cuda {

/**
 * Renderer class combining volume rendering and raycasting.
 */
class CrystalStructureVolumeRenderer : public core::view::Renderer3DModuleDS {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "CrystalStructureVolumeRenderer";
    }

    /** Ctor. */
    CrystalStructureVolumeRenderer(void);

    /** Dtor. */
    virtual ~CrystalStructureVolumeRenderer(void);

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers volume rendering for data based on crystal structures.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        if(!vislib::graphics::gl::GLSLShader::AreExtensionsAvailable())
            return false;
        return true;
    }

protected:

    /** Arrow color mode */
    enum ArrowColorMode {ARRCOL_ELEMENT, ARRCOL_ORIENT, ARRCOL_MAGNITUDE};

    /** Atom rendering modes */
    enum AtomRenderMode {ATOM_NONE, ATOM_SPHERES};

    /** Rendering modes for ba edges */
    enum EdgeBaRenderMode {BA_EDGE_NONE, BA_EDGE_LINES, BA_EDGE_STICK};

    /** Rendering modes for ti edges */
    enum EdgeTiRenderMode {TI_EDGE_NONE, TI_EDGE_LINES, TI_EDGE_STICK};


    /** Slice render mode */
    enum SliceRenderMode {VEC_MAG, VEC_DIR, LIC_GPU, ROT_MAG, SLICE_DENSITYMAP,
        SLICE_DELTA_X, SLICE_DELTA_Y, SLICE_DELTA_Z,
        SLICE_COLORMAP, SLICE_NONE};

    /** Rendering modes for vectors */
    enum VecRenderMode {VEC_NONE, VEC_ARROWS};

    /** Coloring modes for the raymarching iso surface */
    enum VolColorMode {VOL_UNI, VOL_DIR, VOL_MAG, VOL_LIC};

    /** Texture to be sued for raymarching */
    enum RayMarchTex {DENSITY, DIR_MAG, CURL_MAG};

    /**
     * (Re)calculate visibility of atoms, edges and cells. Should get called
     * before any rendering happens.
     *
     * @param[in] dc The data call
     */
    void ApplyPosFilter(const protein_calls::CrystalStructureDataCall *dc);

    /**
     * Calculate density map after filtering dipole vectors and setup density
     * texture. 'CalcUniGrid' and 'CalcMagCurl' have to be called first.
     *
     * @param dc The data call
     * @param[in] atomPos The atom positions
     * @return 'True' on success, 'false' otherwise
     */
    bool CalcDensityTex(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos);

    /**
     * Calculate the curl magnitude for the vector field.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool CalcMagCurlTex();

    /**
     * Calculate uniform grid based on the vector field.
     *
     * @param[in] dc The data call.
     * @param[in] atomPos The atom positions
     * @param[in] atomCol The atom colors (may be NULL)
     * @return True, if the displacement map could be calculated.
     */
    bool CalcUniGrid(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos, const float *col);

    /**
     * Implementation of 'create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Create fbo to hold depth values for ray casting.
     *
     * @param[in] width The width of the fbo
     * @param[in] height The height of the fbo
     *
     * @return 'True', if the fbo could be created
     */
    bool CreateFbo(UINT width, UINT height);

    /**
     * Create source fbo.
     *
     * @param[in] width The width of the fbo
     * @param[in] height The height of the fbo
     *
     * @return 'True', if the fbo could be created
     */
    bool CreateSrcFbo(size_t width, size_t height);

    /**
     * Free all memory allocated by the class.
     */
    void FreeBuffs();

    /**
     * Apply several filter to the vector field in order to ger clusters.
     *
     * @param[in] dc      The data call
     * @param[in] atomPos The atom positions
     */
    void FilterVecField(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param[in] call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Initialize parameters for the LIC calculation and setup random texture.
     *
     * @param[in] dc The data call
     * @return 'True' on success, 'false' otherwise
     */
    bool InitLIC();

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * The Open GL Render callback.
     *
     * @param[in] call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

    /**
     * Render the vector field as arrows glyphs.
     *
     * @param[in] dc The data call
     * @param[in] atomPos The atom positions
     * @param[in] atomCol The atom colors (may be NULL)
     * @return 'True' on success, 'false' otherwise
     */
    bool RenderVecFieldArrows(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos, const float *col);

    /**
     * Render atom positions as spheres.
     *
     * @param[in] dc The data call
     */
	void RenderAtomsSpheres(const protein_calls::CrystalStructureDataCall *dc);

    /**
     * Render critical points as spheres.
     *
     * @param[in] dc The data call
     */
	void RenderCritPointsSpheres(const protein_calls::CrystalStructureDataCall *dc);

    /**
     * Render edges between Ba atoms as lines.
     *
     * @param[in] dc      The data call
     * @param[in] atomPos The atom positions
     * @param[in] atomCol The atom colors (may be NULL)
     */
	void RenderEdgesBaLines(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos, const float *atomCol);

    /**
     * Render edges between Ba atoms as sticks. Overloaded function which
     * receives explicit atom positions and atom colors.
     *
     * @param[in] dc      The data call
     * @param[in] atomPos The atom positions
     * @param[in] atomCol The atom colors
     */
	void RenderEdgesBaStick(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos, const float *atomCol);

    /**
     * Render edges between Ti atoms as lines.
     *
     * @param[in] dc      The data call
     * @param[in] atomPos The atom positions
     * @param[in] atomCol The atom colors
     */
	void RenderEdgesTiLines(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos, const float *atomCol);

    /**
     * Render edges between Ti atoms as sticks. Overloaded function which
     * receives explicit atom positions and atom colors.
     *
     * @param[in] dc      The data call
     * @param[in] atomPos The atom positions
     * @param[in] atomCol The atom colors
     */
	void RenderEdgesTiStick(const protein_calls::CrystalStructureDataCall *dc,
            const float *atomPos, const float *atomCol);

    /**
     * Render the unigrid curl magnitude as iso surface using CUDA.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool RenderIsoSurfMC();

    /**
     * Helper function to render a cube representing the volume.
     */
    void RenderVolCube();

    /**
     * Render the volume via raycasting.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool RenderVolume();

    /**
     * Sets up the array with the atom colors.
     *
     * @param[in] dc The data call
     * @return 'True' on success, 'false' otherwise
     */
	bool SetupAtomColors(const protein_calls::CrystalStructureDataCall *dc);

    /**
     * Update all parameters if necessary.
     *
     * @param[in] dc The data call
     * @return 'True' on success, 'false' otherwise
     */
	bool UpdateParams(const protein_calls::CrystalStructureDataCall *dc);

private:

    /**
     * Calculates the volume of one cell defined by the points A-H. The cell
     * doesn't have to be symmetrical. The calculation takes place by
     * dividing the cell into 5 tetrahedrons and calculating their respective
     * volume.
     *
     * @param[in] A,B,C,D,E,F,G,H The corner points of the cell
     * @return The volume of the cell
     */
    float calcCellVolume(
            vislib::math::Vector<float, 3> A,
            vislib::math::Vector<float, 3> B,
            vislib::math::Vector<float, 3> C,
            vislib::math::Vector<float, 3> D,
            vislib::math::Vector<float, 3> E,
            vislib::math::Vector<float, 3> F,
            vislib::math::Vector<float, 3> G,
            vislib::math::Vector<float, 3> H);

    /**
     * Calculates the volume of one tetrahedron defined by the points A-D.
     *
     * @param[in] A,B,C,D The corner points
     * @return The volume of the tetrahedron
     */
    float calcVolTetrahedron(
            vislib::math::Vector<float, 3> A,
            vislib::math::Vector<float, 3> B,
            vislib::math::Vector<float, 3> C,
            vislib::math::Vector<float, 3> D);

    /**
     * Computes color gradient based on scalar value.
     *
     * @param[in] val The scalar value
     * @return The color gradient
     */
    vislib::math::Vector<float, 3> getColorGrad(float val);

    /**
     * Load a VTK mesh.
     */
    bool loadVTKMesh(vislib::StringA filename);

    /// Caller slot
    core::CallerSlot dataCallerSlot;

    /// Parameter for positional interpolation
    core::param::ParamSlot interpolParam;
    bool interpol;

    /// Parameter for minimal vector length
    core::param::ParamSlot minVecMagParam;
    float minVecMag;

    /// Parameter for maximum vector length
    core::param::ParamSlot maxVecMagParam;
    float maxVecMag;

    /// Parameter for maximum curl
    core::param::ParamSlot maxVecCurlParam;
    float maxVecCurl;

    /// Parameter for the atom render mode
    core::param::ParamSlot atomRenderModeParam;
    AtomRenderMode atomRM;

    /// Parameter for the sphere radius scale factor
    core::param::ParamSlot sphereRadParam;
    float sphereRad;

    /// Parameter slot for the ba edge render mode
    core::param::ParamSlot edgeBaRenderModeParam;
    EdgeBaRenderMode edgeBaRM;

    /// Parameter slot for the stick radius for ba edge rendering
    core::param::ParamSlot baStickRadiusParam;
    float baStickRadius;

    /// Parameter for the ti edge render mode
    core::param::ParamSlot edgeTiRenderModeParam;
    EdgeTiRenderMode edgeTiRM;

    /// Parameter for the stick radius for ti edge rendering
    core::param::ParamSlot tiStickRadiusParam;
    float tiStickRadius;

    /// Parameter for the displacement render mode
    core::param::ParamSlot vecRMParam;
    VecRenderMode vecRM;

    /// Parameter arrow radius
    core::param::ParamSlot arrowRadParam;
    float arrowRad;

    /// Parameter toggle filtering of arrow glyphs
    core::param::ParamSlot arrowUseFilterParam;
    bool arrowUseFilter;

    /// Parameter for the ba atom filtering
    core::param::ParamSlot showBaAtomsParam;
    bool showBaAtoms;

    /// Parameter for the ti atom filtering
    core::param::ParamSlot showTiAtomsParam;
    bool showTiAtoms;

    /// Parameter for the o atom filtering
    core::param::ParamSlot showOAtomsParam;
    bool showOAtoms;

    /// Parameter to change the arrow color mode
    core::param::ParamSlot arrColorModeParam;
    ArrowColorMode arrColorMode;

    /// Parameter for vector scale factor
    core::param::ParamSlot vecSclParam;
    float vecScl;

    /// Parameter to toggle vector normalization
    core::param::ParamSlot toggleNormVecParam;
    bool toggleNormVec;

    /// Parameter slot for the atom filter
    core::param::ParamSlot filterAllParam;
    float posFilterAll;

    /// Parameter for max x atom filter
    core::param::ParamSlot filterXMaxParam;
    float posXMax;

    /// Parameter for max y atom filter
    core::param::ParamSlot filterYMaxParam;
    float posYMax;

    /// Parameter for max z atom filter
    core::param::ParamSlot filterZMaxParam;
    float posZMax;

    /// Parameter for min x atom filter
    core::param::ParamSlot filterXMinParam;
    float posXMin;

    /// Parameter for min y atom filter
    core::param::ParamSlot filterYMinParam;
    float posYMin;

    /// Parameter for min z atom filter
    core::param::ParamSlot filterZMinParam;
    float posZMin;

    /// Parameter for grid render mode
    core::param::ParamSlot sliceRenderModeParam;
    SliceRenderMode sliceRM;

    /// Parameter for grid spacing
    core::param::ParamSlot gridSpacingParam;
    float gridSpacing;

    /// Parameter for assumed radius of grid data
    core::param::ParamSlot gridDataRadParam;
    float gridDataRad;

    /// Parameter for assumed radius of density grid data
    core::param::ParamSlot densGridRadParam;
    float densGridRad;

    /// Parameter for assumed radius of density grid data
    core::param::ParamSlot densGridGaussLimParam;
    float densGridGaussLim;

    /// Parameter for assumed radius of density grid data
    core::param::ParamSlot densGridSpacingParam;
    float densGridSpacing;

    /// Parameter for grid interpolation quality
    core::param::ParamSlot gridQualityParam;
    int gridQuality;

    /// Parameter for x-Plane
    core::param::ParamSlot xPlaneParam;
    float xPlane;

    /// Parameter to toggle x-Plane
    core::param::ParamSlot toggleXPlaneParam;
    bool showXPlane;

    /// Parameter for y-Plane
    core::param::ParamSlot yPlaneParam;
    float yPlane;

    /// Parameter to toggle y-Plane
    core::param::ParamSlot toggleYPlaneParam;
    bool showYPlane;

    /// Parameter for z-Plane
    core::param::ParamSlot zPlaneParam;
    float zPlane;

    /// Parameter to toggle z-Plane
    core::param::ParamSlot toggleZPlaneParam;
    bool showZPlane;

    /// Parameter to scale LIC direction vectors
    core::param::ParamSlot licDirSclParam;
    float licDirScl;

    /// Parameter for LIC stream line length
    core::param::ParamSlot licStreamlineLengthParam;
    unsigned int licStreamlineLength;

    /// Parameter to toggle 2d projection of vectors
    core::param::ParamSlot projectVec2DParam;
    bool projectVec2D;

    /// Parameter for size of random buffer used for LIC computation
    core::param::ParamSlot licRandBuffSizeParam;
    int licRandBuffSize;

    /// Parameter for LIC contrast stretching
    core::param::ParamSlot licContrastStretchingParam;
    float licContrastStretching;

    /// Parameter for LIC brightness
    core::param::ParamSlot licBrightParam;
    float licBright;

    /// Parameter to scale texture coordinates for lic
    core::param::ParamSlot licTCSclParam;
    float licTCScl;

    /// Parameter for LIC contrast stretching
    core::param::ParamSlot volLicContrastStretchingParam;
    float volLicContrastStretching;

    /// Parameter for LIC brightness
    core::param::ParamSlot volLicBrightParam;
    float volLicBright;

    /// Parameter to scale data visualized by slices
    core::param::ParamSlot sliceDataSclParam;
    float sliceDataScl;

    /// Parameter for toggling the rendering of critical points
    core::param::ParamSlot showCritPointsParam;
    bool showCritPoints;

    /// Parameter to toggle positional filtering for cirical points
    core::param::ParamSlot cpUsePosFilterParam;
    bool cpUsePosFilter;

    /// Parameter to change color mode
    core::param::ParamSlot vColorModeParam;
    VolColorMode vColorMode;

    /// Parameter to scale LIC direction vectors on isosurface
    core::param::ParamSlot volLicDirSclParam;
    float volLicDirScl;

    /// Parameter for LIC stream line length on isosurface
    core::param::ParamSlot volLicLenParam;
    unsigned int volLicLen;

    /// Parameter to toggle rendering of volume texture
    core::param::ParamSlot volShowParam;
    bool volShow;

    /// Parameter for step size for rendering of volume texture
    core::param::ParamSlot volDeltaParam;
    float volDelta;

    /// Parameter for iso value for rendering of volume texture
    core::param::ParamSlot volIsoValParam;
    float volIsoVal;

    /// Parameter to scale alpha value of volume
    core::param::ParamSlot volAlphaSclParam;
    float volAlphaScl;

    /// Parameter to sclae texture coordinates for isosurface lic
    core::param::ParamSlot volLicTCSclParam;
    float volLicTCScl;

    /// Parameter to toggle rendering of the iso surface
    core::param::ParamSlot showIsoSurfParam;
    bool showIsoSurf;

    /// Parameter to determine the texture used for ray marching
    core::param::ParamSlot rmTexParam;
    RayMarchTex rmTex;

    /// Parameter for the minimum z-value for fog
    core::param::ParamSlot fogStartParam;
    float fogStart;

    /// Parameter for the maximum z-value for fog
    core::param::ParamSlot fogEndParam;
    float fogEnd;

    /// Parameter for fog density
    core::param::ParamSlot fogDensityParam;
    float fogDensity;

    /// Parameter for fog color
    core::param::ParamSlot fogColourParam;
    float fogColour[4];

    // Toggle rendering of ridges
    core::param::ParamSlot showRidgeParam;
    bool showRidge;

    /// Parameter or meshfile (containing ridges)
    core::param::ParamSlot meshFileParam;
    bool renderMesh;
    vislib::Array<double> meshVertices;
    vislib::Array<double> meshNormals;
    vislib::Array<int> meshFaces;
    vislib::Array<double> mesh;

    core::param::ParamSlot toggleIsoSurfaceSlot;
    core::param::ParamSlot toggleCurlFilterSlot;


    /// Flag whether the uniform grid has to be recalculated
    bool recalcGrid;

    /// Flag whether critical points have to be recalculated
    bool recalcCritPoints;

    /// Flag whether the curl magnitude texture has to be recalculated
    bool recalcCurlMag;

    /// Flag whether displacement data has to be recalculated
    bool recalcArrowData;

    /// Flag whether posInter array has to be recalculated
    bool recalcPosInter;

    /// Tells whether the arrow data array has to be recalculated
    bool recalcVisibility;

    /// Flag whether the dipole has to be recalculated
    bool recalcDipole;

    /// Flag whether the density grid has to be recalculated
    bool recalcDensityGrid;

    /// Flag whether the vector field filter has to be reapplied
    bool filterVecField;


    /// Texture holding atom displacement
    GLuint uniGridTex;

    /// Texture holding atom density map
    GLuint uniGridDensityTex;

    /// Texture holding colors for the atom density map
    GLuint uniGridColorTex;

    /// Frame buffer object for raycasting
    vislib::graphics::gl::FramebufferObject rcFbo;

    /// Frame buffer object for opaque objects of the scene
    vislib::graphics::gl::FramebufferObject srcFbo;

    /// Texture for curl magnitude
    GLuint curlMagTex;

    /// Random noise texture
    GLuint randNoiseTex;


    /// Shader for rendering volume slice
    vislib::graphics::gl::GLSLShader vrShader;

    /// Shader for rendering spheres
    vislib::graphics::gl::GLSLShader sphereShader;

    /// Shader for rendering arrows
    vislib::graphics::gl::GLSLGeometryShader arrowShader;

    /// Shader for rendering cylinders
    vislib::graphics::gl::GLSLShader cylinderShader;

    /// Shader for raycasting
    vislib::graphics::gl::GLSLShader rcShader;

    /// Shader for rendering the cube backface
    vislib::graphics::gl::GLSLShader rcShaderDebug;

    /// Shader for per pixel lighting and clipping
    vislib::graphics::gl::GLSLShader pplShaderClip;

    /// Attribute array for cylinder shader
    GLint attribLocInParams;

    /// Attribute array for cylinder shader
    GLint attribLocQuatC;

    /// Attribute array for cylinder shader
    GLint attribLocColor1;

    /// Attribute array for cylinder shader
    GLint attribLocColor2;

    /// Shader for per pixel lighting
    vislib::graphics::gl::GLSLShader pplShader;


    /// Array for current frame
    float *frame0;

    /// Array for interpolation frame data
    float *frame1;

    /// Array for critical points of the vector field
    vislib::Array<float> critPoints;

    /// Array containing the result of the curl operator (device memory)
    float *gridCurlMagD;

    /// Array containing the result of the curl operator (device memory)
    float *gridCurlD;

    /// Vertex output array for CUDA marching cubes (host memory)
    float *mcVertOut;

    /// Vertex output array for CUDA marching cubes (device memory)
    float3 *mcVertOut_D;

    /// Normal output array for CUDA marching cubes (host memory)
    float *mcNormOut;

    /// Normal output array for CUDA marching cubes (device memory)
    float3 *mcNormOut_D;

    /// Array holding interpolated atom  positions
    vislib::Array<float> posInter;

    ///  Array with visibility information of displacement vectors
    vislib::Array<bool> arrowVis;

    /// Array with displacement data
    vislib::Array<float> arrowData;

    /// Array with displacement data
    vislib::Array<float> arrowDataDipole;

    /// Array with displacement data
    vislib::Array<float> arrowDataPos;

    /// Color array for the displacement arrows
    vislib::Array<float> arrowCol;

    /// Vertex array for atom positions
    vislib::Array<float> atomPosSpheres;

    /// Color array for atom colors
    vislib::Array<float> atomColor;

    /// Vertex array for ba edges in stick mode
    vislib::Array<int> edgeIdxBa;

    /// Vertex array for ti edges in stick mode
    vislib::Array<int> edgeIdxTi;

    ///  Array with indices of visible atoms
    vislib::Array<int> visAtomIdx;

    ///  Array with visibility information of atoms
    vislib::Array<bool> visAtom;

    ///  Array with visibility information of dipoles
    vislib::Array<bool> visDipole;

    ///  Array with indices of visible displacement vectors
    vislib::Array<int> arrowVisIdx;

    /// Vertex array for cylinders
    vislib::Array<float> vertCylinders;

    /// Attribute array for quaternions of the cylinders
    vislib::Array<float> quatCylinders;

    /// attribute array for inParam of the cylinders (radius and length)
    vislib::Array<float> inParaCylinders;

    /// first color array for cylinder
    vislib::Array<float> color1Cylinders;

    /// second color array for cylinder
    vislib::Array<float> color2Cylinders;

    /// Array with all visible cells
    vislib::Array<int> visCell;

    /// Uniform grid with the vector field
    UniGrid3D<float3> uniGridVecField;

    /// Uniform grid with the density map used to extract the iso surface
    UniGrid3D<float> uniGridDensity;

    /// Uniform grid containing the curl magnitude
    UniGrid3D<float> uniGridCurlMag;

    /// Uniform grid with the density map used to extract the iso surface
    UniGrid3D<float3> uniGridColor;

    /// Uniform grid containing random buffer
    UniGrid3D<float> licRandBuff;

    /// The index of the last frame
    int idxLastFrame;

    /// Pointer to CUDAQuickSurf object if it exists */
    void *cudaqsurf;

    /// The number of atoms
    unsigned int atomCnt;

    /// The number of visible atoms
    unsigned int visAtomCnt;

    /// The number of visible ba edges
    unsigned int edgeCntBa;

    /// The number of visible ti edges
    unsigned int edgeCntTi;

    /// Call time of the last frame
    float callTimeOld;

    /// The dimensions of the rc fbo
    vislib::math::Vector<int, 2> fboDim;

    /// The dimensions of the source fbo
    vislib::math::Vector<int, 2> srcFboDim;

    /// Camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

    /// CUDA Parameters for curl calculation
    CurlGridParams params;

    /// CUDA marching cubes object
    CUDAMarchingCubes *cudaMC;

    /// Maximum vertices for CUDA marching cubes
    unsigned int nVerticesMCOld;

    /// The data sets bounding box
    megamol::core::BoundingBoxes bbox;

    /// The maximum scaled vector magnitude (used for color gradient)
    float vecMagSclMax;

    /// The minimum scaled vector magnitude (used for color gradient)
    float vecMagSclMin;

    /// The maximum scaled vector length
    float maxLenDiff;

    int frameOld;
    
    bool setCUDAGLDevice;
};


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_CRYSTALSTRUCTUREVOLUMERENDERER_H */
