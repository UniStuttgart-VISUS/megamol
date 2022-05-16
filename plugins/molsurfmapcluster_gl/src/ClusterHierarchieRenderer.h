/*
 * ClusterRenderer.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MOLSURFMAPCLUSTER_CLUSTERHIERARCHIERENDERER_INCLUDED
#define MOLSURFMAPCLUSTER_CLUSTERHIERARCHIERENDERER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "vislib/math/Rectangle.h"
#include "vislib/Array.h"

#include "CallCluster.h"
#include "ClusterRenderer.h"
#include <set>

namespace megamol {
namespace molsurfmapcluster {

/**
 * Mesh-based renderer for bézier curve tubes
 */
class ClusterHierarchieRenderer : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ClusterHierarchieRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Render the Hierarchie of the given Clustering"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ClusterHierarchieRenderer(void);

    /** Dtor. */
    virtual ~ClusterHierarchieRenderer(void);

    /** The mouse button pressed/released callback. */
    virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) override;

    /** The mouse movement callback. */
    virtual bool OnMouseMove(double x, double y) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     */
    virtual bool GetExtents(core_gl::view::CallRender2DGL& call);

    /**
     * The render callback.
     */
    virtual bool Render(core_gl::view::CallRender2DGL& call);

    /**
     * The GetData and GetExtents Callback for Position
     */
    virtual bool GetPositionExtents(core::Call& call);
    virtual bool GetPositionData(core::Call& call);

private:
    /**********************************************************************
     * variables
     **********************************************************************/

    enum class DistanceColorMode {
        NONE = 0,
        BRENDA = 1,
        TMSCORE = 2
    };

    /** Parameters */
    core::param::ParamSlot showEnzymeClassesParam;
    core::param::ParamSlot showPDBIdsParam;
    core::param::ParamSlot fontSizeParam;
    core::param::ParamSlot useDistanceColors;
    core::param::ParamSlot sizeMultiplierParam;

    core::param::ParamSlot addMapParam;
    core::param::ParamSlot addIdParam;
    core::param::ParamSlot addBrendaParam;

    core::param::ParamSlot minColorParam;
    core::param::ParamSlot midColorParam;
    core::param::ParamSlot maxColorParam;
    core::param::ParamSlot failColorParam;
    core::param::ParamSlot distanceMatrixParam;

    core::param::ParamSlot windowWidthParam;
    core::param::ParamSlot windowHeightParam;

    /**The Viewport*/
    vislib::math::Vector<float, 2> viewport;
    glm::ivec2 windowMeasurements;
    float zoomFactor;

    // Font rendering
    megamol::core::utility::SDFFont theFont;

    /** DataHash*/
    SIZE_T lastHashClustering;
    SIZE_T lastHashPosition;
    SIZE_T DataHashPosition;

    /** Clustering*/
    HierarchicalClustering* clustering;
    HierarchicalClustering::CLUSTERNODE* position;
    HierarchicalClustering::CLUSTERNODE* root;

    std::vector<HierarchicalClustering::CLUSTERNODE*>* cluster;
    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>* colors;

    bool rendered;
    unsigned int counter;

    bool newcolor;
    SIZE_T hashoffset;
    SIZE_T colorhash;

    vislib_gl::graphics::gl::GLSLShader passthroughShader;
    vislib_gl::graphics::gl::GLSLShader textureShader;
    std::unique_ptr<glowl::BufferObject> texBuffer;
    std::unique_ptr<glowl::BufferObject> geometrySSBO;

    GLuint texVa;
    GLuint dummyVa;

    // Popup
    HierarchicalClustering::CLUSTERNODE* popup;
    HierarchicalClustering::CLUSTERNODE* leftmarked;
    HierarchicalClustering::CLUSTERNODE* rightmarked;
    int x;
    int y;


    /* Mouse Variablen*/
    /** The current mouse coordinates */
    float mouseX;
    float mouseY;

    /** The last mouse coordinates */
    float lastMouseX;
    float lastMouseY;

    core::view::MouseButton mouseButton;
    core::view::MouseButtonAction mouseAction;


    /*** INPUT ********************************************************/


    /**********************************************************************
     * functions
     **********************************************************************/
    double drawTree(HierarchicalClustering::CLUSTERNODE*, glm::mat4, double, double, double, double,
        std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>*);
    void renderPopup(glm::mat4);
    void renderMap(glm::mat4 mvp, glm::vec2 lowerleft, glm::vec2 upperright, PictureData* data);
    double checkposition(HierarchicalClustering::CLUSTERNODE*, float, float, double, double, double, double, double = 5.0, double = 5.0);
    float enzymeClassDistance(const std::array<int, 4>& arr1, const std::array<int, 4>& arr2);


    /**********************************************************************
     * callback stuff
     **********************************************************************/

    /** The input data slot. */
    core::CallerSlot clusterDataSlot;
    core::CallerSlot positionDataSlot;

    core::CalleeSlot positionoutslot;
    bool newposition;
    bool dbscanclustercolor;
    std::set<std::string> dbscancluster;
};

} // namespace MolSurfMapCluster
} // namespace megamol

#endif /*MOLSURFMAPCLUSTER_CLUSTERHIERARCHIERENDERER_INCLUDED*/
