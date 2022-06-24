/*
 * ClusterRenderer.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MOLSURFMAPCLUSTER_CLUSTERRENDERER_INCLUDED
#define MOLSURFMAPCLUSTER_CLUSTERRENDERER_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include "glowl/glowl.h"
#include "vislib/Array.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"

#include "CallCluster.h"


namespace megamol {
namespace molsurfmapcluster {

/**
 * Mesh-based renderer for bézier curve tubes
 */
class ClusterRenderer : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ClusterRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renders a set of PNG-Pictures in their Clusters";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    ClusterRenderer(void);

    /** Dtor. */
    virtual ~ClusterRenderer(void);

    /** The mouse button pressed/released callback. */
    virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) override;

    /** The mouse movement callback. */
    virtual bool OnMouseMove(double x, double y) override;

    // struct rgb
    struct RGBCOLOR {
        double r;
        double g;
        double b;
    };

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

    core::param::ParamSlot colorTableFileParam;

    /**The Viewport*/
    vislib::math::Vector<float, 2> viewport;

    // Font rendering
    megamol::core::utility::SDFFont theFont;
    float fontSize;

    /* Textures */
    GLuint* textureNumber;

    /** DataHash*/
    SIZE_T lastHash;
    SIZE_T DataHashPosition;
    SIZE_T hashoffset;
    SIZE_T GetPositionDataHash;

    /** Clustering*/
    HierarchicalClustering* clustering;
    HierarchicalClustering::CLUSTERNODE* root;
    std::vector<HierarchicalClustering::CLUSTERNODE*>* cluster;

    bool reloadTexures;
    bool init;
    double maxX;
    double maxY;
    double minX;
    double minY;

    double maxdistance;

    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, ClusterRenderer::RGBCOLOR*>*>* colors;

    /*** INPUT ********************************************************/

    /** The current mouse coordinates */
    float mouseX;
    float mouseY;

    /** The last mouse coordinates */
    float lastMouseX;
    float lastMouseY;

    core::view::MouseButton mouseButton;
    core::view::MouseButtonAction mouseAction;

    bool actionavailable;

    /**********************************************************************
     * functions
     **********************************************************************/
    void renderNode(HierarchicalClustering::CLUSTERNODE*, glm::mat4, double = 0, double = 0, double = 0, double = 0);
    void renderAllLeaves(
        HierarchicalClustering::CLUSTERNODE*, glm::mat4, double = 0, double = 0, double = 0, double = 0);
    void renderRootNode(
        HierarchicalClustering::CLUSTERNODE*, glm::mat4, double = 0, double = 0, double = 0, double = 0);
    void renderLeaveNode(HierarchicalClustering::CLUSTERNODE*, glm::mat4);
    void renderClusterText(HierarchicalClustering::CLUSTERNODE*, glm::mat4, double, double);
    void renderDistanceIndikator(glm::mat4);
    void renderText(vislib::StringA, glm::mat4, double, double, megamol::core::utility::SDFFont::Alignment);
    void connectNodes(
        HierarchicalClustering::CLUSTERNODE* node1, HierarchicalClustering::CLUSTERNODE* node2, glm::mat4);
    void setMinMax(std::vector<HierarchicalClustering::CLUSTERNODE*>*);
    void DrawCircle(glm::mat4 mvp, float cx, float cy, float r, int num_segments);
    std::vector<glm::uvec4> loadColorTable(void);

    std::vector<std::tuple<HierarchicalClustering::CLUSTERNODE*, RGBCOLOR*>*>* getNdiffrentColors(
        std::vector<HierarchicalClustering::CLUSTERNODE*>*);


    /**********************************************************************
     * callback stuff
     **********************************************************************/

    /** The input data slot. */
    core::CallerSlot clusterDataSlot;
    core::CallerSlot setPosition;

    vislib_gl::graphics::gl::GLSLShader passthroughShader;
    vislib_gl::graphics::gl::GLSLShader textureShader;
    std::unique_ptr<glowl::BufferObject> texBuffer;
    std::unique_ptr<glowl::BufferObject> geometrySSBO;
    GLuint texVa;
    GLuint dummyVa;

    /** The slot for requesting data. */
    core::CalleeSlot getPosition;
    bool newposition;
    std::vector<glm::uvec4> colortab;
};

} // namespace molsurfmapcluster
} // namespace megamol

#endif /*MOLSURFMAPCLUSTER_CLUSTERRENDERER_INCLUDED*/
