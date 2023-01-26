/*
 * Diagram2DRenderer.h
 *
 * Author: Guido Reina
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "protein_calls/DiagramCall.h"
#include "protein_calls/IntSelectionCall.h"
#include "vislib_gl/graphics/gl/OpenGLTexture2D.h"
#include "vislib_gl/graphics/gl/OutlineFont.h"

namespace megamol {
namespace protein_gl {

class DiagramRenderer : public megamol::mmstd_gl::Renderer2DModuleGL {
public:
    enum DiagramTypes {
        DIAGRAM_TYPE_LINE = 0,
        DIAGRAM_TYPE_LINE_STACKED = 1,
        DIAGRAM_TYPE_LINE_STACKED_NORMALIZED = 2,
        DIAGRAM_TYPE_COLUMN = 4,
        DIAGRAM_TYPE_COLUMN_STACKED = 8,
        DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED = 16
    };

    enum DiagramStyles { DIAGRAM_STYLE_WIRE = 0, DIAGRAM_STYLE_FILLED = 1 };

    enum XAxisTypes { DIAGRAM_XAXIS_FLOAT = 0, DIAGRAM_XAXIS_INTEGRAL = 1, DIAGRAM_XAXIS_CATEGORICAL = 3 };

    enum MarkerVisibility {
        DIAGRAM_MARKERS_SHOW_NONE = 0,
        DIAGRAM_MARKERS_SHOW_SELECTED = 1,
        DIAGRAM_MARKERS_SHOW_ALL = 2
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "DiagramRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers better diagram renderings.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** ctor */
    DiagramRenderer();

    /** dtor */
    ~DiagramRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags) override;

private:
    /**********************************************************************
     * 'render'-functions
     **********************************************************************/

    bool CalcExtents();

    void drawYAxis();

    /**
     * sets the xTickOff!
     */
    void drawXAxis(XAxisTypes xType);

    void drawLegend();

    void drawLineDiagram();
    void drawColumnDiagram();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    VISLIB_FORCEINLINE bool isCategoricalMappable(const protein_calls::DiagramCall::DiagramMappable* dm) const {
        return (dm->IsCategoricalAbscissa(0));
    }

    bool LoadIcon(vislib::StringA filename, int ID);

    void getBarXY(int series, int index, int type, float* x, float* y);

    bool onCrosshairToggleButton(megamol::core::param::ParamSlot& p);

    bool onShowAllButton(megamol::core::param::ParamSlot& p);

    bool onHideAllButton(megamol::core::param::ParamSlot& p);

    void prepareData(bool stack, bool normalize, bool drawCategorical);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**********************************************************************
     * variables
     **********************************************************************/

    /** caller slot */
    core::CallerSlot dataCallerSlot;

    /** caller slot */
    core::CallerSlot selectionCallerSlot;

    /** caller slot */
    core::CallerSlot hiddenCallerSlot;

    ///** clear diagram parameter */
    // megamol::core::param::ParamSlot clearDiagramParam;

    /** the mouse position */
    vislib::math::Vector<float, 3> mousePos;

    vislib_gl::graphics::gl::OutlineFont theFont;

    vislib::Pair<float, float> xRange;
    vislib::Pair<float, float> yRange;

    megamol::core::param::ParamSlot diagramTypeParam;

    megamol::core::param::ParamSlot diagramStyleParam;

    megamol::core::param::ParamSlot numXTicksParam;

    megamol::core::param::ParamSlot numYTicksParam;

    megamol::core::param::ParamSlot drawYLogParam;

    megamol::core::param::ParamSlot foregroundColorParam;

    megamol::core::param::ParamSlot drawCategoricalParam;

    megamol::core::param::ParamSlot aspectRatioParam;

    megamol::core::param::ParamSlot autoAspectParam;

    megamol::core::param::ParamSlot lineWidthParam;

    vislib::math::Vector<float, 4> fgColor;

    const vislib::math::Vector<float, 4> unselectedColor;

    const float decorationDepth;

    // warning: Z encodes the previous y-coordinate, Y the actual value (draw + click ranges between values!)
    vislib::PtrArray<vislib::PtrArray<vislib::math::Point<float, 3>>>* preparedData;

    vislib::Array<protein_calls::DiagramCall::DiagramSeries*> preparedSeries;

    vislib::Array<vislib::StringA> categories;

    vislib::Array<float> xValues;

    vislib::Array<vislib::Array<int>> localXIndexToGlobal;

    vislib::Array<vislib::Pair<int, vislib::SmartPtr<vislib_gl::graphics::gl::OpenGLTexture2D>>> markerTextures;

    float xAxis;

    float yAxis;

    float xTickOff;

    float barWidth;

    float fontSize;

    float legendOffset;

    float legendWidth;

    float legendHeight;

    float legendMargin;

    const float barWidthRatio;

    megamol::protein_calls::DiagramCall::DiagramSeries* selectedSeries;

    // vislib::Array<int> selectedSeriesIndices;

    megamol::protein_calls::DiagramCall* diagram;

    megamol::protein_calls::IntSelectionCall* selectionCall;

    megamol::protein_calls::IntSelectionCall* hiddenCall;

    const megamol::protein_calls::DiagramCall::DiagramMarker* hoveredMarker;

    int hoveredSeries;

    megamol::core::param::ParamSlot showCrosshairToggleParam;

    megamol::core::param::ParamSlot showCrosshairParam;

    megamol::core::param::ParamSlot showGuidesParam;

    megamol::core::param::ParamSlot showMarkersParam;

    megamol::core::param::ParamSlot showAllParam;

    megamol::core::param::ParamSlot hideAllParam;

    vislib::math::Point<float, 2> hoverPoint;

    vislib::Array<bool> seriesVisible;

    // EVIL EVIL HACK HACK
    void dump();
};

} // namespace protein_gl
} /* end namespace megamol */
