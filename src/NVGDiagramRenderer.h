/*
 * NVGDiagramRenderer.h
 *
 * Author: Guido Reina, ...
 * Copyright (C) 2012-2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOL_INFOVIS_NVGDIAGRAMRENDERER_H_INCLUDED
#define MEGAMOL_INFOVIS_NVGDIAGRAMRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <functional>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer2DModule.h"

#include "vislib/math/Matrix.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/PtrArray.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

#include "DiagramSeriesCall.h"


//#include "protein_calls/DiagramCall.h"
//#include "protein_calls/IntSelectionCall.h"

namespace megamol {
namespace infovis {
class DiagramSeriesCall;

class NVGDiagramRenderer : public core::view::Renderer2DModule {
public:

    //typedef void (*fpSeriesInsertionCB)(const DiagramSeriesCall::DiagramSeriesTuple &tuple);

    //typedef std::function<void(const DiagramSeriesCall::DiagramSeriesTuple &tuple)> fpSeriesInsertionCB;

    enum DiagramTypes {
        DIAGRAM_TYPE_LINE = 0,
        DIAGRAM_TYPE_LINE_STACKED = 1,
        DIAGRAM_TYPE_LINE_STACKED_NORMALIZED = 2,
        DIAGRAM_TYPE_COLUMN = 4,
        DIAGRAM_TYPE_COLUMN_STACKED = 8,
        DIAGRAM_TYPE_COLUMN_STACKED_NORMALIZED = 16,
        DIAGRAM_TYPE_POINT_SPLATS = 32
    };

    enum DiagramStyles {
        DIAGRAM_STYLE_WIRE = 0,
        DIAGRAM_STYLE_FILLED = 1
    };

    enum XAxisTypes {
        DIAGRAM_XAXIS_FLOAT = 0,
        DIAGRAM_XAXIS_INTEGRAL = 1,
        DIAGRAM_XAXIS_CATEGORICAL = 3
    };

    enum YAxisTypes {
        DIAGRAM_YAXIS_FLOAT = 0,
        DIAGRAM_YAXIS_CLUSTERED = 1
    };

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
    static const char *ClassName(void) {
        return "NVGDiagramRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers diagram renderings based on NanoVG.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    NVGDiagramRenderer(void);

    void lockSingle(GLsync & syncObj);

    void waitSingle(GLsync & syncObj);

    /** dtor */
    ~NVGDiagramRenderer(void);

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
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

private:

    std::vector<megamol::infovis::DiagramSeriesCall::DiagramSeriesTuple> columnSelectors;

    void seriesInsertionCB(const megamol::infovis::DiagramSeriesCall::DiagramSeriesTuple &tuple);

    core::CallerSlot getSelectorsSlot;

    bool updateColumnSelectors(void);

    megamol::stdplugin::datatools::floattable::CallFloatTableData *floatTable;

    bool isAnythingDirty();

    void resetDirtyFlags();

    size_t myHash;

    size_t inputHash;

    bool assertData(megamol::stdplugin::datatools::floattable::CallFloatTableData * ft);

    core::param::ParamSlot abcissaSelectorSlot;

    size_t abcissaIdx;

    DiagramSeriesCall::fpSeriesInsertionCB fpsicb;

    void *nvgCtxt;

    vislib::math::Point<uint32_t, 2> screenSpaceMidPoint;

    vislib::math::Dimension<uint32_t, 2> screenSpaceCanvasSize;

    vislib::math::Dimension<uint32_t, 2> screenSpaceDiagramSize;

    void defineLayout(float w, float h);

    core::param::ParamSlot screenSpaceCanvasOffsetParam;

    int nvgFontSans;

    float scaleX, scaleY;

    vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> transform;
    vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> transformT;

    float sWidth;
    float sHeight;

    std::vector<bool> selected;

    std::vector<vislib::math::Rectangle<float>> bndBtns;

    void drawPointSplats(float w, float h);

    GLuint theSingleBuffer;
    unsigned int currBuf;
    GLsizeiptr bufSize;
    int numBuffers;
    void *theSingleMappedMem;
    GLuint singleBufferCreationBits;
    GLuint singleBufferMappingBits;
    std::vector<GLsync> fences;
    std::shared_ptr<vislib::graphics::gl::GLSLShader> newShader;

    std::vector<float> pointData;

    core::param::ParamSlot alphaScalingParam;
    core::param::ParamSlot attenuateSubpixelParam;

    core::view::CallRender2D *callR2D;

    void clusterYRange(void);

    void showToolTip(const float x, const float y, const std::string &symbol, const std::string &module, const std::string &file, const size_t &line, const size_t &memaddr, const size_t &memsize) const;

    /**********************************************************************
     * 'render'-functions
     **********************************************************************/

    bool CalcExtents();

    void drawYAxis();

    /**
     * sets the xTickOff!
     */
    void drawXAxis(XAxisTypes xType);

    void drawLegend(float w, float h);

    void drawLineDiagram(float w, float h);
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
    virtual bool GetExtents(megamol::core::view::CallRender2D& call);

    /*VISLIB_FORCEINLINE bool isCategoricalMappable(const protein_calls::DiagramCall::DiagramMappable *dm) const {
        return (dm->IsCategoricalAbscissa(0));
    }*/

    bool LoadIcon(vislib::StringA filename, int ID);

    void getBarXY(int series, int index, int type, float *x, float *y);

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
    virtual bool Render(megamol::core::view::CallRender2D& call);

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
    //megamol::core::param::ParamSlot clearDiagramParam;

    /** the mouse position */
    vislib::math::Vector<float, 3> mousePos;

    vislib::graphics::gl::OutlineFont theFont;

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
    vislib::PtrArray<vislib::PtrArray<vislib::math::Point<float, 3> > > *preparedData;

    vislib::Array<const megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnInfo *> preparedSeries;

    vislib::Array<vislib::StringA> categories;

    vislib::Array<float> xValues;

    vislib::Array<vislib::Array<int> > localXIndexToGlobal;

    vislib::Array<vislib::Pair<int, vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D> > > markerTextures;

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

    //megamol::protein_calls::DiagramCall::DiagramSeries *selectedSeries;

    //vislib::Array<int> selectedSeriesIndices;

    //megamol::protein_calls::DiagramCall *diagram;

    //megamol::protein_calls::IntSelectionCall *selectionCall;

    //megamol::protein_calls::IntSelectionCall *hiddenCall;

    //const megamol::protein_calls::DiagramCall::DiagramMarker *hoveredMarker;

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
    //void dump();
};

} /* end namespace infovis */
} /* end namespace megamol */

#endif // MEGAMOL_INFOVIS_NVGDIAGRAMRENDERER_H_INCLUDED
