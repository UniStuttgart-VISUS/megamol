#ifndef MEGAMOL_INFOVIS_NVGDIAGRAMRENDERER_H_INCLUDED
#define MEGAMOL_INFOVIS_NVGDIAGRAMRENDERER_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Matrix.h"
#include "vislib/graphics/gl/GLSLShader.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

#include "DiagramSeriesCall.h"

#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#include "nanoflann.hpp"
#pragma pop_macro("min")
#pragma pop_macro("max")

namespace megamol {
namespace infovis {

class NVGDiagramRenderer : public core::view::Renderer2DModule {
public:
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
        return "Offers diagram renderings based on NanoVG Part 2.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return true;
    }

    NVGDiagramRenderer();

    virtual ~NVGDiagramRenderer();
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
    virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags);
private:
    enum diagramType {
        DIAGRAM_TYPE_LINE_PLOT,
        DIAGRAM_TYPE_SCATTER_PLOT
    };

    typedef struct _selectorIdxs {
        size_t abcissaIdx;
        size_t colorIdx;
        size_t descIdx;
    } selectorIdxs_t;

    typedef struct _shaderInfo {
        vislib::graphics::gl::GLSLShader shader;
        GLuint ssboBindingPoint;
        GLuint bufferId;
        int numBuffers;
        GLsizeiptr bufSize;
        void *memMapPtr;
        GLuint bufferCreationBits;
        GLuint bufferMappingBits;
        std::vector<GLsync> fences;
        unsigned int currBuf;
    } shaderInfo_t;

    typedef struct _nvgRenderInfo {
        float fontSize;
    } nvgRenderInfo_t;

    typedef std::tuple<float, float, float, float> point_t;

    typedef std::tuple<int, int> viewport_t;

    typedef std::tuple<float, float> range_t;

    typedef std::tuple<float, float, size_t> call_t;

    typedef std::vector<float> abcissa_t;

    typedef std::vector<std::vector<float>> series_t;

    //// https://github.com/jlblancoc/nanoflann/blob/master/examples/KDTreeVectorOfVectorsAdaptor.h
    //template <int DIM = -1, class Distance = nanoflann::metric_L2>
    //struct KDTreeAdaptor {
    //    typedef KDTreeAdaptor<DIM, Distance> self_t;
    //    typedef typename Distance::template traits<float, self_t>::distance_t metric_t;
    //    typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, size_t> index_t;

    //    series_t &series;

    //    size_t relevant;

    //    index_t *index;

    //    KDTreeAdaptor(series_t &series, size_t idx, int leaf_max_size = 10) : series(series), relevant(idx) {
    //        index = new index_t(DIM, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
    //        index->buildIndex();
    //    }

    //    ~KDTreeAdaptor(void) {
    //        delete index;
    //    }

    //    const self_t &derived() const {
    //        return *this;
    //    }

    //    self_t &derived() {
    //        return *this;
    //    }

    //    inline size_t kdtree_get_point_count(void) const {
    //        return series[relevant].size() / 4;
    //    }

    //    inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t size) const {
    //        float s = 0;
    //        for (size_t i = 0; i<size; i++) {
    //            const float d = p1[i] - series[relevant][idx_p2 * 4 + i];
    //            s += d*d;
    //        }
    //        return s;
    //    }

    //    inline float kdtree_get_pt(const size_t idx, int dim) const {
    //        return series[relevant][idx * 4 + dim];
    //    }

    //    template <class BBOX>
    //    bool kdtree_get_bbox(BBOX & /*bb*/) const {
    //        return false;
    //    }
    //};

    //typedef KDTreeAdaptor<2> kdTree_t;

    /**
    * The Open GL Render callback.
    *
    * @param call The calling call.
    * @return The return value of the function.
    */
    virtual bool Render(core::view::CallRender2D& call);

    /**
    * The get extents callback. The module should set the members of
    * 'call' to tell the caller the extents of its data (bounding boxes
    * and times).
    *
    * @param call The calling call.
    *
    * @return The return value of the function.
    */
    virtual bool GetExtents(core::view::CallRender2D& call);

    void seriesInsertionCB(const DiagramSeriesCall::DiagramSeriesTuple &tuple);

    bool assertData(void);

    bool isAnythingDirty(void) const;

    void resetDirtyFlag(void);

    bool updateColumnSelectors(void);

    void drawLinePlot(void); //< NanoVG-based

    void drawScatterPlot(void); //< Shader-based

    void drawXAxis(void);

    void drawYAxis(void);

    /*void drawCallStack(void);

    void drawPyjama(void);

    void drawToolTip(const float x, const float y, const std::string &text) const;

    size_t searchAndDispPointAttr(const float x, const float y);*/

    void lockSingle(GLsync &syncObj);

    void waitSingle(GLsync &syncObj);

    core::CallerSlot floatTableInSlot;

    core::CallerSlot getColumnSelectorsSlot;

    core::CallerSlot getTransFuncSlot;

    //core::CallerSlot getCallTraceSlot;

    //core::CallerSlot getPointInfoSlot;

    core::param::ParamSlot abcissaSelectParam;

    core::param::ParamSlot colorSelectParam;

    core::param::ParamSlot descSelectParam;

    core::param::ParamSlot diagramTypeParam;

    core::param::ParamSlot lineWidthParam;

    core::param::ParamSlot numXTicksParam;

    core::param::ParamSlot numYTicksParam;

    core::param::ParamSlot aspectParam;

    core::param::ParamSlot alphaScalingParam;

    core::param::ParamSlot attenuateSubpixelParam;

    core::param::ParamSlot pointSizeParam;

    core::param::ParamSlot yScalingParam;

    core::param::ParamSlot textThresholdParam;

    core::param::ParamSlot pyjamaModeParam;

    DiagramSeriesCall::fpSeriesInsertionCB fpSeriesInsertionCB;

    std::vector<DiagramSeriesCall::DiagramSeriesTuple> columnSelectors;

    std::vector<bool> selectedSeries;

    selectorIdxs_t columnIdxs;

    abcissa_t abcissa;

    series_t series;

    viewport_t viewport;

    range_t yRange;

    nvgRenderInfo_t nvgRenderInfo;

    shaderInfo_t shaderInfo;

    std::vector<std::vector<call_t>> callStack;

    const stdplugin::datatools::floattable::CallFloatTableData::ColumnInfo *columnInfos;

    vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> nvgTrans;

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> oglTrans;

    void *nvgCtx;

    //kdTree_t *tree;

    bool mouseRightPressed;

    float mouseX;

    float mouseY;

    size_t dataHash;
}; /* end class NVGDiagramRenderer */

} /* end namespace infovis */
} /* end namespace meagmol */

#endif // end ifndef MEGAMOL_INFOVIS_NVGDIAGRAMRENDERER_H_INCLUDED