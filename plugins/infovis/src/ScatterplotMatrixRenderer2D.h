#ifndef MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED

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


namespace megamol {
namespace infovis {

class ScatterplotMatrixRenderer2D : public core::view::Renderer2DModule {
public:
    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "ScatterplotMatrixRenderer2D";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Scatterplot matrix renderer for generic float tables.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return true;
    }

	/**
	* Initialises a new instance.
	*/
    ScatterplotMatrixRenderer2D();

	/**
	* Finalises an instance.
	*/
    virtual ~ScatterplotMatrixRenderer2D();

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
	struct MouseData {
        float x;
        float y;
        bool selects;
        bool inspects;
	};

    enum GeometryType {
        GEOMETRY_TYPE_POINT,
		GEOMETRY_TYPE_LINE
	};

      struct Selectors {
        size_t colorIdx;
        size_t labelIdx;
    } ;

      struct ShaderInfo {
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
    } ;

    typedef std::tuple<float, float, float, float> point_t;

    typedef std::tuple<int, int> viewport_t;

    typedef std::tuple<float, float> range_t;

    typedef std::vector<std::vector<float>> series_t;

    /**
    * The OpenGL Render callback.
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

	bool makeProgram(std::string prefix, vislib::graphics::gl::GLSLShader& program);

    bool isDirty(void) const;

    void resetDirty(void);

	bool validateData(void);

	void drawPoints(void);

    void drawLines(void);

    void drawXAxis(void);

    void drawYAxis(void);

    void drawToolTip(const float x, const float y, const std::string &text) const;

    size_t searchAndDispPointAttr(const float x, const float y);

    void lockSingle(GLsync &syncObj);

    void waitSingle(GLsync &syncObj);

    core::CallerSlot floatTableInSlot;

	core::CallerSlot transferFunctionInSlot;

	core::CallerSlot flagStorageInSlot;

    core::param::ParamSlot columnsParam;

    core::param::ParamSlot colorSelectorParam;

    core::param::ParamSlot labelSelectorParam;

    core::param::ParamSlot geometryTypeParam;

	core::param::ParamSlot geometryWidthParam;
	
	core::param::ParamSlot axisColorParam;
	
	core::param::ParamSlot axisWidthParam;

    core::param::ParamSlot axisTicksXParam;

    core::param::ParamSlot axisTicksYParam;

    core::param::ParamSlot scaleXParam;

	core::param::ParamSlot scaleYParam;

    core::param::ParamSlot alphaScalingParam;

    core::param::ParamSlot attenuateSubpixelParam;

    std::vector<DiagramSeriesCall::DiagramSeriesTuple> columnSelectors;

    Selectors columnIdxs;

    series_t series;

    viewport_t viewport;

    range_t yRange;

    ShaderInfo shaderInfo;

    const stdplugin::datatools::floattable::CallFloatTableData::ColumnInfo *columnInfos;

    vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> nvgTrans;

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> oglTrans;

    MouseData mouse;

    size_t dataHash;
};

} /* end namespace infovis */
} /* end namespace meagmol */

#endif /* MEGAMOL_INFOVIS_SCATTERPLOTRENDERER2D_H_INCLUDED */