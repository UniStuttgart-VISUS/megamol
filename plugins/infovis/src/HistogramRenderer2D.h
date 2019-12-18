#ifndef MEGAMOL_INFOVIS_HISTOGRAMRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_HISTOGRAMRENDERER2D_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/FlagCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include "Renderer2D.h"

namespace megamol::infovis {

class HistogramRenderer2D : public Renderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "HistogramRenderer2D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Histogram renderer for generic tables."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

    /**
     * Initialises a new instance.
     */
    HistogramRenderer2D();

    /**
     * Finalises an instance.
     */
    ~HistogramRenderer2D() override;

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
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core::view::CallRender2D& call) override;

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender2D& call) override;

    bool handleCall(core::view::CallRender2D& call);

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

private:
    core::CallerSlot tableDataCallerSlot;
    core::CallerSlot transferFunctionCallerSlot;
    core::CallerSlot flagStorageCallerSlot;

    core::param::ParamSlot numberOfBinsParam;
    core::param::ParamSlot logPlotParam;

    size_t currentTableDataHash;
    unsigned int currentTableFrameId;
    core::FlagStorage::FlagVersionType currentFlagStorageVersion;

    size_t bins;
    size_t colCount;
    std::vector<float> colMinimums;
    std::vector<float> colMaximums;
    std::vector<std::string> colNames;
    std::vector<float> histogram;
    std::vector<float> selectedHistogram;
    size_t maxBinValue;

    vislib::graphics::gl::GLSLShader histogramProgram;
    vislib::graphics::gl::GLSLShader axesProgram;

    GLuint quadVertexArray;
    GLuint quadVertexBuffer;
    GLuint quadIndexBuffer;

    megamol::core::utility::SDFFont font;
};

} // namespace megamol::infovis

#endif // MEGAMOL_INFOVIS_HISTOGRAMRENDERER2D_H_INCLUDED
