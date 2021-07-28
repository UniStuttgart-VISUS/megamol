/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_INFOVIS_BASEHISTOGRAMRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_BASEHISTOGRAMRENDERER2D_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/SDFFont.h"
#include "mmcore/utility/ShaderFactory.h"
#include "mmcore/view/CallGetTransferFunction.h"

#include "Renderer2D.h"

namespace megamol::infovis {

class BaseHistogramRenderer2D : public Renderer2D {
public:
    BaseHistogramRenderer2D();

    ~BaseHistogramRenderer2D() override = default;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() final;

    /**
     * Implementation of 'Release'.
     */
    void release() final;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core::view::CallRender2DGL& call) final;

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender2DGL& call) final;

    bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) final;

    bool OnMouseMove(double x, double y) final;

    void bindCommon(std::unique_ptr<glowl::GLSLProgram>& program);

    bool binsChanged() const {
        return numBins_ != static_cast<std::size_t>(binsParam_.Param<core::param::IntParam>()->Value());
    }

    std::size_t numBins() const {
        return numBins_;
    }

    std::size_t numCols() const {
        return numCols_;
    }

    void setColHeaders(
        std::vector<std::string> colNames, std::vector<float> colMinimums, std::vector<float> colMaximums);

    void resizeAndClearHistoBuffers();

    virtual bool createHistoImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) = 0;

    virtual void releaseHistoImpl() = 0;

    virtual bool handleCall(core::view::CallRender2DGL& call) = 0;

    virtual void updateSelection(int selectionMode, int selectedCol, int selectedBin) = 0;

private:
    core::CallerSlot transferFunctionCallerSlot_;

    core::param::ParamSlot binsParam_;
    core::param::ParamSlot logPlotParam_;
    core::param::ParamSlot selectionColorParam_;

    std::size_t numBins_;
    std::size_t numCols_;
    std::vector<std::string> colNames_;
    std::vector<float> colMinimums_;
    std::vector<float> colMaximums_;

    std::unique_ptr<glowl::GLSLProgram> drawHistogramProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawAxesProgram_;
    std::unique_ptr<glowl::GLSLProgram> maxBinProgram_;

    GLuint histogramBuffer_ = 0;
    GLuint selectedHistogramBuffer_ = 0;
    GLuint minBuffer_ = 0;
    GLuint maxBuffer_ = 0;

    std::size_t maxBinValue_;
    bool needMaxBinValueUpdate_;

    megamol::core::utility::SDFFont font_;

    float mouseX_;
    float mouseY_;

    bool needSelectionUpdate_;
    int selectionMode_;
    int selectedCol_;
    int selectedBin_;
};

} // namespace megamol::infovis

#endif // MEGAMOL_INFOVIS_BASEHISTOGRAMRENDERER2D_H_INCLUDED
