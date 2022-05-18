/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/view/CallGetTransferFunctionGL.h"

#include "Renderer2D.h"

namespace megamol::infovis_gl {

class BaseHistogramRenderer2D : public Renderer2D {
public:
    BaseHistogramRenderer2D();

    ~BaseHistogramRenderer2D() override = default;

protected:
    enum class SelectionMode {
        PICK = 0,
        APPEND = 1,
        REMOVE = 2,
    };

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
    bool GetExtents(core_gl::view::CallRender2DGL& call) final;

    /**
     * The OpenGL Render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core_gl::view::CallRender2DGL& call) final;

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

    std::size_t numComponents() const {
        return numComponents_;
    }

    void setComponentHeaders(std::vector<std::string> names, std::vector<float> minimums, std::vector<float> maximums);

    void resetHistogramBuffers();

    virtual bool createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) = 0;

    virtual void releaseImpl() = 0;

    virtual bool handleCall(core_gl::view::CallRender2DGL& call) = 0;

    virtual void updateSelection(SelectionMode selectionMode, int selectedComponent, int selectedBin) = 0;

private:
    core::CallerSlot transferFunctionCallerSlot_;

    core::param::ParamSlot binsParam_;
    core::param::ParamSlot logPlotParam_;
    core::param::ParamSlot selectionColorParam_;

    std::size_t numBins_;
    std::size_t numComponents_;
    std::vector<std::string> componentNames_;
    std::vector<float> componentMinimums_;
    std::vector<float> componentMaximums_;

    std::unique_ptr<glowl::GLSLProgram> drawHistogramProgram_;
    std::unique_ptr<glowl::GLSLProgram> drawAxesProgram_;
    std::unique_ptr<glowl::GLSLProgram> calcMaxBinProgram_;

    GLuint histogramBuffer_ = 0;
    GLuint selectedHistogramBuffer_ = 0;
    GLuint componentMinBuffer_ = 0;
    GLuint componentMaxBuffer_ = 0;

    std::size_t maxBinValue_;
    bool needMaxBinValueUpdate_;

    megamol::core::utility::SDFFont font_;

    float mouseX_;
    float mouseY_;
    glm::ivec2 viewRes_;
    core::view::Camera camera_;

    bool needSelectionUpdate_;
    SelectionMode selectionMode_;
    int selectedComponent_;
    int selectedBin_;
};

} // namespace megamol::infovis_gl
