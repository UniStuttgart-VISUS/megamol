/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "BaseHistogramRenderer2D.h"

namespace megamol::infovis_gl {

class TextureHistogramRenderer2D : public BaseHistogramRenderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TextureHistogramRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Histogram renderer for textures.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TextureHistogramRenderer2D();

    /**
     * Finalises an instance.
     */
    ~TextureHistogramRenderer2D() override;

private:
    bool createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) override;

    void releaseImpl() override;

    bool handleCall(mmstd_gl::CallRender2DGL& call) override;

    void updateSelection(SelectionMode selectionMode, int selectedComponent, int selectedBin) override;
    void applySelections();

    core::CallerSlot textureDataCallerSlot_;
    core::CallerSlot flagStorageReadCallerSlot_;
    core::CallerSlot flagStorageWriteCallerSlot_;

    std::unique_ptr<glowl::GLSLProgram> calcMinMaxLinesProgram_;
    std::unique_ptr<glowl::GLSLProgram> calcMinMaxAllProgram_;
    std::unique_ptr<glowl::GLSLProgram> calcHistogramProgram_;
    std::unique_ptr<glowl::GLSLProgram> selectionProgram_;

    std::shared_ptr<glowl::Texture2D> data_;

    GLuint minValueBuffer = 0;
    GLuint maxValueBuffer = 0;

    glm::ivec2 lastTexSize = glm::ivec2(0, 0);
    std::vector<std::pair<int, int>> selectedBinComps_;

    GLint selectionWorkgroupSize_[3];
    GLint maxWorkgroupCount_[3];
};

} // namespace megamol::infovis_gl
