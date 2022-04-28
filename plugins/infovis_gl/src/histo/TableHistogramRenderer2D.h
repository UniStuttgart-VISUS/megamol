/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "BaseHistogramRenderer2D.h"

namespace megamol::infovis_gl {

class TableHistogramRenderer2D : public BaseHistogramRenderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TableHistogramRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Histogram renderer for generic tables.";
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
    TableHistogramRenderer2D();

    /**
     * Finalises an instance.
     */
    ~TableHistogramRenderer2D() override;

private:
    bool createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) override;

    void releaseImpl() override;

    bool handleCall(core_gl::view::CallRender2DGL& call) override;

    void updateSelection(SelectionMode selectionMode, int selectedComponent, int selectedBin) override;

    core::CallerSlot tableDataCallerSlot_;
    core::CallerSlot flagStorageReadCallerSlot_;
    core::CallerSlot flagStorageWriteCallerSlot_;

    std::unique_ptr<glowl::GLSLProgram> calcHistogramProgram_;
    std::unique_ptr<glowl::GLSLProgram> selectionProgram_;

    std::size_t numRows_;
    std::size_t currentTableDataHash_;
    unsigned int currentTableFrameId_;

    GLuint floatDataBuffer_ = 0;

    GLint selectionWorkgroupSize_[3];
    GLint maxWorkgroupCount_[3];
};

} // namespace megamol::infovis_gl
