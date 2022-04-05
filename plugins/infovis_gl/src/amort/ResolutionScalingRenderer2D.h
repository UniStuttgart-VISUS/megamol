/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glm/glm.hpp>
#include <glowl/glowl.h>

#include "BaseAmortizedRenderer2D.h"

namespace megamol::infovis_gl {

class ResolutionScalingRenderer2D : public BaseAmortizedRenderer2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "ResolutionScalingRenderer2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Amortizes chained InfoVis renderers to improve response time\n";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /** Constructor. */
    ResolutionScalingRenderer2D();

    /** Destructor. */
    ~ResolutionScalingRenderer2D() override;

protected:
    bool createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) override;

    void releaseImpl() override;

    bool renderImpl(core_gl::view::CallRender2DGL& call, core_gl::view::CallRender2DGL& nextRendererCall) override;

    void updateSize(int a, int w, int h);

    void setupCamera(core::view::Camera& cam, int width, int height, int a);

    void reconstruct(std::shared_ptr<glowl::FramebufferObject> const& fbo, core::view::Camera const& cam, int a);

private:
    core::param::ParamSlot amortLevelParam;
    core::param::ParamSlot skipInterpolationParam;

    std::unique_ptr<glowl::GLSLProgram> shader_;
    std::shared_ptr<glowl::FramebufferObject> lowResFBO_;
    glowl::TextureLayout texLayout_;
    glowl::TextureLayout distTexLayout_;
    std::unique_ptr<glowl::Texture2D> texRead_;
    std::unique_ptr<glowl::Texture2D> texWrite_;
    std::unique_ptr<glowl::Texture2D> distTexRead_;
    std::unique_ptr<glowl::Texture2D> distTexWrite_;

    int oldWidth_ = -1;
    int oldHeight_ = -1;
    int oldAmortLevel_ = -1;

    int frameIdx_ = 0;
    int samplingSequencePosition_;
    std::vector<int> samplingSequence_;
    std::vector<glm::vec3> camOffsets_;
    glm::mat4 viewProjMx_;
    glm::mat4 lastViewProjMx_;
};
} // namespace megamol::infovis_gl
