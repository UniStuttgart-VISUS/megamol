/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glm/glm.hpp>
#include <glowl/glowl.h>

#include "BaseAmortization2D.h"

namespace megamol::mmstd_gl {

class ImageSpaceAmortization2D : public BaseAmortization2D {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "ImageSpaceAmortization2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Amortizes chained 2D renderers to improve response time\n";
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
    ImageSpaceAmortization2D();

    /** Destructor. */
    ~ImageSpaceAmortization2D() override;

protected:
    enum AmortMode {
        MODE_2D = 0,
        MODE_HORIZONTAL = 1,
        MODE_VERTICAL = 2,
        MODE_NONE = -1,
    };

    bool createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) override;

    void releaseImpl() override;

    bool renderImpl(CallRender2DGL& call, CallRender2DGL& nextRendererCall) override;

    void updateSize(AmortMode m, int a, int w, int h);

    void setupCamera(core::view::Camera& cam, int width, int height, AmortMode m, int a);

    void reconstruct(
        std::shared_ptr<glowl::FramebufferObject> const& fbo, core::view::Camera const& cam, AmortMode m, int a);

private:
    core::param::ParamSlot amortModeParam;
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

    AmortMode oldAmortMode_ = MODE_NONE;
    int oldAmortLevel_ = -1;
    int oldWidth_ = -1;
    int oldHeight_ = -1;

    int frameIdx_ = 0;
    int samplingSequencePosition_ = 0;
    std::vector<int> samplingSequence_;
    std::vector<glm::vec3> camOffsets_;
    glm::mat4 viewProjMx_;
    glm::mat4 lastViewProjMx_;
};
} // namespace megamol::mmstd_gl
