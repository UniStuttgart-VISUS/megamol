#ifndef MEGAMOL_INFOVIS_RESOLUTIONSCALINGRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_RESOLUTIONSCALINGRENDERER2D_H_INCLUDED

#include "BaseAmortizedRenderer2D.h"

#include <glm/matrix.hpp>
#include <glowl/glowl.h>

namespace megamol {
namespace infovis_gl {
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

    bool renderImpl(core_gl::view::CallRender2DGL& nextRendererCall,
        std::shared_ptr<core_gl::view::CallRender2DGL::FBO_TYPE> fbo, core::view::Camera cam) override;

    void updateSize(int a, int w, int h);

    void setupCamera(core::view::Camera& cam, int width, int height, int a);

    void reconstruct(std::shared_ptr<glowl::FramebufferObject>& fbo, int a);

private:
    core::param::ParamSlot amortLevelParam;
    core::param::ParamSlot debugParam;
    core::param::ParamSlot debugFloatParam;

    std::unique_ptr<glowl::GLSLProgram> shader_;
    std::unique_ptr<glowl::GLSLProgram> linshader_;
    std::shared_ptr<glowl::FramebufferObject> lowResFBO_;
    glowl::TextureLayout texLayout_;
    glowl::TextureLayout distTexLayout_;
    std::unique_ptr<glowl::Texture2D> texA_;
    std::unique_ptr<glowl::Texture2D> texB_;
    std::unique_ptr<glowl::Texture2D> distTexRead_;
    std::unique_ptr<glowl::Texture2D> distTexWrite_;

    int oldWidth_ = -1;
    int oldHeight_ = -1;
    int oldLevel_ = -1;

    int frameIdx_ = 0;
    int samplingSequencePosition_;
    std::vector<int> samplingSequence_;
    std::vector<glm::vec3> camOffsets_;
    glm::mat4 movePush_;
    glm::mat4 lastProjViewMx_;
};
} // namespace infovis_gl
} // namespace megamol

#endif // MEGAMOL_INFOVIS_RESOLUTIONSCALINGRENDERER2D_H_INCLUDED
