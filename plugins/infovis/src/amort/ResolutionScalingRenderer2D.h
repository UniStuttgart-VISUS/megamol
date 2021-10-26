#ifndef MEGAMOL_INFOVIS_RESOLUTIONSCALINGRENDERER2D_H_INCLUDED
#define MEGAMOL_INFOVIS_RESOLUTIONSCALINGRENDERER2D_H_INCLUDED

#include "BaseAmortizedRenderer2D.h"

#include <glm/matrix.hpp>
#include <glowl/glowl.h>

namespace megamol {
namespace infovis {
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

        bool renderImpl(core::view::CallRender2DGL& targetRenderer,
            std::shared_ptr<core::view::CallRender2DGL::FBO_TYPE> fbo, core::view::Camera cam) override;

        void resizeArrays(int w, int h);

        void setupAccel(int ow, int oh, core::view::Camera* cam);

        void doReconstruction(int w, int h);

    private:
        core::param::ParamSlot amortLevelParam;

        // required Shaders for different kinds of reconstruction
        std::unique_ptr<glowl::GLSLProgram> amort_reconstruction_shdr_array[7];

        std::shared_ptr<glowl::FramebufferObject> glowlFBO;
        std::unique_ptr<glowl::Texture2D> texA;
        std::unique_ptr<glowl::Texture2D> texB;
        glowl::TextureLayout texstore_layout;

        int frametype = 0;
        int parity = 0;

        int oldApp = -1;
        int oldW = -1;
        int oldH = -1;
        int oldaLevel = -1;
        int windowWidth = 1;
        int windowHeight = 1;

        std::shared_ptr<glowl::FramebufferObject> fbo = nullptr;

        int framesNeeded = 1;
        glm::mat4 projMatrix;
        glm::mat4 mvMatrix;
        std::vector<glm::mat4> invMatrices;
        std::vector<glm::mat4> moveMatrices;
        std::vector<glm::vec3> camOffsets;
        glm::mat4 movePush;
        glm::mat4 lastPmvm;
    };
} // namespace infovis
} // namespace megamol

#endif // MEGAMOL_INFOVIS_RESOLUTIONSCALINGRENDERER2D_H_INCLUDED
