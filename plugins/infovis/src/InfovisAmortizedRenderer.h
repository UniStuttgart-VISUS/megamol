#ifndef MEGAMOL_INFOVIS_AMORTIZEDRENDERER_H_INCLUDED
#define MEGAMOL_INFOVIS_AMORTIZEDRENDERER_H_INCLUDED

#include <glm/matrix.hpp>
#include <glowl/glowl.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/view/MouseFlags.h"

#include "Renderer2D.h"

namespace megamol {
namespace infovis {

    class InfovisAmortizedRenderer : public Renderer2D {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char* ClassName() {
            return "InfovisAmortizedRenderer";
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
        InfovisAmortizedRenderer();

        /** Destructor. */
        ~InfovisAmortizedRenderer() override;

    protected:
        bool create() override;

        void release() override;

        bool GetExtents(core::view::CallRender2DGL& call) override;

        bool Render(core::view::CallRender2DGL& call) override;

        bool OnMouseButton(
            core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

        bool OnMouseMove(double x, double y) override;

        bool OnMouseScroll(double dx, double dy) override;

        bool OnChar(unsigned int codePoint) override;

        bool OnKey(megamol::core::view::Key key, megamol::core::view::KeyAction action,
            megamol::core::view::Modifiers mods) override;

        bool createShaders();

        bool createBuffers();

        void resizeArrays(int approach, int w, int h);

        void setupAccel(int approach, int ow, int oh, core::view::Camera* cam);

        void doReconstruction(int approach, int w, int h);

    private:
        enum AmortizationModes { MS_AR = 0, QUAD_AR, QUAD_AR_C, SS_AR, PARAMETER_AR, DEBUG_PLACEHOLDER, PUSH_AR };

        megamol::core::CallerSlot nextRendererSlot;
        core::param::ParamSlot enabledParam;
        core::param::ParamSlot approachParam;
        core::param::ParamSlot amortLevelParam;

        // required Shaders for different kinds of reconstruction
        std::unique_ptr<glowl::GLSLProgram> amort_reconstruction_shdr_array[7];

        GLuint amortizedFboA = 0;
        GLuint amortizedMsaaFboA = 0;

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

#endif // MEGAMOL_INFOVIS_AMORTIZEDRENDERER_H_INCLUDED
