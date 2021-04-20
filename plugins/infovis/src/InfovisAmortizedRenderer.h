#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/view/Renderer2DModule.h"
#include "vislib/graphics/gl/FramebufferObject.h"

#include "glm/matrix.hpp"
#include "glowl/FramebufferObject.hpp"
#include "mmcore/view/MouseFlags.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"

//#include "mmcore/view/MouseFlags.h"
#include "Renderer2D.h"

namespace megamol {
namespace infovis {
    class InfovisAmortizedRenderer : public Renderer2D {
    public:
        static inline const char* ClassName() {
            return "InfovisAmortizedRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char* Description(void) {
            return "Amortized Renderer.\n"
                   "Amortizes following Renderers to improve response time\n";
        }


        static inline bool IsAvailable(void) {
            return true;
        }

        /**
         * Initialises a new instance.
         */
        InfovisAmortizedRenderer(void);

        virtual ~InfovisAmortizedRenderer(void);

    protected:
        virtual bool create(void);

        virtual void release(void);

        virtual bool Render(core::view::CallRender2DGL& call);

        virtual bool GetExtents(core::view::CallRender2DGL& call);

        void setupBuffers();

        std::vector<glm::fvec3> calculateHammersley(int until, int ow, int oh);

        void makeShaders();

        void setupAccel(int approach, int ow, int oh, int ssLevel);

        void doReconstruction(int approach, int w, int h, int ssLevel);

        void resizeArrays(int approach, int w, int h, int ssLevel);

        bool OnMouseButton(
            core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

        bool OnMouseMove(double x, double y) override;

        bool OnMouseScroll(double dx, double dy) override;

        bool OnChar(unsigned int codePoint) override;

        bool OnKey(megamol::core::view::Key key, megamol::core::view::KeyAction action,
            megamol::core::view::Modifiers mods) override;

    private:
        // required Shaders for different kinds of reconstruction
        std::unique_ptr<vislib::graphics::gl::GLSLShader> amort_reconstruction_shdr_array[7];
        vislib::graphics::gl::ShaderSource vertex_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_src;

        enum AmortizationModes { MS_AR = 0, QUAD_AR, QUAD_AR_C, SS_AR, PARAMETER_AR, DEBUG_PLACEHOLDER, PUSH_AR };

        GLuint amortizedFboA = 0;
        GLuint amortizedMsaaFboA = 0;
        GLuint amortizedPushFBO = 0;
        GLuint msImageArray = 0;
        GLuint pushImage = 0;
        GLuint imageArrayA = 0;
        GLuint ssboMatrices = 0;
        GLuint imStoreArray = 0;
        GLuint imStoreA = 0;
        GLuint imStoreB = 0;
        int frametype = 0;
        int parity = 0;

        int oldApp = -1;
        int oldW = -1;
        int oldH = -1;
        int oldssLevel = -1;
        int oldaLevel = -1;
        int windowWidth = 1;
        int windowHeight = 1;

        GLint origFBO = 0;
        int framesNeeded = 1;
        GLfloat modelViewMatrix_column[16];
        GLfloat projMatrix_column[16];
        std::vector<glm::mat4> invMatrices;
        std::vector<glm::mat4> moveMatrices;
        std::vector<glm::fvec2> hammerPositions;
        std::vector<glm::vec3> camOffsets;
        glm::mat4 movePush;
        glm::mat4 lastPmvm;

        float backgroundColor[4];

        megamol::core::CallerSlot nextRendererSlot;
        core::param::ParamSlot halveRes;
        core::param::ParamSlot approachEnumSlot;
        core::param::ParamSlot superSamplingLevelSlot;
        core::param::ParamSlot amortLevel;
    };
} // namespace infovis
} // namespace megamol
