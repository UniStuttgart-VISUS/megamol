#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/Renderer2DModule.h"
#include "vislib/graphics/gl/FramebufferObject.h"

#include "glowl/FramebufferObject.hpp"
#include "glm/matrix.hpp"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"

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

        virtual bool Render(core::view::CallRender2D& call);

        virtual bool GetExtents(core::view::CallRender2D& call);

        void setupBuffers();

        std::vector<glm::fvec2> calculateHammersley(int until);

        void makeShaders();

        void setupAccel(int approach, int ow, int oh, int ssLevel);

        void doReconstruction(int approach, int w, int h, int ssLevel);

    private:
        // required Shaders for different kinds of reconstruction
        std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction0_shdr;
        std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction1_shdr;
        std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction2_shdr;
        std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction3_shdr;
        std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction3h_shdr;
        vislib::graphics::gl::ShaderSource vertex_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_src;

        GLuint amortizedFboA = 0;
        GLuint amortizedFboB = 0;
        GLuint amortizedMsaaFboA = 0;
        GLuint amortizedMsaaFboB = 0;
        GLuint msImageStorageA = 0;
        GLuint msImageArray = 0;

        GLuint imageArrayA = 0;
        GLuint imageArrayB = 0;
        GLuint ssboMatrices = 0;
        int frametype = 0;

        GLint origFBO = 0;
        int framesNeeded = 1;
        GLfloat modelViewMatrix_column[16];
        GLfloat projMatrix_column[16];
        std::vector<glm::mat4> invMatrices;
        std::vector<glm::mat4> moveMatrices;
        std::vector<glm::fvec2> hammerPositions;

         float backgroundColor[4];

        megamol::core::CallerSlot nextRendererSlot;
        core::param::ParamSlot halveRes;
        core::param::ParamSlot approachSlot;
        core::param::ParamSlot superSamplingLevelSlot;
    };
} // namespace infovis
} // namespace megamol
