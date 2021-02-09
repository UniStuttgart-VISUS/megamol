#pragma once
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/AbstractRenderingView.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "glm/gtc/functions.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "E://Thesis/MegamolBranch/plugins/infovis/src/Renderer2D.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/InputCall.h"
#include "mmcore/view/MouseFlags.h"
#include <AbstractInputScope.h>

namespace megamol {
namespace core {
    namespace view {

        /*
         * Forward declaration of incoming render calls
         */
        class CallRenderView;

        class AmortizedRenderer : public AbstractRenderingView {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName(void) {
                return "AmortizedRenderer";
            }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable(void) {
                return true;
            }

            /**
             * Initializes new Instance
             */
            AmortizedRenderer(void);

            /**
             * Finalizes the Instance
             */
            virtual ~AmortizedRenderer(void);


            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description(void) {
                return "Amortized Renderer.\n"
                       "The following Renderers will render at reduced resolution.\n"
                       "Later Reconstruction will reassamble full resolution image";
            }

            /**
             * The render callback.
             *
             * @param call The calling call.
             *
             * @return The return value of the function.
             */
            virtual bool Render(core::view::CallRender2D& call);

            virtual void Render(const mmcRenderViewContext& context);

            virtual bool GetExtents(core::view::CallRender2D& call);

            //Desperation ahead

            virtual float DefaultTime(double instTime) const {
                return 0;
            }

            virtual unsigned int GetCameraSyncNumber(void) const;

            virtual void SerialiseCamera(vislib::Serialiser& serialiser) const;

            virtual void DeserialiseCamera(vislib::Serialiser& serialiser);

            virtual void ResetView(void);

            virtual void Resize(unsigned int width, unsigned int height);

            virtual bool OnRenderView(Call& call);

            virtual void UpdateFreeze(bool freeze);

            virtual bool OnKey(Key key, KeyAction action, Modifiers mods) override;

            virtual bool OnChar(unsigned int codePoint) override;

            virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) override;

            virtual bool OnMouseMove(double x, double y) override;

            virtual bool OnMouseScroll(double dx, double dy) override;

        protected:  

            void setupBuffers();

            std::vector<glm::fvec2> calculateHammersley(int until);

            void setupAccel(int approach, int ow, int oh, int ssLevel);

            void doReconstruction(int approach, int w, int h, int ssLevel);


        private:
            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create(void);

            /**
             * Implementation of 'Release'.
             */
            virtual void release(void);

            // required Shaders for different kinds of reconstruction
            std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction0_shdr;
            std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction1_shdr;
            std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction2_shdr;
            std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction3_shdr;
            std::unique_ptr<vislib::graphics::gl::GLSLShader> pc_reconstruction3h_shdr;
            vislib::graphics::gl::ShaderSource vertex_shader_src;
            vislib::graphics::gl::ShaderSource fragment_shader_src;

            CalleeSlot inputCall;

            GLuint amortizedFboA;
            GLuint amortizedFboB;
            GLuint amortizedMsaaFboA;
            GLuint amortizedMsaaFboB;
            GLuint msImageStorageA;
            GLuint msImageArray;

            GLuint imageArrayA;
            GLuint imageArrayB;
            GLuint ssboMatrices;
            int frametype;

            GLint origFBO = 0;
            int framesNeeded = 1;
            GLfloat modelViewMatrix_column[16];
            GLfloat projMatrix_column[16];
            std::vector<glm::mat4> invMatrices;
            std::vector<glm::mat4> moveMatrices;
            std::vector<glm::fvec2> hammerPositions;

            megamol::core::CallerSlot nextRendererSlot;
            core::param::ParamSlot halveRes;
            core::param::ParamSlot approachSlot;
            core::param::ParamSlot superSamplingLevelSlot;
        };
    }// namespace view
} // namespace infovis
} // namespace megamol
