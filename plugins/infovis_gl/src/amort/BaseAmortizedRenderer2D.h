#ifndef MEGAMOL_INFOVIS_AMORTIZEDRENDERER_H_INCLUDED
#define MEGAMOL_INFOVIS_AMORTIZEDRENDERER_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/view/CallRender2DGL.h"

#include "Renderer2D.h"

namespace megamol {
namespace infovis_gl {

    class BaseAmortizedRenderer2D : public Renderer2D {
    public:

        BaseAmortizedRenderer2D();

        ~BaseAmortizedRenderer2D() override = default;

    protected:
        bool create() override;

        void release() override;

        bool GetExtents(core_gl::view::CallRender2DGL& call) final;

        bool Render(core_gl::view::CallRender2DGL& call) final;

        bool OnMouseButton(
            core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) final;

        bool OnMouseMove(double x, double y) final;

        bool OnMouseScroll(double dx, double dy) final;

        bool OnChar(unsigned int codePoint) final;

        bool OnKey(megamol::core::view::Key key, megamol::core::view::KeyAction action,
            megamol::core::view::Modifiers mods) final;

        virtual bool createImpl(const msf::ShaderFactoryOptionsOpenGL& shaderOptions) = 0;

        virtual void releaseImpl() = 0;

        virtual bool renderImpl(core_gl::view::CallRender2DGL& nextRendererCall,
            std::shared_ptr<core_gl::view::CallRender2DGL::FBO_TYPE> fbo, core::view::Camera cam) = 0;

    private:
        megamol::core::CallerSlot nextRendererSlot;
        core::param::ParamSlot enabledParam;
    };
} // namespace infovis_gl
} // namespace megamol

#endif // MEGAMOL_INFOVIS_AMORTIZEDRENDERER_H_INCLUDED
