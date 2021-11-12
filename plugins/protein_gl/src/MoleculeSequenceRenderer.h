#ifndef MOLECULESEQUENCERENDERER_H
#define MOLECULESEQUENCERENDERER_H

#include "mmcore/view/MouseFlags.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include "GlButton.h"
#include "GlWidgetLibrary.h"

namespace megamol {
namespace core {
    class CallerSlot;
}

namespace protein_gl {
    class MoleculeSequenceRenderer : public megamol::core_gl::view::Renderer2DModuleGL {
    public:
        static const char* ClassName(void) {
            return "MoleculeSequenceRenderer";
        }

        static const char* Description(void) {
            return "Offers access to molecule sequence data.";
        }

        static bool IsAvailable(void) {
            return true;
        }

        MoleculeSequenceRenderer(void);
        virtual ~MoleculeSequenceRenderer(void);

    protected:
        virtual bool create(void);
        virtual void release(void);

        virtual bool GetExtents(megamol::core_gl::view::CallRender2DGL& call);
        virtual bool Render(megamol::core_gl::view::CallRender2DGL& call);
        virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

    private: /* methods */
        void paintButton(float x, float y, float w, float h, float r, float g, float b, const char* text);

    private: /* fields */
        megamol::core::CallerSlot* dataCall;
        GlWidgetLibrary widgetLibrary;
    };

} // namespace protein_gl
} // namespace megamol
#endif
