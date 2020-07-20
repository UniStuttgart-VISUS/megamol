#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule_2.h"
#include "vislib/graphics/gl/GLSLShader.h"

#include "thermodyn/BoxDataCall.h"


namespace megamol {
namespace thermodyn {
namespace rendering {

class BoxRenderer : public core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "BoxRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renderer for box glyphs."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef _WIN32
#    if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#    endif // DEBUG || _DEBUG
#endif     // _WIN32
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
    }

    /** Ctor. */
    BoxRenderer(void);

    /** Dtor. */
    virtual ~BoxRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(megamol::core::view::CallRender3D_2& call) override;

    bool GetExtents(core::view::CallRender3D_2& call) override;

private:
    static std::pair<std::vector<float>, std::vector<float>> prepareData(
        std::vector<BoxDataCall::box_entry_t> const& data) {
        std::vector<float> pos;
        pos.reserve(data.size() * 72);
        std::vector<float> col;
        col.reserve(data.size() * 96);

        for (auto const& e : data) {
            auto const& box = e.box_;
            auto lef = box.GetLeft();
            auto bot = box.GetBottom();
            auto bac = box.GetBack();
            auto rig = box.GetRight();
            auto top = box.GetTop();
            auto fro = box.GetFront();

            // Quad 1
            pos.push_back(lef);
            pos.push_back(bot);
            pos.push_back(bac);

            pos.push_back(rig);
            pos.push_back(bot);
            pos.push_back(bac);

            pos.push_back(rig);
            pos.push_back(top);
            pos.push_back(bac);

            pos.push_back(lef);
            pos.push_back(top);
            pos.push_back(bac);

            // Quad 2
            pos.push_back(rig);
            pos.push_back(bot);
            pos.push_back(bac);

            pos.push_back(rig);
            pos.push_back(bot);
            pos.push_back(fro);

            pos.push_back(rig);
            pos.push_back(top);
            pos.push_back(fro);

            pos.push_back(rig);
            pos.push_back(top);
            pos.push_back(bac);

            // Quad 3
            pos.push_back(rig);
            pos.push_back(bot);
            pos.push_back(fro);

            pos.push_back(lef);
            pos.push_back(bot);
            pos.push_back(fro);

            pos.push_back(lef);
            pos.push_back(top);
            pos.push_back(fro);

            pos.push_back(rig);
            pos.push_back(top);
            pos.push_back(fro);

            // Quad 4
            pos.push_back(lef);
            pos.push_back(bot);
            pos.push_back(fro);

            pos.push_back(lef);
            pos.push_back(bot);
            pos.push_back(bac);

            pos.push_back(lef);
            pos.push_back(top);
            pos.push_back(bac);

            pos.push_back(lef);
            pos.push_back(top);
            pos.push_back(fro);

            // Quad 5
            pos.push_back(lef);
            pos.push_back(bot);
            pos.push_back(bac);

            pos.push_back(lef);
            pos.push_back(bot);
            pos.push_back(fro);

            pos.push_back(rig);
            pos.push_back(bot);
            pos.push_back(fro);

            pos.push_back(rig);
            pos.push_back(bot);
            pos.push_back(bac);

            // Quad 6
            pos.push_back(lef);
            pos.push_back(top);
            pos.push_back(bac);

            pos.push_back(rig);
            pos.push_back(top);
            pos.push_back(bac);

            pos.push_back(rig);
            pos.push_back(top);
            pos.push_back(fro);

            pos.push_back(lef);
            pos.push_back(top);
            pos.push_back(fro);

            // color
            for (int i = 0; i < 24; ++i) {
                col.push_back(e.color_[0]);
                col.push_back(e.color_[1]);
                col.push_back(e.color_[2]);
                col.push_back(e.color_[3]);
            }
        }

        return std::make_pair(pos, col);
    }

    core::CallerSlot dataInSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    unsigned int frameID_ = 0;

    vislib::graphics::gl::GLSLShader boxShader_;

    GLuint vao_, vvbo_, cvbo_;

    float scaling_ = 1.0f;

}; // end class BoxRenderer

} // end namespace rendering
} // end namespace thermodyn
} // end namespace megamol
