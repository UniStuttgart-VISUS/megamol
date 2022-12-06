#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "thermodyn/BoxDataCall.h"


namespace megamol::thermodyn_gl::rendering {

class BoxRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "BoxRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for box glyphs.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
        return true;
    }

    /** Ctor. */
    BoxRenderer();

    /** Dtor. */
    virtual ~BoxRenderer();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

private:
    std::pair<std::vector<float>, std::vector<float>> drawData;
    static void prepareData(std::vector<thermodyn::BoxDataCall::box_entry_t> const& indata,
        std::pair<std::vector<float>, std::vector<float>>& outdata) {

        auto& pos = outdata.first;
        pos.clear();
        pos.reserve(indata.size() * 72);
        auto& col = outdata.second;
        col.clear();
        col.reserve(indata.size() * 96);

        for (auto const& e : indata) {
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
    }

    core::CallerSlot dataInSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    core::param::ParamSlot calculateGlobalBoundingBoxParam;

    unsigned int frameID_ = 0;

    std::unique_ptr<glowl::GLSLProgram> boxShader_;

    GLuint vao_, vvbo_, cvbo_;

    float scaling_ = 1.0f;

}; // end class BoxRenderer

} // namespace megamol::thermodyn_gl::rendering
