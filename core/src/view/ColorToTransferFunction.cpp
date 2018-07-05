#include "stdafx.h"
#include "mmcore/view/ColorToTransferFunction.h"

#include "mmcore/param/ColorParam.h"

#include "vislib/sys/Log.h"

megamol::core::view::ColorToTransferFunction::ColorToTransferFunction(void)
    : megamol::core::Module()
    , megamol::core::LuaInterpreter<megamol::core::view::ColorToTransferFunction>(this)
    , getTFSlot("getTF", "Transfer function output")
    , colorSlot("transfunc", "Param slot recieving transfer function from configurator")
    , tfInvalidated(true) {
    this->getTFSlot.SetCallback(megamol::core::view::CallGetTransferFunction::ClassName(),
        megamol::core::view::CallGetTransferFunction::FunctionName(0), &ColorToTransferFunction::getTFCallback);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->colorSlot << new megamol::core::param::ColorParam("");
    this->colorSlot.SetUpdateCallback(&ColorToTransferFunction::colorUpdated);
    this->MakeSlotAvailable(&this->colorSlot);

    RegisterCallback<ColorToTransferFunction, &ColorToTransferFunction::parseTF>("mmliParseTF");
    FinalizeEnvironment();
}

megamol::core::view::ColorToTransferFunction::~ColorToTransferFunction(void) { this->Release(); }

bool megamol::core::view::ColorToTransferFunction::getTFCallback(megamol::core::Call& c) {
    megamol::core::view::CallGetTransferFunction* outCall =
        dynamic_cast<megamol::core::view::CallGetTransferFunction*>(&c);
    if (outCall == nullptr) return false;

    if (tfInvalidated) {
        if (glIsTexture(this->texID)) {
            glDeleteTextures(1, &this->texID);
        }

        GLint otid = 0;
        glGenTextures(1, &this->texID);
        glGetIntegerv(GL_TEXTURE_BINDING_1D, &otid);
        glBindTexture(GL_TEXTURE_1D, this->texID);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, this->tf.size() / 4, 0, GL_RGBA, GL_FLOAT, this->tf.data());
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glBindTexture(GL_TEXTURE_1D, otid);

        outCall->SetTexture(
            this->texID, this->tf.size() / 4, this->tf.data(), CallGetTransferFunction::TEXTURE_FORMAT_RGBA);

        tfInvalidated = false;
    }

    return true;
}

bool megamol::core::view::ColorToTransferFunction::colorUpdated(megamol::core::param::ParamSlot& p) {
    std::string script = p.Param<megamol::core::param::ColorParam>()->Value();
    std::string result;
    auto ret = RunString(script, result);
    if (!ret) {
        vislib::sys::Log::DefaultLog.WriteError("ColorToTransferFunction Lua ERROR: %s", result.c_str());
    }
    return true;
}

int megamol::core::view::ColorToTransferFunction::parseTF(lua_State* L) {
    std::vector<std::string> tokens;

    if (lua_isstring(L, 1)) {
        size_t len = 0;
        auto const args = lua_tolstring(L, 1, &len);
        std::string s;
        for (size_t i = 0; i < len; ++i) {
            if (args[i] == ',' || args[i] == '\0') {
                tokens.push_back(s);
                s.clear();
                continue;
            }

            if (args[i] == ' ') continue;

            s.push_back(args[i]);
        }
        tokens.push_back(s);
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "ColorToTransferFunction ERROR: mmliParseTF expected string as argument\n");
    }

    size_t n = tokens.size();

    if (!(n % 4) && n > 0) {
        // valid input
        std::vector<float> tmp;
        tmp.reserve(n);

        for (auto const& tok : tokens) {
            auto const val = atof(tok.c_str());
            tmp.push_back(val);
        }

        this->tf = tmp;
        this->tfInvalidated = true;
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "ColorToTransferFunction ERROR: Number of arguments must be multiple of four\n");
    }

    return 0;
}
