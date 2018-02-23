#include "stdafx.h"
#include "mmcore/view/TransferFunction1D.h"

#include "mmcore/param/TransferFunc1DParam.h"

#include "vislib/sys/Log.h"

megamol::core::view::TransferFunction1D::TransferFunction1D(void)
    : megamol::core::Module()
    , megamol::core::LuaInterpreter<megamol::core::view::TransferFunction1D>(this)
    , getTFSlot("getTF", "Transfer function output")
    , tfParamSlot("transfunc", "Param slot recieving transfer function from configurator") {
    this->getTFSlot.SetCallback(megamol::core::view::CallGetTransferFunction::ClassName(),
        megamol::core::view::CallGetTransferFunction::FunctionName(0), &TransferFunction1D::getTFCallback);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->tfParamSlot << new megamol::core::param::TransferFunc1DParam("");
    this->tfParamSlot.SetUpdateCallback(&TransferFunction1D::tfUpdated);
    this->MakeSlotAvailable(&this->tfParamSlot);

    RegisterCallback<TransferFunction1D, &TransferFunction1D::parseTF>("mmliParseTF");
    FinalizeEnvironment();
}


megamol::core::view::TransferFunction1D::~TransferFunction1D(void) { this->Release(); }


bool megamol::core::view::TransferFunction1D::create(void) {
    return true;
}


void megamol::core::view::TransferFunction1D::release(void) {}


bool megamol::core::view::TransferFunction1D::getTFCallback(megamol::core::Call& c) {
    megamol::core::view::CallGetTransferFunction* outCall = dynamic_cast<megamol::core::view::CallGetTransferFunction*>(&c);
    if (outCall == nullptr) return false;

    if (tfChanged) {
        if (glIsTexture(this->texID)) {
            glDeleteTextures(1, &this->texID);
        }

        GLint otid = 0;
        glGenTextures(1, &this->texID);
        glGetIntegerv(GL_TEXTURE_BINDING_1D, &otid);
        glBindTexture(GL_TEXTURE_1D, this->texID);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, this->tf.size(), 0, GL_RGBA, GL_FLOAT, this->tf.data());
        glBindTexture(GL_TEXTURE_1D, otid);

        tfChanged = false;
    }

    outCall->SetTexture(this->texID, this->tf.size(), CallGetTransferFunction::TEXTURE_FORMAT_RGBA);

    return true;
}


bool megamol::core::view::TransferFunction1D::tfUpdated(megamol::core::param::ParamSlot& p) {
    std::string script = p.Param<megamol::core::param::TransferFunc1DParam>()->Value();
    std::string result;
    auto ret = RunString(script, result);
    if (!ret) {
        vislib::sys::Log::DefaultLog.WriteError("TransferFunction1D Lua ERROR: %s", result.c_str());
    }
    return true;
}


int megamol::core::view::TransferFunction1D::parseTF(lua_State* L) {
    int n = lua_gettop(L);
    if (!(n % 4) && n > 0) {
        // valid input
        std::vector<float> tmp;
        tmp.reserve(n);

        for (int i = 1; i <= n; i++) {
            if (lua_isnumber(L, i)) {
                tmp.push_back(lua_tonumber(L, i));
            } else {
                vislib::sys::Log::DefaultLog.WriteError("TransferFunction1D ERROR: Argument %d is not a number\n", i - 1);
                return 0;
            }
        }

        this->tf = tmp;
        this->tfChanged = true;
    } else {
        vislib::sys::Log::DefaultLog.WriteError("TransferFunction1D ERROR: Number of arguments must be multiple of four\n");
    }


    return 0;
}
