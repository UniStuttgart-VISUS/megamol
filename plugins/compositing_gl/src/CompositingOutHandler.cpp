#include "CompositingOutHandler.h"


namespace megamol::compositing_gl {

bool CompositingOutHandler::updateSelectionsExternally(core::param::ParamSlot& slot) {
    updateSelections(slot);
    //external update
    return externalUpdateFunc_();
}

bool CompositingOutHandler::updateSelections(core::param::ParamSlot& slot) {
    selectedType_ = GL_FLOAT;
    recentlyChanged_ = true;
    unsigned int e = availableInternalFormats_[slot.Param<core::param::EnumParam>()->Value()];
    //error
    selectedInternal_ = e;
    switch (e) {
    case GL_RGBA32F:
    case GL_RGBA16F:
    case GL_RGBA8_SNORM:
        selectedFormat_ = GL_RGBA;
        return true;
    case GL_RGB32F:
    case GL_RGB16F:
    case GL_RGB8_SNORM:
        selectedFormat_ = GL_RGB;
        return true;
    case GL_RG32F:
    case GL_RG16F:
    case GL_RG8_SNORM:
        selectedFormat_ = GL_RG;
        return true;
    case GL_R32F:
    case GL_R16F:
    case GL_R8_SNORM:
        selectedFormat_ = GL_RED;
        return true;
    default:
        selectedFormat_ = GL_RGBA;
        return false;
    }
}

GLenum CompositingOutHandler::getInternalFormat() {
    return selectedInternal_;
}
GLenum CompositingOutHandler::getFormat() {
    return selectedFormat_;
}
GLenum CompositingOutHandler::getType() {
    return selectedType_;
}

bool CompositingOutHandler::recentlyChanged() {
    if (recentlyChanged_) {
        recentlyChanged_ = false;
        return true;
    } else {
        return false;
    }
}

CompositingOutHandler::CompositingOutHandler(std::string defineName, std::vector<unsigned int> allowedInternalFormats,
    std::string slotName, std::string slotDesc)
        : formatSlot_(slotName.c_str(), slotDesc.c_str())
        , defineName_(defineName)
        , availableInternalFormats_(allowedInternalFormats)
        , selectedInternal_(allowedInternalFormats[0]) {
    auto out_tex_formats = new megamol::core::param::EnumParam(0);
    for (int i = 0; i < allowedInternalFormats.size(); i++) {
        out_tex_formats->SetTypePair(i, enumToString(allowedInternalFormats[i]).c_str());
    }
    formatSlot_.SetParameter(out_tex_formats);
    formatSlot_.SetUpdateCallback(this, &CompositingOutHandler::updateSelections);
    updateSelections(formatSlot_);
}

/**
    Constructor with update function reference parameter.
    Function from parameter is executed after selected paramaters are updated.
*/
CompositingOutHandler::CompositingOutHandler(std::string defineName, std::vector<unsigned int> allowedInternalFormats,
    std::function<bool()> externalUpdateFunc, std::string slotName, std::string slotDesc)
        : formatSlot_(slotName.c_str(), slotDesc.c_str())
        , defineName_(defineName)
        , availableInternalFormats_(allowedInternalFormats)
        , selectedInternal_(allowedInternalFormats[0])
        , externalUpdateFunc_(externalUpdateFunc) {
    auto out_tex_formats = new megamol::core::param::EnumParam(0);
    for (int i = 0; i < allowedInternalFormats.size(); i++) {
        out_tex_formats->SetTypePair(i, enumToString(allowedInternalFormats[i]).c_str());
    }
    formatSlot_.SetParameter(out_tex_formats);
    formatSlot_.SetUpdateCallback(this, &CompositingOutHandler::updateSelectionsExternally);
    updateSelections(formatSlot_);
}

megamol::core::AbstractSlot* CompositingOutHandler::getFormatSelectorSlot() {
    return &formatSlot_;
}

std::string CompositingOutHandler::enumToString(unsigned int e) {
    switch (e) {
    case GL_RGBA32F:
        return "GL_RGBA32F";
    case GL_RGBA16F:
        return "GL_RGBA16F";
    case GL_RGBA8_SNORM:
        return "GL_RGBA8_SNORM";
    case GL_RGB32F:
        return "GL_RGB32F";
    case GL_RGB16F:
        return "GL_RGB16F";
    case GL_RGB8_SNORM:
        return "GL_RGB8_SNORM";
    case GL_RG32F:
        return "GL_RG32F";
    case GL_RG16F:
        return "GL_RG16F";
    case GL_RG8_SNORM:
        return "GL_RG8_SNORM";
    case GL_R32F:
        return "GL_R32F";
    case GL_R16F:
        return "GL_R16F";
    case GL_R8_SNORM:
        return "GL_R8_SNORM";
    }
}

std::string CompositingOutHandler::enumToDefinition(unsigned int e) {
    switch (e) {
    case GL_RGBA32F:
        return "rgba32f";
    case GL_RGBA16F:
        return "rgba16f";
    case GL_RGBA8_SNORM:
        return "rgba8_snorm";
    case GL_RGB32F:
        return "rgb32f";
    case GL_RGB16F:
        return "rgb16f";
    case GL_RGB8_SNORM:
        return "rgb8_snorm";
    case GL_RG32F:
        return "rg32f";
    case GL_RG16F:
        return "rg16f";
    case GL_RG8_SNORM:
        return "rg8_snorm";
    case GL_R32F:
        return "r32f";
    case GL_R16F:
        return "r16f";
    case GL_R8_SNORM:
        return "r8_snorm";
    }
}


std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> CompositingOutHandler::addDefinitions(
    msf::ShaderFactoryOptionsOpenGL shdr_options) {
    auto shader_options_flags = std::make_unique<msf::ShaderFactoryOptionsOpenGL>(shdr_options);
    shader_options_flags->addDefinition(defineName_, enumToDefinition(this->selectedInternal_));
    return shader_options_flags;
}

void CompositingOutHandler::addDefinitions(std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> shdr_options) {
    shdr_options->addDefinition(defineName_, enumToDefinition(this->selectedInternal_));
}
} // namespace megamol::compositing_gl
