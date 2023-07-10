#pragma once
/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */
#include "compositing_gl/CompositingCalls.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/ModuleGL.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#include <string>
#include <vector>


namespace megamol::compositing_gl {

class CompositingOutHandler {
private:
    std::vector<unsigned int> availableInternalFormats_;
    unsigned int selectedFormat_ = 0;
    unsigned int selectedType_ = GL_FLOAT;
    unsigned int selectedInternal_ = 0;
    bool recentlyChanged_ = false;
    std::string defineName_;
    megamol::core::param::ParamSlot outSlot_;
    megamol::core::param::ParamSlot formatSlot_;

    std::function<bool()> externalUpdateFuncStd_;
    std::function<bool()> externalUpdateFunc_;

    std::string enumToString(unsigned int e);
    std::string enumToDefinition(unsigned int e);


    bool updateSelections(core::param::ParamSlot& slot);
    bool updateSelectionsExternally(core::param::ParamSlot& slot);

public:
    bool recentlyChanged();
    GLenum getInternalFormat();
    GLenum getFormat();
    GLenum getType();
    //TODO: make slot names unique
    CompositingOutHandler(std::string defineName, std::vector<unsigned int> allowedTypes,
        std::string slotName = "slot for selecting Out Formats", std::string slotDesc = "Slot for selecting Tex Outs");
    CompositingOutHandler(std::string defineName, std::vector<unsigned int> allowedTypes,
        std::function<bool()> externalUpdatFunc_, std::string slotName = "slot for selecting Out Formats",
        std::string slotDesc = "Slot for selecting Tex Outs");

    //TODO bool return?
    std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> handleDefinitions(msf::ShaderFactoryOptionsOpenGL);
    void handleDefinitions(std::unique_ptr<msf::ShaderFactoryOptionsOpenGL>);

    megamol::core::AbstractSlot* getOutSlot();
    megamol::core::AbstractSlot* getFormatSelectorSlot();
};
} // namespace megamol::compositing_gl
