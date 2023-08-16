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
    /** Vector holding all allowed formats, set by user in constructor */
    std::vector<unsigned int> availableInternalFormats_;

    unsigned int selectedFormat_ = 0;
    unsigned int selectedType_ = GL_FLOAT;
    unsigned int selectedInternal_ = 0;

    bool recentlyChanged_ = false;
    std::string defineName_;
    megamol::core::param::ParamSlot formatSlot_;

    /** Function to be called when formats are changed through GUI parameter. */
    std::function<bool()> externalUpdateFuncStd_;
    /** Function to be called when formats are changed through GUI parameter. */
    std::function<bool()> externalUpdateFunc_;

    std::string enumToString(unsigned int e);
    std::string enumToDefinition(unsigned int e);

    /**
     * \brief Updates currently chosen formats. Sets recently changed to true. Does not recompile shaders.
     *
     * @return 'true' if successfully updated, 'false' otherwise
     */
    bool updateSelections(core::param::ParamSlot& slot);

    /**
     * \brief Updates currently chosen formats and calls external function to recompile shaders and update textures.
     *
     * @return 'true' if both updates of chosen formats and external update call was successful, 'false' otherwise
     */
    bool updateSelectionsExternally(core::param::ParamSlot& slot);

public:
    GLenum getInternalFormat();
    GLenum getFormat();
    GLenum getType();


    /*
    * Constructor for manual handeling of changes in selected formats.
    * On changed format selection, recently changed returns 'true'.
    * 
    * @Param define Name of the String to be replaced within shaders.
    * @Param allowedTypes Enum representation of opengl internal texture formats.
    * @Param slotName Visible name of format selection field in GUI
    * @Param slotDesc Description of selection field in GUI
    */
    CompositingOutHandler(std::string defineName, std::vector<unsigned int> allowedTypes,
        std::string slotName = "slot for selecting Out Formats", std::string slotDesc = "Slot for selecting Tex Outs");

    /*
     * Constructor for automatic handeling of changes in selected formats.
     * This constructor expects a function in owner to handle relevant updates outside this object.
     * On selection change 'externalUpdateFunc' is called.
     * 
     * @param define: Name of the String to be replaced within shaders.
     * @param allowed:Types Enum representation of opengl internal texture formats.
     * @param externalUpdateFunc: (Callback)Function of owner to update shaders and texture format variables when format selection is changed.
     * @param slotName: Visible name of format selection field in GUI
     * @param slotDesc: Description of selection field in GUI
     */
    CompositingOutHandler(std::string defineName, std::vector<unsigned int> allowedTypes,
        std::function<bool()> externalUpdatFunc, std::string slotName = "slot for selecting Out Formats",
        std::string slotDesc = "Slot for selecting Tex Outs");

    /*
    * @return 'true' if selection changed since last call of this function, 'false' if no change happened.
    */
    bool recentlyChanged();

    /*
    * Adds definitions to shader.
    */
    std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> addDefinitions(msf::ShaderFactoryOptionsOpenGL);
    /*
     * Adds definitions to shader.
     */
    void addDefinitions(std::unique_ptr<msf::ShaderFactoryOptionsOpenGL>);

    megamol::core::AbstractSlot* getFormatSelectorSlot();
};
} // namespace megamol::compositing_gl
