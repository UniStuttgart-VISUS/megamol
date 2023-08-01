/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <map>
#include <string>

#include <nlohmann/json.hpp>

#include "mmcore/utility/log/Log.h"

#define GUI_JSON_TAG_GUISTATE_PARAMETERS ("ParameterStates")

namespace megamol::core::param {

class AbstractParamPresentation {
public:
    // Available parameter types
    /// ! Add new type to name function.
    enum ParamType {
        BOOL,
        BUTTON,
        COLOR,
        ENUM,
        FILEPATH,
        FLEXENUM,
        FLOAT,
        INT,
        STRING,
        TERNARY,
        TRANSFERFUNCTION,
        VECTOR2F,
        VECTOR3F,
        VECTOR4F,
        GROUP_ANIMATION,
        GROUP_3D_CUBE,
        UNKNOWN
    };

    /// ! Add new widget presentation to name map in ctor.
    enum Presentation : int { /// (limited to 32 options)
        NONE = 0,
        Basic = 1 << 1,    // Basic widget (is supported for all parameter types - not for groups) -> Default
        String = 1 << 2,   // String widget (is supported for all types, uses ValueString function of parameters)
        Color = 1 << 3,    // Color editor widget
        FilePath = 1 << 4, // File path widget
        TransferFunction = 1 << 5, // Transfer function editor widget
        Knob = 1 << 6,             // Knob widget for float
        Slider = 1 << 7,           // Slider widget for int and float
        Drag = 1 << 8,             // Drag widget for int and float
        Direction = 1 << 9,        // Widget for direction of vec3
        Rotation = 1 << 10,        // Widget for rotation of vec4
        PinMouse = 1 << 11,        // Pin parameter value to mouse position
        Group_Animation = 1 << 12, // Animation widget group
        Group_3D_Cube = 1 << 13,   // 3D cube widget group
        Checkbox = 1 << 14,        // Check box for bool
    };

    /**
     * Initalise presentation for parameter once.
     *
     * @param param_type   The parameters type.
     */
    void InitPresentation(AbstractParamPresentation::ParamType param_type);

    /**
     * Answer visibility in GUI.
     *
     * @return GUI visibility
     */
    inline bool IsGUIVisible() const {
        return this->visible;
    }

    /**
     * Set visibility in GUI.
     *
     * @param visible True: visible in GUI, false: invisible
     */
    inline void SetGUIVisible(bool visible) {
        this->visible = visible;
    }

    /**
     * Answer accessibility in GUI.
     *
     * @return GUI accessibility
     */
    inline bool IsGUIReadOnly() const {
        return this->read_only;
    }

    /**
     * Set accessibility in GUI.
     *
     * @param read_only True: read-only in GUI, false: writable
     */
    inline void SetGUIReadOnly(bool read_only) {
        this->read_only = read_only;
    }

    bool IsHighlight() const {
        return highlight;
    }

    void SetHighlight(bool highlight) {
        this->highlight = highlight;
    }

    /**
     * Set presentation of parameter in GUI.
     *
     * @param presentation Presentation of parameter in GUI.
     *
     * @return True if given presentation is compatible, false otherwise.
     */
    void SetGUIPresentation(AbstractParamPresentation::Presentation presentS);

    /**
     * Answer parameter presentation in GUI.
     *
     * @return Parameter presentation.
     */
    inline AbstractParamPresentation::Presentation GetGUIPresentation() const {
        return this->presentation;
    }

    /**
     * Answer whether given presentation is compatible with parameter.
     *
     * @return True if given presentation is compatible, false otherwise.
     */
    inline bool IsPresentationCompatible(AbstractParamPresentation::Presentation present) const {
        return (AbstractParamPresentation::Presentation::NONE != (present & this->compatible));
    }

    /**
     * Get presentation name map.
     *
     * @return The presentation name map.
     */
    const std::map<AbstractParamPresentation::Presentation, std::string>& GetPresentationNameMap() const {
        return this->presentation_name_map;
    }

    /**
     * Get presentation name.
     *
     * @param presentation Presentation of parameter in GUI.
     *
     * @return The human readable name of the given presentation.
     */
    std::string GetPresentationName(AbstractParamPresentation::Presentation const present) const {
        return this->presentation_name_map.at(present);
    }

    /**
     * Get human readable parameter type.
     *
     * @param type The parameter type.
     *
     * @return The human readable name of the given parameter type.
     */
    static std::string GetTypeName(AbstractParamPresentation::ParamType type);

    /**
     * De-/Serialization of parameters GUI state.
     */
    bool StateFromJSON(const nlohmann::json& in_json, const std::string& param_fullname);
    bool StateToJSON(nlohmann::json& inout_json, const std::string& param_fullname);

    AbstractParamPresentation();

    virtual ~AbstractParamPresentation() = default;

private:
    // VARIABLES --------------------------------------------------------------

    /* Show or hide the parameter in the GUI.
        Parameter is implicitly hidden in GUI if other than raw value view is selected. */
    bool visible;

    /* Make parameter read-only in the GUI. */
    bool read_only;

    /* Highlight the parameter in the GUI */
    bool highlight;

    /* Presentation (= widget representation) of parameter in the GUI. */
    AbstractParamPresentation::Presentation presentation;

    /* Compatible presentations */
    AbstractParamPresentation::Presentation compatible;

    /* Falg ensuring that initialisation can only be applied once. */
    bool initialised;

    /* Presentation name map */
    std::map<Presentation, std::string> presentation_name_map;
};

inline AbstractParamPresentation::Presentation operator|(
    AbstractParamPresentation::Presentation a, AbstractParamPresentation::Presentation b) {
    return static_cast<AbstractParamPresentation::Presentation>(static_cast<int>(a) | static_cast<int>(b));
}

inline AbstractParamPresentation::Presentation operator&(
    AbstractParamPresentation::Presentation a, AbstractParamPresentation::Presentation b) {
    return static_cast<AbstractParamPresentation::Presentation>(static_cast<int>(a) & static_cast<int>(b));
}

} // namespace megamol::core::param
