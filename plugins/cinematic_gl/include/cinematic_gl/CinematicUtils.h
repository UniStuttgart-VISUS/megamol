/*
 * CinematicUtils.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "mmcore/view/Camera.h"
#include "mmcore_gl/utility/RenderUtils.h"


// ##################################################################### //


namespace megamol::cinematic_gl {


/*
 * Cinematic utility functionality (colors, text, menu, ...).
 */
class CinematicUtils : public core_gl::utility::RenderUtils {

public:
    CinematicUtils();

    ~CinematicUtils();

    enum Colors {
        BACKGROUND,
        FOREGROUND,
        KEYFRAME,
        KEYFRAME_DRAGGED,
        KEYFRAME_SELECTED,
        KEYFRAME_SPLINE,
        MENU,
        FONT,
        FONT_HIGHLIGHT,
        LETTER_BOX,
        FRAME_MARKER,
        MANIPULATOR_X,
        MANIPULATOR_Y,
        MANIPULATOR_Z,
        MANIPULATOR_VECTOR,
        MANIPULATOR_ROTATION,
        MANIPULATOR_CTRLPOINT
    };

    bool Initialise(frontend_resources::RuntimeConfig const& runtimeConf);

    const glm::vec4 Color(CinematicUtils::Colors color) const;

    void SetBackgroundColor(glm::vec4 bc) {
        this->background_color = bc;
    };

    void PushMenu(const glm::mat4& ortho, const std::string& left_label, const std::string& middle_label,
        const std::string& right_label, glm::vec2 dim_vp, float depth);

    void HotkeyWindow(bool& inout_show, const glm::mat4& ortho, glm::vec2 dim_vp);

    void Push2DText(const glm::mat4& ortho, const std::string& text, float x, float y);

    void DrawAll(const glm::mat4& mvp, glm::vec2 dim_vp);

    float GetTextLineHeight();

    float GetTextLineWidth(const std::string& text_line);

    void SetTextRotation(float a, glm::vec3 vec);

    void ResetTextRotation();

    bool Initialized() {
        return this->init_once;
    }

private:
    // VARIABLES ------------------------------------------------------- //

    megamol::core::utility::SDFFont font;

    bool init_once;

    float menu_font_size;
    float menu_height;

    glm::vec4 background_color;
    bool hotkey_window_setup_once;

    // FUNCTIONS ------------------------------------------------------- //

    const float lightness(glm::vec4 background) const;

    void gui_update();

    void gui_table_row(const char* left, const char* right);
};

} // namespace megamol::cinematic_gl
