/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#include "AbstractWindow.h"
#include "widgets/HoverToolTip.h"


namespace megamol::gui {

/* ************************************************************************
 * The performance monitor GUI window
 */
class PerformanceMonitor : public AbstractWindow {
public:
    enum TimingMode { TIMINGMODE_FPS, TIMINGMODE_MS };

    explicit PerformanceMonitor(const std::string& window_name);
    ~PerformanceMonitor() = default;

    // Call each new frame
    inline void SetData(float current_averaged_fps, float current_averaged_ms, size_t current_frame_id) {
        this->averaged_fps = current_averaged_fps;
        this->averaged_ms = current_averaged_ms;
        this->frame_id = current_frame_id;
    }

    bool Update() override;
    bool Draw() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

private:
    // VARIABLES --------------------------------------------------------------

    bool win_show_options = false;        // [SAVED] show/hide fps/ms options.
    int win_buffer_size = 20;             // [SAVED] maximum count of values in value array
    float win_refresh_rate = 2.0f;        // [SAVED] maximum delay when fps/ms value should be renewed.
    TimingMode win_mode = TIMINGMODE_FPS; // [SAVED] mode for displaying either FPS or MS
    float win_current_delay = 0.0f;       // current delay between frames
    std::vector<float> win_ms_values;     // current ms values
    std::vector<float> win_fps_values;    // current fps values
    float win_ms_max = 1.0f;              // current ms plot scaling factor
    float win_fps_max = 1.0f;             // current fps plot scaling factor

    size_t frame_id;
    float averaged_fps;
    float averaged_ms;

    // Widgets
    HoverToolTip tooltip;
};

} // namespace megamol::gui
