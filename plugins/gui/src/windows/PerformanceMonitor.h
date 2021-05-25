/*
 * PerformanceMonitor.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PERFORMANCEMONITOR_H_INCLUDED
#define MEGAMOL_GUI_PERFORMANCEMONITOR_H_INCLUDED
#pragma once


#include "WindowConfiguration.h"


namespace megamol {
namespace gui {

    /*
     * The ...
     */
    class PerformanceMonitor : public WindowConfiguration {
    public:

        /** Timing mode for performance windows. */
        enum TimingMode { TIMINGMODE_FPS, TIMINGMODE_MS };

        PerformanceMonitor();
        ~PerformanceMonitor();

        void Update() override;

        void Draw() override;

        bool SpecificStateFromJSON(const nlohmann::json& in_json) override;

        bool SpecificStateToJSON(nlohmann::json& inout_json) override;

    private:
        // VARIABLES --------------------------------------------------------------

        bool fpsms_show_options = false;        // [SAVED] show/hide fps/ms options.
        int fpsms_buffer_size = 20;             // [SAVED] maximum count of values in value array
        float fpsms_refresh_rate = 2.0f;        // [SAVED] maximum delay when fps/ms value should be renewed.
        TimingMode fpsms_mode = TIMINGMODE_FPS; // [SAVED] mode for displaying either FPS or MS
        float tmp_current_delay = 0.0f;         // current delay between frames
        std::vector<float> tmp_ms_values;       // current ms values
        std::vector<float> tmp_fps_values;      // current fps values
        float tmp_ms_max = 1.0f;                // current ms plot scaling factor
        float tmp_fps_max = 1.0f;               // current fps plot scaling factor

        // FUNCTIONS --------------------------------------------------------------


    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PERFORMANCEMONITOR_H_INCLUDED
