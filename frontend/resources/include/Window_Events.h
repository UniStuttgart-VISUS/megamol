/*
 * Window_Events.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace megamol::frontend_resources {

//static std::string WindowState_Req_Name = "WindowState";
static std::string WindowEvents_Req_Name = "WindowEvents";

struct WindowState {
    int width = 1;
    int height = 1;
    bool is_focused = false;
    bool should_close = false;
    bool is_iconified = false;
    float x_contentscale = 1.0f;
    float y_contentscale = 1.0f;
    double time = 0.0;
    // std::vector<std::string> dropped_paths;
    // std::string get_clipboard_state;
    // std::string set_clipboard_state;
};
struct WindowEvents {
    std::vector<std::tuple<int, int>> size_events;
    std::vector<bool> is_focused_events;
    std::vector<bool> should_close_events;
    std::vector<bool> is_iconified_events;
    std::vector<std::tuple<float, float>> content_scale_events;
    std::vector<std::vector<std::string>> dropped_path_events;
    double time = 0.0;

    WindowState previous_state;

    // persistent state, only set once, do not change
    const char* getClipboardString() {
        return _getClipboardString_Func(_clipboard_user_data);
    }

    void setClipboardString(const char* string) {
        _setClipboardString_Func(_clipboard_user_data, string);
    }

    const char* (*_getClipboardString_Func)(void* user_data) = nullptr;
    void (*_setClipboardString_Func)(void* user_data, const char* string) = nullptr;
    void* _clipboard_user_data = nullptr;

    void apply_state() {

        previous_state.time = time;

        if (size_events.size()) {
            this->previous_state.width = std::get<0>(size_events.back());
            this->previous_state.height = std::get<1>(size_events.back());
        }

        if (is_focused_events.size())
            this->previous_state.is_focused = is_focused_events.back();

        if (should_close_events.size())
            this->previous_state.should_close = should_close_events.back();

        if (is_iconified_events.size())
            this->previous_state.is_iconified = is_iconified_events.back();

        if (content_scale_events.size()) {
            this->previous_state.x_contentscale = std::get<0>(content_scale_events.back());
            this->previous_state.y_contentscale = std::get<1>(content_scale_events.back());
        }
    }
    void clear() {
        apply_state();

        size_events.clear();
        is_focused_events.clear();
        should_close_events.clear();
        is_iconified_events.clear();
        content_scale_events.clear();
        dropped_path_events.clear();
        time = 0.0;
    }
};

namespace input = frontend_resources;

} // namespace megamol::frontend_resources
