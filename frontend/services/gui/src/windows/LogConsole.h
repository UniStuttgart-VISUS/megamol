/*
 * LogConsole.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
#define MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
#pragma once


#include "AbstractWindow.h"
#include "CommonTypes.h"
#include "mmcore/utility/log/StreamTarget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"


namespace megamol {
namespace gui {


/* ************************************************************************
 * The log buffer collecting all the messages
 */
class LogBuffer : public std::stringbuf {
public:
    LogBuffer() = default;
    ~LogBuffer() override = default;

    struct LogEntry {
        unsigned int level;
        std::string message;
    };

    int sync() override;

    inline const std::vector<LogEntry>& log() const {
        return this->messages;
    }

private:
    std::vector<LogEntry> messages;
};


/* ************************************************************************
 * The content of the log console GUI window
 */
class LogConsole : public AbstractWindow {
public:
    using lua_func_type = megamol::frontend_resources::common_types::lua_func_type;

    struct InputSharedData {
        bool move_cursor_to_end;
        std::vector<std::pair<std::string, std::string>> commands; // command, parameter hint
        std::vector<std::pair<std::string, std::string>> autocomplete_candidates;
        bool open_autocomplete_popup;
        std::vector<std::string> history;
        size_t history_index;
        std::string param_hint;
        size_t last_input_length;
    };

    explicit LogConsole(const std::string& window_name);
    ~LogConsole();

    bool Update() override;
    bool Draw() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

    void SetLuaFunc(lua_func_type* func) {
        this->input_lua_func = func;
    }

private:
    // VARIABLES --------------------------------------------------------------

    LogBuffer echo_log_buffer;
    std::ostream echo_log_stream;
    std::shared_ptr<megamol::core::utility::log::StreamTarget> echo_log_target;

    size_t log_msg_count;
    unsigned int scroll_down;
    unsigned int scroll_up;
    float last_window_height;

    unsigned int win_log_level; // [SAVED] Log level used in log window
    bool win_log_force_open;    // [SAVED] flag indicating if log window should be forced open on warnings and errors

    // Widgets
    HoverToolTip tooltip;

    // Input
    std::shared_ptr<InputSharedData> input_shared_data;
    bool input_reclaim_focus;
    std::string input_buffer;
    // where would I get this from? and the autocomplete stuff?
    lua_func_type* input_lua_func;
    bool is_autocomplete_popup_open;

    // FUNCTIONS --------------------------------------------------------------
    bool connect_log();
    void print_message(const LogBuffer::LogEntry& entry, unsigned int global_log_level) const;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
