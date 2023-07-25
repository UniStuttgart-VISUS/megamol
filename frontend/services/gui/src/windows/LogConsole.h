/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#include "AbstractWindow.h"
#include "CommonTypes.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"


namespace megamol::gui {


/* ************************************************************************
 * The log buffer collecting all the messages
 */
class LogBuffer : public std::stringbuf {
public:
    LogBuffer() = default;
    ~LogBuffer() override = default;

    struct LogEntry {
        core::utility::log::Log::log_level level;
        std::string message;
    };

    int sync() override;

    inline std::vector<LogEntry> const& log() const {
        return this->messages;
    }

    inline std::vector<size_t> const& warn_log_indices() const {
        return this->warn_msg_indices;
    }

    inline std::vector<size_t> const& error_log_indices() const {
        return this->error_msg_indices;
    }

private:
    std::vector<LogEntry> messages;
    std::vector<size_t> warn_msg_indices;
    std::vector<size_t> error_msg_indices;
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
    };

    explicit LogConsole(const std::string& window_name);
    ~LogConsole();

    bool Update() override;
    bool Draw() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

    void SetLuaFunc(lua_func_type* func);

private:
    // VARIABLES --------------------------------------------------------------

    LogBuffer echo_log_buffer;
    std::ostream echo_log_stream;

    size_t log_msg_count;
    unsigned int scroll_down;
    unsigned int scroll_up;
    float last_window_height;
    int selected_candidate_index;

    core::utility::log::Log::log_level win_log_level; // [SAVED] Log level used in log window
    bool win_log_force_open; // [SAVED] flag indicating if log window should be forced open on warnings and errors

    // Widgets
    HoverToolTip tooltip;

    // Input
    std::shared_ptr<InputSharedData> input_shared_data;
    bool input_reclaim_focus;
    std::string input_buffer;
    // where would I get this from? and the autocomplete stuff?
    lua_func_type* input_lua_func;
    bool is_autocomplete_popup_open;

    std::size_t sink_idx_;
};


} // namespace megamol::gui
