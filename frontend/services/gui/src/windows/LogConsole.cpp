/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "LogConsole.h"

#include <regex>

#include <imgui_stdlib.h>
#include <spdlog/sinks/ostream_sink.h>

#include "widgets/ButtonWidgets.h"

using namespace megamol::gui;


namespace megamol::gui {

int Input_Text_Callback(ImGuiInputTextCallbackData* data) {

    /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[LogConsole] DEBUG: cursor: %d, selection: %d-%d", data->CursorPos, data->SelectionStart, data->SelectionEnd);
    auto user_data = static_cast<megamol::gui::LogConsole::InputSharedData*>(data->UserData);
    if (user_data == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[LogConsole] Unable to get pointer to user data.");
    }

    switch (data->EventFlag) {
    case ImGuiInputTextFlags_CallbackAlways: {

        // Move cursor to end of buffer when buffer was extended outside TextInput widget
        if (user_data->move_cursor_to_end) {
            data->CursorPos = (data->BufTextLen > 0) ? (data->BufTextLen - 1) : (0); // Place cursor within brackets
            user_data->move_cursor_to_end = false;
        }

        auto input_str = std::string(data->Buf, data->BufTextLen);
        if (input_str != user_data->history[user_data->history_index]) {

            //Adjust current input entry in history on any change of selected history entry
            user_data->history.back() = input_str;
            user_data->history_index = user_data->history.size() - 1;

            // Look for suitable parameter hint for given command (only if input changes)
            user_data->param_hint.clear();
            auto bracket_pos = input_str.find('(');
            if (bracket_pos != std::string::npos) {
                auto cmd = input_str.substr(0, bracket_pos);
                for (int i = 0; i < user_data->commands.size(); i++) {
                    if (gui_utils::CaseInsensitiveStringEqual(user_data->commands[i].first, cmd)) {
                        user_data->param_hint = user_data->commands[i].second;
                    }
                }
            }
        }
        break;
    }
    case ImGuiInputTextFlags_CallbackCompletion: {
        // TEXT COMPLETION

        // Locate beginning of current word
        const char* word_end = data->Buf + data->CursorPos;
        const char* word_start = word_end;
        while (word_start > data->Buf) {
            const char c = word_start[-1];
            if (c == ' ' || c == '\t' || c == ',' || c == ';')
                break;
            word_start--;
        }

        // Build a list of candidates
        const std::string input = std::string(word_start, (int) (word_end - word_start));
        user_data->autocomplete_candidates.clear();
        for (int i = 0; i < user_data->commands.size(); i++) {
            if (gui_utils::CaseInsensitiveStringContain(user_data->commands[i].first, input)) {
                user_data->autocomplete_candidates.push_back(user_data->commands[i]);
            }
        }

        if (user_data->autocomplete_candidates.empty()) {
            // No match
        } else if (user_data->autocomplete_candidates.size() == 1) {
            // Single match. Delete the complete current input and replace it entirely so we've got nice casing.
            data->DeleteChars(0, data->BufTextLen);
            data->InsertChars(data->CursorPos, user_data->autocomplete_candidates[0].first.data());
            data->InsertChars(data->CursorPos, "()");
            data->CursorPos--;
        } else {
            // Multiple matches. Complete as much as we can..
            // So inputing "C"+Tab will complete to "CL" then display "CLEAR" and "CLASSIFY" as matches.
            int match_len = (int) (word_end - word_start);
            for (;;) {
                int c = 0;
                bool all_candidates_matches = true;
                for (int i = 0; i < user_data->autocomplete_candidates.size() && all_candidates_matches; i++)
                    if (i == 0)
                        c = toupper(user_data->autocomplete_candidates[i].first[match_len]);
                    else if (c == 0 || c != toupper(user_data->autocomplete_candidates[i].first[match_len]))
                        all_candidates_matches = false;
                if (!all_candidates_matches)
                    break;
                match_len++;
            }

            if (match_len > 0) {
                data->DeleteChars((int) (word_start - data->Buf), (int) (word_end - word_start));
                data->InsertChars(data->CursorPos, user_data->autocomplete_candidates[0].first.data(),
                    user_data->autocomplete_candidates[0].first.data() + match_len);
            }
        }

        user_data->open_autocomplete_popup = (user_data->autocomplete_candidates.size() > 1);
        break;
    }
    case ImGuiInputTextFlags_CallbackHistory: {
        // HISTORY

        int prev_history_pos = static_cast<int>(user_data->history_index);
        if (data->EventKey == ImGuiKey_UpArrow) {
            if (user_data->history_index > 0) {
                user_data->history_index--;
            }
        } else if (data->EventKey == ImGuiKey_DownArrow) {
            if (user_data->history_index < (user_data->history.size() - 1)) {
                user_data->history_index++;
            }
        }

        if (prev_history_pos != static_cast<int>(user_data->history_index)) {
            if (prev_history_pos == (user_data->history.size() - 1)) {
                auto input_str = std::string(data->Buf, data->BufTextLen);
                user_data->history.back() = input_str;
            }
            auto history_str = user_data->history[user_data->history_index];
            data->DeleteChars(0, data->BufTextLen);
            data->InsertChars(0, history_str.c_str());
        }
        break;
    }
    default:
        break;
    }
    return 0;
}

} // namespace megamol::gui


int megamol::gui::LogBuffer::sync() {
    try {
        auto message_str = this->str();
        if (!message_str.empty()) {
            // Split message string
            auto split_index = message_str.find('\n');
            while (split_index != std::string::npos) {
                // Assuming new line of log message of format "<level>|<message>\r\n"
                auto new_message = message_str.substr(0, split_index + 1);
                bool extracted_new_message = false;
                auto seperator_index = new_message.find('|');
                if (seperator_index != std::string::npos) {
                    auto level_str = new_message.substr(0, seperator_index);
                    auto log_level = core::utility::log::Log::ParseLevelAttribute(level_str);
                    if (log_level != megamol::core::utility::log::Log::log_level::none) {
                        this->messages.push_back({log_level, new_message});
                        size_t msg_index = this->messages.size() - 1;
                        if (log_level == megamol::core::utility::log::Log::log_level::warn) {
                            this->warn_msg_indices.push_back(msg_index);
                        } else if (log_level == megamol::core::utility::log::Log::log_level::error) {
                            this->warn_msg_indices.push_back(msg_index);
                            this->error_msg_indices.push_back(msg_index);
                        }
                        extracted_new_message = true;
                    }
                }
                if (!extracted_new_message) {
                    // Append new line of previous log message
                    auto log_level = this->messages.back().level;
                    this->messages.push_back({log_level, new_message});
                    size_t msg_index = this->messages.size() - 1;
                    if (log_level == megamol::core::utility::log::Log::log_level::warn) {
                        this->warn_msg_indices.push_back(msg_index);
                    } else if (log_level == megamol::core::utility::log::Log::log_level::error) {
                        this->warn_msg_indices.push_back(msg_index);
                        this->error_msg_indices.push_back(msg_index);
                    }
                }
                message_str = message_str.substr(split_index + 1);
                split_index = message_str.find('\n');
            }
            this->str("");
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Log Console Buffer Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return 1;
    };
    return 0;
}

// ----------------------------------------------------------------------------

megamol::gui::LogConsole::LogConsole(const std::string& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_LOGCONSOLE)
        , echo_log_buffer()
        , echo_log_stream(&this->echo_log_buffer)
        , log_msg_count(0)
        , scroll_down(2)
        , scroll_up(0)
        , last_window_height(0.0f)
        , selected_candidate_index(0)
        , win_log_level(megamol::core::utility::log::Log::log_level::warn)
        , win_log_force_open(true)
        , tooltip()
        , input_shared_data(nullptr)
        , input_reclaim_focus(false)
        , input_buffer()
        , luaApi(nullptr)
        , is_autocomplete_popup_open(false) {
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(this->echo_log_stream);
    sink->set_pattern(core::utility::log::Log::std_pattern);
    sink->set_level(spdlog::level::level_enum::info);
    sink_idx_ = megamol::core::utility::log::Log::DefaultLog.AddEchoTarget(sink);

    // Configure CONSOLE Window
    this->win_config.size = ImVec2(500.0f * megamol::gui::gui_scaling.Get(), 50.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F9, core::view::Modifier::NONE);

    // Initialise
    this->input_shared_data = std::make_shared<InputSharedData>();
    this->input_shared_data->open_autocomplete_popup = false;
    this->input_shared_data->move_cursor_to_end = false;
    this->input_shared_data->history.push_back(this->input_buffer);
    this->input_shared_data->history_index = this->input_shared_data->history.size() - 1;
}


LogConsole::~LogConsole() {
    megamol::core::utility::log::Log::DefaultLog.RemoveEchoTarget(sink_idx_);
}


bool megamol::gui::LogConsole::Update() {
    core::utility::log::Log::DefaultLog.FlushLog();
    auto new_log_msg_count = this->echo_log_buffer.log().size();
    if (new_log_msg_count > this->log_msg_count) {
        // Scroll down if new message came in
        this->scroll_down = 3;

        for (size_t i = this->log_msg_count; i < new_log_msg_count; i++) {
            auto entry = this->echo_log_buffer.log()[i];

            // Bring log console to front on new warnings and errors
            if (this->win_log_force_open) {
                if (entry.level == megamol::core::utility::log::Log::log_level::warn ||
                    entry.level == megamol::core::utility::log::Log::log_level::error) {
                    this->win_config.show = true;
                }
            }
        }
    }
    this->log_msg_count = new_log_msg_count;

    return true;
}


bool megamol::gui::LogConsole::Draw() {

    // Scroll down if window height changes
    if (this->last_window_height != ImGui::GetWindowHeight()) {
        this->last_window_height = ImGui::GetWindowHeight();
        this->scroll_down = 3;
    }

    // Menu -------------------------------------------------------------------
    if (ImGui::BeginMenuBar()) {

        // Force Open on Warnings and Errors
        megamol::gui::ButtonWidgets::ToggleButton("Force Open", this->win_log_force_open);
        this->tooltip.Marker("Force open log console window on warnings and errors.");
        ImGui::Separator();

        // Log Level
        ImGui::TextUnformatted("Show Log Level");
        ImGui::SameLine();
        if (ImGui::RadioButton(
                "Error", (this->win_log_level == megamol::core::utility::log::Log::log_level::error ||
                             this->win_log_level == megamol::core::utility::log::Log::log_level::warn ||
                             this->win_log_level == megamol::core::utility::log::Log::log_level::info))) {
            if (this->win_log_level == megamol::core::utility::log::Log::log_level::error ||
                this->win_log_level == megamol::core::utility::log::Log::log_level::warn ||
                this->win_log_level == megamol::core::utility::log::Log::log_level::info) {
                this->win_log_level = megamol::core::utility::log::Log::log_level::none;
            } else {
                this->win_log_level = megamol::core::utility::log::Log::log_level::error;
            }
            this->scroll_down = 3;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton(
                "Warnings", (this->win_log_level == megamol::core::utility::log::Log::log_level::warn ||
                                this->win_log_level == megamol::core::utility::log::Log::log_level::info))) {
            if (this->win_log_level == megamol::core::utility::log::Log::log_level::warn ||
                this->win_log_level == megamol::core::utility::log::Log::log_level::info) {
                this->win_log_level = megamol::core::utility::log::Log::log_level::error;
            } else {
                this->win_log_level = megamol::core::utility::log::Log::log_level::warn;
            }
            this->scroll_down = 3;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Infos", (this->win_log_level == megamol::core::utility::log::Log::log_level::info))) {
            if (this->win_log_level == megamol::core::utility::log::Log::log_level::info) {
                this->win_log_level = megamol::core::utility::log::Log::log_level::warn;
            } else {
                this->win_log_level = megamol::core::utility::log::Log::log_level::info;
            }
            this->scroll_down = 3;
        }

        // Scrolling
        std::string scroll_label = "Scroll";
        ImGui::SameLine(0.0f, ImGui::GetContentRegionAvail().x - (2.25f * ImGui::GetFrameHeightWithSpacing()) -
                                  ImGui::CalcTextSize(scroll_label.c_str()).x);
        ImGui::TextUnformatted(scroll_label.c_str());
        ImGui::SameLine();
        if (ImGui::ArrowButton("scroll_up", ImGuiDir_Up)) {
            this->scroll_up = 2;
        }
        this->tooltip.ToolTip("Scroll to first log entry.");
        ImGui::SameLine();
        if (ImGui::ArrowButton("scroll_down", ImGuiDir_Down)) {
            this->scroll_down = 3;
        }
        this->tooltip.ToolTip("Scroll to last log entry.");

        ImGui::EndMenuBar();
    }

    // Print messages ---------------------------------------------------------
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::BeginChild("log_messages",
        ImVec2(0.0f,
            ImGui::GetWindowHeight() - (3.0f * ImGui::GetFrameHeightWithSpacing()) - (3.0f * style.FramePadding.y)),
        true, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_AlwaysVerticalScrollbar);

    auto message_count = 0ull;
    if (this->win_log_level != megamol::core::utility::log::Log::log_level::none) {
        message_count = this->echo_log_buffer.log().size();
        if (this->win_log_level == megamol::core::utility::log::Log::log_level::warn) {
            message_count = this->echo_log_buffer.warn_log_indices().size();
        } else if (this->win_log_level == megamol::core::utility::log::Log::log_level::error) {
            message_count = this->echo_log_buffer.error_log_indices().size();
        }
    }
    const int modified_count = std::min<int>(static_cast<int>(message_count), 14000000);

    ImGuiListClipper clipper;
    clipper.Begin(modified_count, ImGui::GetTextLineHeight());
    while (clipper.Step()) {
        for (auto row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {

            auto index = static_cast<size_t>(row);
            if (this->win_log_level == megamol::core::utility::log::Log::log_level::warn) {
                index = this->echo_log_buffer.warn_log_indices()[row];
            } else if (this->win_log_level == megamol::core::utility::log::Log::log_level::error) {
                index = this->echo_log_buffer.error_log_indices()[row];
            }
            auto entry = this->echo_log_buffer.log()[index];

            if (entry.level == megamol::core::utility::log::Log::log_level::info) {
                ImGui::TextUnformatted(entry.message.c_str());
            } else if (entry.level == megamol::core::utility::log::Log::log_level::warn) {
                ImGui::TextColored(GUI_COLOR_TEXT_WARN, entry.message.c_str());
            } else if (entry.level == megamol::core::utility::log::Log::log_level::error) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, entry.message.c_str());
            }
        }
    }
    clipper.End();

    // Scroll - Requires 3 frames for being applied!
    if (this->scroll_down > 0) {
        const auto max_offset = 5.0f * ImGui::GetTextLineHeight(); /// XXX IO Why is that neccessary?
        ImGui::SetScrollY(ImGui::GetScrollMaxY() + max_offset);
        this->scroll_down--;
    }
    if (this->scroll_up > 0) {
        ImGui::SetScrollY(0.0f);
        this->scroll_up--;
    }

    ImGui::EndChild();

    // Console Input ----------------------------------------------------------

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Input");
    this->tooltip.Marker("[TAB] Activate autocomplete.\n[Arrow up/down] Browse history of valid lua commands.");
    ImGui::SameLine();
    auto popup_pos = ImGui::GetCursorScreenPos();
    if (this->input_reclaim_focus) {
        ImGui::SetKeyboardFocusHere();
        this->input_reclaim_focus = false;
    }
    std::string hint;
    if (!this->input_shared_data->param_hint.empty()) {
        hint = "Parameter(s): " + this->input_shared_data->param_hint;
    }

    ImGui::PushItemWidth(
        ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(hint.c_str()).x - (2.0f * style.ItemSpacing.x));

    ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                           ImGuiInputTextFlags_CallbackCompletion |
                                           ImGuiInputTextFlags_CallbackHistory | ImGuiInputTextFlags_CallbackAlways;
    if (ImGui::InputText("###console_input", &this->input_buffer, input_text_flags, Input_Text_Callback,
            (void*) this->input_shared_data.get())) {
        std::string command = this->input_buffer;
        auto result = this->luaApi->RunString(command);
        if (result.valid()) {
            // command was fine, no editing required
            auto blah = result.get<std::string>();
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(blah.c_str());
            this->input_shared_data->history.back() = this->input_buffer;
            this->input_shared_data->history.emplace_back("");
            this->input_shared_data->history_index = this->input_shared_data->history.size() - 1;
            this->input_buffer.clear();
        } else {
            auto blah = result.get<std::string>();
            megamol::core::utility::log::Log::DefaultLog.WriteError(blah.c_str());
        }
        this->input_reclaim_focus = true;
    }

    if (!hint.empty()) {
        ImGui::SameLine();
        ImGui::TextUnformatted(hint.c_str());
    }
    ImGui::PopItemWidth();

    std::string popup_id = "autocomplete_selector";
    if (this->input_shared_data->open_autocomplete_popup) {
        ImGui::OpenPopup(popup_id.c_str());
        ImGui::SetNextWindowPos(popup_pos + ImGui::CalcTextSize(this->input_buffer.c_str()) +
                                (ImVec2(2.0f, 0.0f) * style.ItemInnerSpacing));
        ImGui::SetNextWindowFocus();
        this->input_shared_data->open_autocomplete_popup = false;
    }
    if (ImGui::BeginPopup(popup_id.c_str())) {

        int candidates_count = this->input_shared_data->autocomplete_candidates.size();
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            this->selected_candidate_index = (this->selected_candidate_index + 1) % candidates_count;
        } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            this->selected_candidate_index = (candidates_count + this->selected_candidate_index - 1) % candidates_count;
        }
        ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetColorU32(ImGuiCol_HeaderHovered));
        for (int i = 0; i < candidates_count; i++) {
            /// XXX IO ImGui::ScrollToItem(ImGuiScrollFlags_AlwaysCenterY);
            /// Re-implementing behaviour because of globally disabled ImGuiConfigFlags_NavEnableKeyboard
            bool selected_candidate =
                ImGui::Selectable(this->input_shared_data->autocomplete_candidates[i].first.c_str(),
                    (i == this->selected_candidate_index), ImGuiSelectableFlags_AllowDoubleClick);
            if (selected_candidate || ((i == this->selected_candidate_index) && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
                this->input_buffer = this->input_shared_data->autocomplete_candidates[i].first;
                this->input_buffer.append("()");
                this->input_shared_data->autocomplete_candidates.clear();
                this->input_shared_data->move_cursor_to_end = true;
                this->input_reclaim_focus = true;
                this->selected_candidate_index = 0;
                ImGui::CloseCurrentPopup();
                break;
            }
        }
        ImGui::PopStyleColor();

        ImGui::EndPopup();
    }
    // Check if pop-up was closed last frame (needed to detect pop-up closing for reclaiming input focus )
    if (this->is_autocomplete_popup_open && !ImGui::IsPopupOpen(popup_id.c_str())) {
        this->input_reclaim_focus = true;
    }
    this->is_autocomplete_popup_open = ImGui::IsPopupOpen(popup_id.c_str());

    return true;
}


void LogConsole::SetLuaAPI(core::LuaAPI* luaApi) {
    this->luaApi = luaApi;

    if (this->input_shared_data->commands.empty()) {
        auto result = luaApi->RunString("return mmHelp()");
        if (result.valid()) {
            auto res = result.get<std::string>();
            std::regex cmd_regex("mm[A-Z]\\w+(.*)", std::regex_constants::ECMAScript);
            auto cmd_begin = std::sregex_iterator(res.begin(), res.end(), cmd_regex);
            auto cmd_end = std::sregex_iterator();
            for (auto i = cmd_begin; i != cmd_end; ++i) {
                auto match_str = (*i).str();
                auto bracket_pos = match_str.find('(');
                auto command = match_str.substr(0, bracket_pos);
                auto param_hint =
                    match_str.substr(bracket_pos + 1, (match_str.length() - bracket_pos - 2)); // omit brackets
                this->input_shared_data->commands.push_back({command, param_hint});
            }
        }
    }
}


void LogConsole::SpecificStateFromJSON(const nlohmann::json& in_json) {

    for (auto& header_item : in_json.items()) {
        if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
            for (auto& config_item : header_item.value().items()) {
                if (config_item.key() == this->Name()) {
                    auto config_values = config_item.value();

                    auto win_log_level_val = static_cast<unsigned int>(
                        std::underlying_type_t<core::utility::log::Log::log_level>(this->win_log_level));
                    megamol::core::utility::get_json_value<unsigned int>(
                        config_values, {"log_level"}, &win_log_level_val);
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"log_force_open"}, &this->win_log_force_open);
                }
            }
        }
    }
}


void LogConsole::SpecificStateToJSON(nlohmann::json& inout_json) {

    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["log_level"] = this->win_log_level;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["log_force_open"] = this->win_log_force_open;
}
