/*
 * LogConsole.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "LogConsole.h"
#include "imgui_stdlib.h"
#include "mmcore/utility/log/OfflineTarget.h"
#include "widgets/ButtonWidgets.h"

#include <regex>

using namespace megamol::gui;


namespace megamol {
namespace gui {

int Input_Text_Callback(ImGuiInputTextCallbackData* data) {

    /// megamol::core::utility::log::Log::DefaultLog.WriteInfo("[LogConsole] DEBUG: cursor: %d, selection: %d-%d",
    /// data->CursorPos, data->SelectionStart, data->SelectionEnd);
    auto user_data = static_cast<megamol::gui::LogConsole::InputSharedData*>(data->UserData);
    if (user_data == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[LogConsole] Unable to get pointer to user data.");
    }

    switch (data->EventFlag) {
    case ImGuiInputTextFlags_CallbackAlways: {
        if (user_data->autocomplete_appended) {
            data->CursorPos = data->BufTextLen;
            user_data->autocomplete_appended = false;
        }
        break;
    }
    case ImGuiInputTextFlags_CallbackCompletion: {
        // Example of TEXT COMPLETION

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
        user_data->autocomplete_candidates.clear();
        for (int i = 0; i < user_data->commands.size(); i++)
            if (strncmp(user_data->commands[i].data(), word_start, (int)(word_end - word_start)) == 0)
                user_data->autocomplete_candidates.push_back(user_data->commands[i]);

        if (user_data->autocomplete_candidates.size() == 0) {
            // No match
            // AddLog("No match for \"%.*s\"!\n", (int) (word_end - word_start), word_start);
        } else if (user_data->autocomplete_candidates.size() == 1) {
            // Single match. Delete the beginning of the word and replace it entirely so we've got nice casing.
            data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
            data->InsertChars(data->CursorPos, user_data->autocomplete_candidates[0].data());
            data->InsertChars(data->CursorPos, "(");
        } else {
            // Multiple matches. Complete as much as we can..
            // So inputing "C"+Tab will complete to "CL" then display "CLEAR" and "CLASSIFY" as matches.
            int match_len = (int)(word_end - word_start);
            for (;;) {
                int c = 0;
                bool all_candidates_matches = true;
                for (int i = 0; i < user_data->autocomplete_candidates.size() && all_candidates_matches; i++)
                    if (i == 0)
                        c = toupper(user_data->autocomplete_candidates[i][match_len]);
                    else if (c == 0 || c != toupper(user_data->autocomplete_candidates[i][match_len]))
                        all_candidates_matches = false;
                if (!all_candidates_matches)
                    break;
                match_len++;
            }

            if (match_len > 0) {
                data->DeleteChars((int)(word_start - data->Buf), (int)(word_end - word_start));
                data->InsertChars(data->CursorPos, user_data->autocomplete_candidates[0].data(),
                    user_data->autocomplete_candidates[0].data() + match_len);
            }

            // List matches
            // AddLog("Possible matches:\n");
            // for (int i = 0; i < candidates.Size; i++)
            // AddLog("- %s\n", candidates[i]);
        }

        user_data->open_autocomplete_popup = (user_data->autocomplete_candidates.size() > 1);
        break;
    }
    case ImGuiInputTextFlags_CallbackHistory: {
        // Example of HISTORY

        // const int prev_history_pos = HistoryPos;
        // if (data->EventKey == ImGuiKey_UpArrow) {
        //    if (HistoryPos == -1)
        //        HistoryPos = History.Size - 1;
        //    else if (HistoryPos > 0)
        //        HistoryPos--;
        //} else if (data->EventKey == ImGuiKey_DownArrow) {
        //    if (HistoryPos != -1)
        //        if (++HistoryPos >= History.Size)
        //            HistoryPos = -1;
        //}

        //// A better implementation would preserve the data on the current input line along with cursor position.
        // if (prev_history_pos != HistoryPos) {
        //    const char* history_str = (HistoryPos >= 0) ? History[HistoryPos] : "";
        //    data->DeleteChars(0, data->BufTextLen);
        //    data->InsertChars(0, history_str);
        //}
        break;
    }
    default:
        break;
    }
    return 0;
}

} // namespace gui
} // namespace megamol


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
                    unsigned int log_level = megamol::core::utility::log::Log::LEVEL_NONE;
                    auto level_str = new_message.substr(0, seperator_index);
                    log_level = std::stoi(level_str);
                    std::istringstream(level_str) >> log_level; // 0 if failed = LEVEL_NONE
                    if (log_level != megamol::core::utility::log::Log::LEVEL_NONE) {
                        this->messages.push_back({log_level, new_message});
                        extracted_new_message = true;
                    }
                }
                if (!extracted_new_message) {
                    // Append to previous message
                    this->messages.back().message.append(new_message);
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
        , echo_log_target(nullptr)
        , log_msg_count(0)
        , scroll_down(2)
        , scroll_up(0)
        , last_window_height(0.0f)
        , win_log_level(static_cast<int>(megamol::core::utility::log::Log::LEVEL_ALL))
        , win_log_force_open(true)
        , tooltip()
        , input_shared_data(nullptr)
        , input_reclaim_focus(false)
        , input_buffer()
        , input_lua_func(nullptr) {

    this->echo_log_target = std::make_shared<megamol::core::utility::log::StreamTarget>(
        this->echo_log_stream, megamol::core::utility::log::Log::LEVEL_ALL);
    this->connect_log();

    // Configure CONSOLE Window
    this->win_config.size = ImVec2(500.0f * megamol::gui::gui_scaling.Get(), 50.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F9, core::view::Modifier::NONE);

    // Initialise
    this->input_shared_data = std::make_shared<InputSharedData>();
    this->input_shared_data->open_autocomplete_popup = false;
    this->input_shared_data->autocomplete_appended = false;
}


LogConsole::~LogConsole() {

    // Reset echo target only if log target of this class instance is used
    if (megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget() == this->echo_log_target) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(nullptr);
    }
    this->echo_log_target.reset();
}


bool megamol::gui::LogConsole::Update() {

    auto new_log_msg_count = this->echo_log_buffer.log().size();
    if (new_log_msg_count > this->log_msg_count) {
        // Scroll down if new message came in
        this->scroll_down = 3;

        for (size_t i = this->log_msg_count; i < new_log_msg_count; i++) {
            auto entry = this->echo_log_buffer.log()[i];

            // Bring log console to front on new warnings and errors
            if (this->win_log_force_open) {
                if (entry.level < megamol::core::utility::log::Log::LEVEL_INFO) {
                    if (this->win_log_level < megamol::core::utility::log::Log::LEVEL_WARN) {
                        this->win_log_level = megamol::core::utility::log::Log::LEVEL_WARN;
                    }
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
        ImGui::TextUnformatted("Show");
        ImGui::SameLine();
        if (ImGui::RadioButton("Errors", (this->win_log_level >= megamol::core::utility::log::Log::LEVEL_ERROR))) {
            if (this->win_log_level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
                this->win_log_level = megamol::core::utility::log::Log::LEVEL_NONE;
            } else {
                this->win_log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            }
            this->scroll_down = 3;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Warnings", (this->win_log_level >= megamol::core::utility::log::Log::LEVEL_WARN))) {
            if (this->win_log_level >= megamol::core::utility::log::Log::LEVEL_WARN) {
                this->win_log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            } else {
                this->win_log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            }
            this->scroll_down = 3;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Infos", (this->win_log_level == megamol::core::utility::log::Log::LEVEL_ALL))) {
            if (this->win_log_level == megamol::core::utility::log::Log::LEVEL_ALL) {
                this->win_log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            } else {
                this->win_log_level = megamol::core::utility::log::Log::LEVEL_ALL;
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

    for (auto& entry : this->echo_log_buffer.log()) {
        this->print_message(entry, this->win_log_level);
    }

    // Scroll - Requires 3 frames for being applied!
    if (this->scroll_down > 0) {
        ImGui::SetScrollY(ImGui::GetScrollMaxY());
        this->scroll_down--;
    }
    if (this->scroll_up > 0) {
        ImGui::SetScrollY(0.0f);
        this->scroll_up--;
    }

    ImGui::EndChild();

    // Console Input ----------------------------------------------------------

    // Lazy init of command list through lua interface itself
    if (this->input_shared_data->commands.size() == 0) {
        auto result = (*this->input_lua_func)("return mmHelp()");
        if (std::get<0>(result)) {
            auto res = std::get<1>(result);
            std::regex cmd_regex("mm\\w+", std::regex_constants::ECMAScript);
            auto cmd_begin = std::sregex_iterator(res.begin(), res.end(), cmd_regex);
            auto cmd_end = std::sregex_iterator();
            for (auto i = cmd_begin; i != cmd_end; ++i) {
                auto match = *i;
                this->input_shared_data->commands.push_back(match.str());
            }
        }
    }

    if (this->input_reclaim_focus) {
        ImGui::SetKeyboardFocusHere();
        this->input_reclaim_focus = false;
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Input");
    this->tooltip.Marker("Confirm autocomplete selection with 'Space'.");
    ImGui::SameLine();
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeightWithSpacing());
    auto popup_pos = ImGui::GetCursorScreenPos();
    ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                           ImGuiInputTextFlags_CallbackCompletion |
                                           ImGuiInputTextFlags_CallbackHistory | ImGuiInputTextFlags_CallbackAlways;
    if (ImGui::InputText("###console_input", &this->input_buffer, input_text_flags, Input_Text_Callback,
            (void*)this->input_shared_data.get())) {
        std::string command = "return " + this->input_buffer;
        auto result = (*this->input_lua_func)(command.c_str());
        if (std::get<0>(result)) {
            // command was fine, no editing required
            auto blah = std::get<1>(result);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(blah.c_str());
            this->input_buffer.clear();
        } else {
            auto blah = std::get<1>(result);
            megamol::core::utility::log::Log::DefaultLog.WriteError(blah.c_str());
        }
        this->input_reclaim_focus = true;
    }
    ImGui::PopItemWidth();

    if (this->input_shared_data->open_autocomplete_popup) {
        ImGui::OpenPopup("autocomplete_selector");
        ImGui::SetNextWindowPos(popup_pos + ImGui::CalcTextSize(this->input_buffer.c_str()) + style.ItemInnerSpacing);
        ImGui::SetItemDefaultFocus();
        this->input_shared_data->open_autocomplete_popup = false;
    }
    if (ImGui::BeginPopup("autocomplete_selector")) {
        bool selected_candidate = false;
        for (int i = 0; i < this->input_shared_data->autocomplete_candidates.size(); i++) {
            if (ImGui::Selectable(this->input_shared_data->autocomplete_candidates[i].data())) {
                this->input_buffer = this->input_shared_data->autocomplete_candidates[i];
                this->input_buffer.append("(");
                this->input_shared_data->autocomplete_candidates.clear();
                this->input_shared_data->autocomplete_appended = true;
                selected_candidate = true;
            }
        }

        if (selected_candidate || ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
            this->input_reclaim_focus = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    return true;
}


bool megamol::gui::LogConsole::connect_log() {

    auto current_echo_target = megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget();
    std::shared_ptr<megamol::core::utility::log::OfflineTarget> offline_echo_target =
        std::dynamic_pointer_cast<megamol::core::utility::log::OfflineTarget>(current_echo_target);

    // Only connect if echo target is still default OfflineTarget
    /// Note: A second log console is temporarily created when "GUIView" module is loaded in configurator for complete
    /// module list. For this "GUIView" module NO log is connected, because the main LogConsole instance is already
    /// connected and the taget is not the default OfflineTarget.
    if ((offline_echo_target != nullptr) && (this->echo_log_target != nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(this->echo_log_target);
        megamol::core::utility::log::Log::DefaultLog.SetEchoLevel(megamol::core::utility::log::Log::LEVEL_ALL);
    }

    return true;
}


void megamol::gui::LogConsole::print_message(const LogBuffer::LogEntry& entry, unsigned int global_log_level) const {

    if (entry.level <= global_log_level) {
        if (entry.level >= megamol::core::utility::log::Log::LEVEL_INFO) {
            ImGui::TextUnformatted(entry.message.c_str());
        } else if (entry.level >= megamol::core::utility::log::Log::LEVEL_WARN) {
            ImGui::TextColored(GUI_COLOR_TEXT_WARN, entry.message.c_str());
        } else if (entry.level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
            ImGui::TextColored(GUI_COLOR_TEXT_ERROR, entry.message.c_str());
        }
    }
}


void LogConsole::SpecificStateFromJSON(const nlohmann::json& in_json) {

    for (auto& header_item : in_json.items()) {
        if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
            for (auto& config_item : header_item.value().items()) {
                if (config_item.key() == this->Name()) {
                    auto config_values = config_item.value();

                    megamol::core::utility::get_json_value<unsigned int>(
                        config_values, {"log_level"}, &this->win_log_level);
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
