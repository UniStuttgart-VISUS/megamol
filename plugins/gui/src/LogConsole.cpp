/*
 * LogConsole.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "LogConsole.h"


using namespace megamol;
using namespace megamol::gui;


int megamol::gui::LogBuffer::sync(void) {
    try {
        auto message_str = this->str();
        if (!message_str.empty()) {
            // Split message string
            auto split_index = message_str.find("\n");
            while (split_index != std::string::npos) {
                // Assuming new line of log message of format "<level>|<message>\r\n"
                auto new_message = message_str.substr(0, split_index + 1);
                bool extracted_new_message = false;
                auto seperator_index = new_message.find("|");
                if (seperator_index != std::string::npos) {
                    unsigned int log_level = megamol::core::utility::log::Log::LEVEL_NONE;
                    auto level_str = new_message.substr(0, seperator_index);
                    try {
                        log_level = std::stoi(level_str);
                    } catch (...) {}
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
                split_index = message_str.find("\n");
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


megamol::gui::LogConsole::LogConsole()
        : echo_log_buffer()
        , echo_log_stream(&this->echo_log_buffer)
        , echo_log_target(nullptr)
        , log_msg_count(0)
        , scroll_down(2)
        , scroll_up(0)
        , last_window_height(0.0f)
        , window_title()
        , log_popups()
        , tooltip() {

    this->echo_log_target = std::make_shared<megamol::core::utility::log::StreamTarget>(
        this->echo_log_stream, megamol::core::utility::log::Log::LEVEL_ALL);
    this->connect_log();
}


LogConsole::~LogConsole() {

    // Reset echo target only if log target of this class instance is used
    if (megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget() == this->echo_log_target) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(nullptr);
    }
    this->echo_log_target.reset();
}


void megamol::gui::LogConsole::Update(WindowCollection::WindowConfiguration& wc) {

    this->window_title = wc.win_name;

    auto new_log_msg_count = this->echo_log_buffer.log().size();
    if (new_log_msg_count > this->log_msg_count) {
        // Scroll down if new message came in
        this->scroll_down = 2;

        for (size_t i = this->log_msg_count; i < new_log_msg_count; i++) {
            auto entry = this->echo_log_buffer.log()[i];

            // Bring log console to front on new warnings and errors
            if (wc.log_force_open) {
                if (entry.level < megamol::core::utility::log::Log::LEVEL_INFO) {
                    if (wc.log_level < megamol::core::utility::log::Log::LEVEL_WARN) {
                        wc.log_level = megamol::core::utility::log::Log::LEVEL_WARN;
                    }
                    wc.win_show = true;
                }
            }

            // Check for tags indicating log message pop-up
            auto start_tag_pos = entry.message.find(LOGMESSAGE_GUI_POPUP_START_TAG);
            auto end_tag_pos = entry.message.find(LOGMESSAGE_GUI_POPUP_END_TAG, start_tag_pos);
            if ((start_tag_pos != std::string::npos) && (end_tag_pos != std::string::npos)) {
                start_tag_pos += std::string(LOGMESSAGE_GUI_POPUP_START_TAG).length();
                auto title = entry.message.substr(start_tag_pos, (end_tag_pos - start_tag_pos));
                entry.message = entry.message.substr(end_tag_pos + std::string(LOGMESSAGE_GUI_POPUP_END_TAG).length());
                // Append to existing log pop-up with same title
                bool found_existing_popup = false;
                for (auto& log_popup : this->log_popups) {
                    if (log_popup.title == title) {
                        log_popup.show = true;
                        log_popup.entries.push_back(entry);
                        found_existing_popup = true;
                        break;
                    }
                }
                // .. else create new pop-up
                if (!found_existing_popup) {
                    LogPopUpData log_popup;
                    log_popup.title = title;
                    log_popup.show = true;
                    log_popup.disable = false;
                    log_popup.entries.push_back(entry);
                    this->log_popups.push_back(log_popup);
                }
            }
        }
    }
    this->log_msg_count = new_log_msg_count;
}


bool megamol::gui::LogConsole::Draw(WindowCollection::WindowConfiguration& wc) {

    // Scroll down if window height changes
    if (this->last_window_height != ImGui::GetWindowHeight()) {
        this->last_window_height = ImGui::GetWindowHeight();
        this->scroll_down = 2;
    }

    // Menu -------------------------------------------------------------------
    if (ImGui::BeginMenuBar()) {

        // Force Open on Warnings and Errors
        ImGui::Checkbox("Force Open", &wc.log_force_open);
        this->tooltip.Marker("Force open log console window on warnings and errors.");
        ImGui::Separator();

        // Log Level
        ImGui::TextUnformatted("Show");
        ImGui::SameLine();
        if (ImGui::RadioButton("Errors", (wc.log_level >= megamol::core::utility::log::Log::LEVEL_ERROR))) {
            if (wc.log_level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
                wc.log_level = megamol::core::utility::log::Log::LEVEL_NONE;
            } else {
                wc.log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            }
            this->scroll_down = 2;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Warnings", (wc.log_level >= megamol::core::utility::log::Log::LEVEL_WARN))) {
            if (wc.log_level >= megamol::core::utility::log::Log::LEVEL_WARN) {
                wc.log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            } else {
                wc.log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            }
            this->scroll_down = 2;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Infos", (wc.log_level == megamol::core::utility::log::Log::LEVEL_ALL))) {
            if (wc.log_level == megamol::core::utility::log::Log::LEVEL_ALL) {
                wc.log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            } else {
                wc.log_level = megamol::core::utility::log::Log::LEVEL_ALL;
            }
            this->scroll_down = 2;
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
            this->scroll_down = 2;
        }
        this->tooltip.ToolTip("Scroll to last log entry.");

        ImGui::EndMenuBar();
    }
    // Scroll - Requires 2 frames for being applied!
    if (this->scroll_down > 0) {
        ImGui::SetScrollY(ImGui::GetScrollMaxY());
        this->scroll_down--;
    }
    if (this->scroll_up > 0) {
        ImGui::SetScrollY(0);
        this->scroll_up--;
    }

    // Print messages ---------------------------------------------------------
    for (auto& entry : this->echo_log_buffer.log()) {
        this->print_message(entry, wc.log_level);
    }

    return true;
}


void megamol::gui::LogConsole::PopUps() {

    for (auto& log_popup : log_popups) {
        this->draw_popup(log_popup);
    }
}


bool megamol::gui::LogConsole::connect_log(void) {

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


void megamol::gui::LogConsole::print_message(LogBuffer::LogEntry entry, unsigned int global_log_level) const {
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


void megamol::gui::LogConsole::draw_popup(LogPopUpData& log_popup) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiStyle& style = ImGui::GetStyle();

    std::string popup_title = this->window_title + " -  [" + log_popup.title + "]";

    if ((!log_popup.disable && log_popup.show) && !ImGui::IsPopupOpen(popup_title.c_str())) {
        ImGui::OpenPopup(popup_title.c_str());
    }

    if (ImGui::BeginPopupModal(popup_title.c_str(), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar)) {

        for (auto& entry : log_popup.entries) {
            this->print_message(entry, megamol::core::utility::log::Log::LEVEL_ALL);
            ImGui::Separator();
        }

        bool close = false;
        if (ImGui::Button("Ok")) {
            close = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Ok, disable further notification.")) {
            log_popup.disable = true;
            close = true;
        }
        if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
            close = true;
        }
        if (close) {
            log_popup.entries.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    log_popup.show = false;
}
