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
                unsigned int log_level = megamol::core::utility::log::Log::LEVEL_NONE;
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


bool megamol::gui::LogConsole::Draw(WindowCollection::WindowConfiguration& wc) {

    // Scroll down if window height changes
    if (this->last_window_height != ImGui::GetWindowHeight()) {
        this->last_window_height = ImGui::GetWindowHeight();
        this->scroll_down = 2;
    }

    // Menu
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

    // Print messages
    for (auto& entry : this->echo_log_buffer.log()) {
        if (entry.level <= wc.log_level) {
            if (entry.level >= megamol::core::utility::log::Log::LEVEL_INFO) {
                ImGui::TextUnformatted(entry.message.c_str());
            } else if (entry.level >= megamol::core::utility::log::Log::LEVEL_WARN) {
                ImGui::TextColored(GUI_COLOR_TEXT_WARN, entry.message.c_str());
            } else if (entry.level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, entry.message.c_str());
            }
        }
    }

    return true;
}


void megamol::gui::LogConsole::Update(WindowCollection::WindowConfiguration& wc) {

    auto new_log_msg_count = this->echo_log_buffer.log().size();
    if (new_log_msg_count > this->log_msg_count) {
        // Scroll down if new message came in
        this->scroll_down = 2;
        // Bring log console to front on new warnings and errors
        if (wc.log_force_open) {
            for (size_t i = this->log_msg_count; i < new_log_msg_count; i++) {
                auto entry = this->echo_log_buffer.log()[i];
                if (wc.log_level < megamol::core::utility::log::Log::LEVEL_INFO) {
                    wc.log_level = megamol::core::utility::log::Log::LEVEL_WARN;
                    wc.win_show = true;
                }
            }
        }
    }
    this->log_msg_count = new_log_msg_count;
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
    }

    return true;
}
