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


megamol::gui::LogConsole::LogConsole()
        : echo_log_buffer()
        , echo_log_stream(&this->echo_log_buffer)
        , echo_log_target(nullptr)
        , log()
        , log_level(megamol::core::utility::log::Log::LEVEL_ALL)
        , scroll_log_down(2)
        , scroll_log_up(0)
        , tooltip() {

    this->echo_log_target =
        std::make_shared<megamol::core::utility::log::StreamTarget>(this->echo_log_stream, this->log_level);

    /// Note: Do not connect log in ctor, because otherwise temporary "GUIView" module loading in configurator would
    /// overwrite echo target.
}


LogConsole::~LogConsole() {

    // Reset echo target only if log target of this class instance is used
    if (megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget() == this->echo_log_target) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(
            std::make_shared<megamol::core::utility::log::OfflineTarget>(
                100, megamol::core::utility::log::Log::LEVEL_ALL));
    }
    this->echo_log_target.reset();
}


bool megamol::gui::LogConsole::Draw(WindowCollection::WindowConfiguration& wc) {

    // Menu
    if (ImGui::BeginMenuBar()) {
        // Log Level
        ImGui::TextUnformatted("Show");
        ImGui::SameLine();
        if (ImGui::RadioButton("Errors", (this->log_level >= megamol::core::utility::log::Log::LEVEL_ERROR))) {
            if (this->log_level >= megamol::core::utility::log::Log::LEVEL_ERROR) {
                this->log_level = megamol::core::utility::log::Log::LEVEL_NONE;
            } else {
                this->log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Warnings", (this->log_level >= megamol::core::utility::log::Log::LEVEL_WARN))) {
            if (this->log_level >= megamol::core::utility::log::Log::LEVEL_WARN) {
                this->log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
            } else {
                this->log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            }
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Infos", (this->log_level == megamol::core::utility::log::Log::LEVEL_ALL))) {
            if (this->log_level == megamol::core::utility::log::Log::LEVEL_ALL) {
                this->log_level = megamol::core::utility::log::Log::LEVEL_WARN;
            } else {
                this->log_level = megamol::core::utility::log::Log::LEVEL_ALL;
            }
        }

        // Scrolling
        std::string scroll_label = "Scroll";
        ImGui::SameLine(0.0f, ImGui::GetContentRegionAvail().x - (2.25f * ImGui::GetFrameHeightWithSpacing()) - ImGui::CalcTextSize(scroll_label.c_str()).x);
        ImGui::TextUnformatted(scroll_label.c_str());
        ImGui::SameLine();
        if (ImGui::ArrowButton("scroll_up", ImGuiDir_Up)) {
            this->scroll_log_up = 2;
        }
        this->tooltip.ToolTip("Scroll to first log entry.");
        ImGui::SameLine();
        if (ImGui::ArrowButton("scroll_down", ImGuiDir_Down)) {
            this->scroll_log_down = 2;
        }
        this->tooltip.ToolTip("Scroll to last log entry.");

        ImGui::EndMenuBar();
    }

    // Scroll
    if (this->scroll_log_down > 0) {
        ImGui::SetScrollY(ImGui::GetScrollMaxY());
        this->scroll_log_down--;
    }
    if (this->scroll_log_up > 0) {
        ImGui::SetScrollY(0);
        this->scroll_log_up--;
    }

    // Print messages
    for (auto& entry : this->log) {
        if (entry.level <= this->log_level) {
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


bool megamol::gui::LogConsole::Update(WindowCollection::WindowConfiguration& wc) {

    this->connect_log();

    // Get new messages
    bool updated = false;
    std::vector<std::string> new_messages;
    if (this->echo_log_buffer.ConsumeMessage(new_messages)) {
        for (auto& msg : new_messages) {
            // Assuming one log message of format "<level>|<message>\r\n"
            auto seperator_index = msg.find("|");
            if (seperator_index != std::string::npos) {
                unsigned int new_log_level = megamol::core::utility::log::Log::LEVEL_NONE;
                auto level_str = msg.substr(0, seperator_index);
                try {
                    new_log_level = std::stoi(level_str);
                } catch (...) {}
                if (new_log_level != megamol::core::utility::log::Log::LEVEL_NONE) {
                    LogEntry new_entry;
                    new_entry.level = new_log_level;
                    new_entry.message = msg;
                    this->log.push_back(new_entry);

                    // Force open log window if there is an error
                    if (new_log_level == megamol::core::utility::log::Log::LEVEL_ERROR) {
                        this->log_level = megamol::core::utility::log::Log::LEVEL_ERROR;
                        wc.win_show = true;
                    }

                    this->scroll_log_down = 2;
                    updated = true;
                }
            }
        }
    }
    return updated;
}


bool megamol::gui::LogConsole::connect_log(void) {

    auto current_echo_target = megamol::core::utility::log::Log::DefaultLog.AccessEchoTarget();
    std::shared_ptr<megamol::core::utility::log::OfflineTarget> offline_echo_target =
        std::dynamic_pointer_cast<megamol::core::utility::log::OfflineTarget>(current_echo_target);

    if ((offline_echo_target != nullptr) && (this->echo_log_target != nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.SetEchoTarget(this->echo_log_target);
    }

    return true;
}
