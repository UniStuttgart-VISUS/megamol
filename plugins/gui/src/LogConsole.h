/*
 * LogConsole.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
#define MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
#pragma once


#include "WindowCollection.h"
#include "mmcore/utility/log/StreamTarget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"


namespace megamol {
namespace gui {

    /*
     * The log buffer collecting all the messages
     */
    class LogBuffer : public std::stringbuf {
    public:
        LogBuffer() = default;
        ~LogBuffer() = default;

        struct LogEntry {
            unsigned int level;
            std::string message;
        };

        int sync(void);

        inline const std::vector<LogEntry>& log(void) const {
            return this->messages;
        }

    private:
        std::vector<LogEntry> messages;
    };


    /*
     * The content of the log cnsole GUI window
     */
    class LogConsole {
    public:
        LogConsole();
        ~LogConsole();

        void Update(WindowConfiguration& wc);

        bool Draw(WindowConfiguration& wc);

        void PopUps(void);

    private:
        // VARIABLES --------------------------------------------------------------

        LogBuffer echo_log_buffer;
        std::ostream echo_log_stream;
        std::shared_ptr<megamol::core::utility::log::StreamTarget> echo_log_target;

        size_t log_msg_count;
        unsigned int scroll_down;
        unsigned int scroll_up;
        float last_window_height;
        std::string window_title;

        struct LogPopUpData {
            std::string title;
            bool disable;
            bool show;
            std::vector<LogBuffer::LogEntry> entries;
        };
        std::vector<LogPopUpData> log_popups;

        // Widgets
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool connect_log(void);

        void print_message(LogBuffer::LogEntry entry, unsigned int global_log_level) const;

        void draw_popup(LogPopUpData& log_popup);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
