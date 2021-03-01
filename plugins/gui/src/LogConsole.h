/*
 * LogConsole.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
#define MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED


#include "GUIUtils.h"
#include "WindowCollection.h"
#include "widgets/HoverToolTip.h"

#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/log/OfflineTarget.h"
#include "mmcore/utility/log/StreamTarget.h"


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

        /**
         * Draw log console window.
         */
        bool Draw(WindowCollection::WindowConfiguration& wc);

        void Update(WindowCollection::WindowConfiguration& wc);

    private:
        // VARIABLES --------------------------------------------------------------

        LogBuffer echo_log_buffer;
        std::ostream echo_log_stream;
        std::shared_ptr<megamol::core::utility::log::StreamTarget> echo_log_target;

        size_t log_msg_count;
        unsigned int scroll_down;
        unsigned int scroll_up;
        float last_window_height;

        // Widgets
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool connect_log(void);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
