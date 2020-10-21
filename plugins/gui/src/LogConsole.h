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


    class LogConsole {
    public:
        /**
         * CTOR.
         */
        LogConsole();

        /**
         * DTOR.
         */
        virtual ~LogConsole();

        /**
         * Draw log console window.
         */
        bool Draw(WindowCollection::WindowConfiguration& wc);

        /**
         * Update log.
         */
        bool Update(WindowCollection::WindowConfiguration& wc);

    private:
        // DATA TYPES ---------------------------------------------------------

        class LogBuffer : public std::stringbuf {
        public:
            virtual int sync() {
                try {
                    auto message_str = this->str();
                    // Split message string
                    auto split_index = message_str.find("\n");
                    while (split_index != std::string::npos) {
                        auto new_message = message_str.substr(0, split_index + 1);
                        this->messages.push_back(new_message);
                        message_str = message_str.substr(split_index + 1);
                        split_index = message_str.find("\n");
                    }
                    this->updated = true;
                    this->str("");
                } catch (...) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Log Console Buffer Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                    return 1;
                };
                return 0;
            }

            bool ConsumeMessage(std::vector<std::string>& msg) {
                if (this->updated) {
                    msg = this->messages;
                    this->messages.clear();
                    this->updated = false;
                    return true;
                }
                return false;
            }

        private:
            bool updated = false;
            std::vector<std::string> messages;
        };

        struct LogEntry {
            unsigned int level;
            std::string message;
        };

        // VARIABLES --------------------------------------------------------------

        LogBuffer echo_log_buffer;
        std::ostream echo_log_stream;
        std::shared_ptr<megamol::core::utility::log::StreamTarget> echo_log_target;

        std::vector<LogEntry> log;
        unsigned int log_level;

        unsigned int scroll_log_down;
        unsigned int scroll_log_up;
        float last_window_height;

        // Widgets
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool connect_log(void);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_LOGCONSOLE_H_INCLUDED
