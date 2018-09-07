/*
*timelog.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*
*/

#ifndef MEGAMOL_CORE_UTILITY_TIMELOG_H_INCLUDED
#define MEGAMOL_CORE_UTILITY_TIMELOG_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/String.h"
#include "mmcore/CoreInstance.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <chrono>


namespace megamol {
namespace core {
namespace utility {

    class MEGAMOLCORE_API timelog {

    public:

        typedef std::chrono::system_clock::time_point time_point;

        enum TIME {
            START   = 0,
            WRITE   = 1,
            REQUEST = 2,
        };

        timelog();

        ~timelog(void);

        bool clear(vislib::StringA fn);

        void set_filename(vislib::StringA fn) {
            this->filename_ = fn;
        }

        double delta_time(TIME t) const;

        bool write_time_line(vislib::StringA line);

        bool write_line(vislib::StringA line);

        void set_time(TIME t) {
            time_point ct = std::chrono::system_clock::now();
            switch (t) {
                case (TIME::START):   this->start_time_   = ct; break;
                case (TIME::WRITE):   this->write_time_   = ct; break;
                case (TIME::REQUEST): this->request_time_ = ct; break;
                default: break;
            }
        }

    private:

        bool open_file(void);

        bool close_file(void);

// Disable dll export warning for not exported classes in ::vislib and ::std 
#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */

        vislib::StringA    filename_;

        time_point         start_time_;
        time_point         request_time_;
        time_point         write_time_;

        std::ofstream      logfile_;

#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_CORE_UTILITY_TIMELOG_H_INCLUDED */