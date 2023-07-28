/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/DateTime.h"

#include <ctime>

std::string megamol::core::utility::DateTime::CurrentDateTimeFormatted(const std::string& format) {
    time_t nowtime;
    char buffer[1024];
    struct tm* now;
    time(&nowtime);
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
    struct tm nowdata;
    now = &nowdata;
    localtime_s(now, &nowtime);
#else  /* defined(_WIN32) && (_MSC_VER >= 1400) */
    now = localtime(&nowtime);
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
    strftime(buffer, 1024, format.c_str(), now);
    std::string ret(buffer);
    return ret;
}
