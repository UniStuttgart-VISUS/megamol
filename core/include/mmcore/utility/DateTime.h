/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>

namespace megamol::core::utility {

class DateTime {
public:
    static std::string CurrentDateFormatted() {
        return CurrentDateTimeFormatted("%#x");
    }

    static std::string CurrentTimeFormatted() {
        return CurrentDateTimeFormatted("%#X");
    }

    static std::string CurrentDateTimeFormatted(const std::string& format = "%#c");
};

} // namespace megamol::core::utility
