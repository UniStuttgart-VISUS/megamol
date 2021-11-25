#pragma once

#include <string>

namespace megamol {
namespace core {
namespace utility {

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

} // namespace utility
} // namespace core
} // namespace megamol
