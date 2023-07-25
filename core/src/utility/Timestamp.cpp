/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/Timestamp.h"

#include <sstream>

std::string megamol::core::utility::serialize_timestamp(std::chrono::system_clock::time_point const& tp) {
    auto const t = std::chrono::system_clock::to_time_t(tp);
    auto const lt = std::localtime(&t);
    auto const fs = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count() % 1000;
    std::stringstream str;
    str << std::to_string(1900 + lt->tm_year) << "-" << std::to_string(1 + lt->tm_mon) << "-"
        << std::to_string(lt->tm_mday) << "T" << std::to_string(lt->tm_hour) << ":" << std::to_string(lt->tm_min) << ":"
        << std::to_string(lt->tm_sec) << "." << std::to_string(fs);

    return str.str();
}
