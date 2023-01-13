/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace megamol::core::utility::string {

static inline std::vector<std::string> Split(std::string const& s, char delim) {
    std::stringstream stream(s);
    std::vector<std::string> segments;
    while (stream.good()) {
        std::string str;
        std::getline(stream, str, delim);
        segments.push_back(str);
    }
    return segments;
}

static inline std::string LeftTrim(std::string s) {
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), [](unsigned char ch) { return std::isspace(ch); }));
    return s;
}

static inline std::string RightTrim(std::string s) {
    s.erase(std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch) { return std::isspace(ch); }).base(), s.end());
    return s;
}

static inline std::string Trim(std::string s) {
    return LeftTrim(RightTrim(std::move(s)));
}

static inline bool EqualAsciiCaseInsensitive(std::string const& a, std::string const& b) {
    if (a.length() != b.length()) {
        return false;
    }
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char a, char b) { return tolower(a) == tolower(b); });
}

static inline bool IsBooleanTrueString(std::string const& v) {
    return EqualAsciiCaseInsensitive(v, "true") || EqualAsciiCaseInsensitive(v, "t") ||
           EqualAsciiCaseInsensitive(v, "yes") || EqualAsciiCaseInsensitive(v, "y") ||
           EqualAsciiCaseInsensitive(v, "on") || EqualAsciiCaseInsensitive(v, "1");
}

static inline bool IsBooleanFalseString(std::string const& v) {
    return EqualAsciiCaseInsensitive(v, "false") || EqualAsciiCaseInsensitive(v, "f") ||
           EqualAsciiCaseInsensitive(v, "no") || EqualAsciiCaseInsensitive(v, "n") ||
           EqualAsciiCaseInsensitive(v, "off") || EqualAsciiCaseInsensitive(v, "0");
}

static inline bool ParseBool(std::string const& v) {
    if (IsBooleanTrueString(v)) {
        return true;
    }
    if (IsBooleanFalseString(v)) {
        return false;
    }
    try {
        int i = std::stoi(v);
        return i != 0;
    } catch (...) {
        throw std::invalid_argument("Invalid boolean string value!");
    }
}

} // namespace megamol::core::utility::string
