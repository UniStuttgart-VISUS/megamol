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

static inline void LeftTrim(std::string& s) {
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), [](unsigned char ch) { return std::isspace(ch); }));
}

static inline void RightTrim(std::string& s) {
    s.erase(std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch) { return std::isspace(ch); }).base(), s.end());
}

static inline void Trim(std::string& s) {
    LeftTrim(s);
    RightTrim(s);
}

static inline std::string LeftTrimCopy(std::string s) {
    LeftTrim(s);
    return s;
}

static inline std::string RightTrimCopy(std::string s) {
    RightTrim(s);
    return s;
}

static inline std::string TrimCopy(std::string s) {
    Trim(s);
    return s;
}

static inline void ToLowerAscii(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
}

static inline std::string ToLowerAsciiCopy(std::string s) {
    ToLowerAscii(s);
    return s;
}

static inline void ToUpperAscii(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
}

static inline std::string ToUpperAsciiCopy(std::string s) {
    ToUpperAscii(s);
    return s;
}

static inline bool EqualAsciiCaseInsensitive(std::string const& a, std::string const& b) {
    if (a.length() != b.length()) {
        return false;
    }
    return std::equal(
        a.begin(), a.end(), b.begin(), b.end(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

static inline bool StartsWith(std::string const& str, std::string const& prefix) {
    return str.rfind(prefix, 0) == 0;
}

static inline bool StartsWithAsciiCaseInsensitive(std::string const& str, std::string const& prefix) {
    if (prefix.size() > str.size()) {
        return false;
    }
    return std::equal(
        prefix.begin(), prefix.end(), str.begin(), [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

static inline bool EndsWith(std::string const& str, std::string const& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

static inline bool EndsWithAsciiCaseInsensitive(std::string const& str, std::string const& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
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
