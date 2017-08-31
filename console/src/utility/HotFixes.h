/*
 * utility/HotFixes.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once
#include <string>
#include <vector>

namespace megamol {
namespace console {
namespace utility {

    class HotFixes {
    public:
        static const HotFixes& Instance();
        void Clear();
        void EnableHotFix(const char* name);
        bool IsHotFixed(const char* name) const;
    private:
        HotFixes();
        ~HotFixes();
        std::vector<std::string> entries;
    };

}
}
}
