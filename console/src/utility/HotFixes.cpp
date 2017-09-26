/*
 * utility/HotFixes.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "utility/HotFixes.h"
#include <algorithm>

using namespace megamol;
using namespace megamol::console;

const utility::HotFixes& utility::HotFixes::Instance() {
    static utility::HotFixes inst;
    return inst;
}

utility::HotFixes::HotFixes() : entries() {
}

utility::HotFixes::~HotFixes() {
}

void utility::HotFixes::Clear() {
    entries.clear();
}

void utility::HotFixes::EnableHotFix(const char* name) {
    if (IsHotFixed(name)) return;
    entries.push_back(name);
}

bool utility::HotFixes::IsHotFixed(const char* name) const {
    return std::find(entries.begin(), entries.end(), name) != entries.end();
}
