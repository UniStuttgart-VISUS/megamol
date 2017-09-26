/*
 * ParamFileManager.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */

#ifndef MEGAMOLCON_UTILITY_PARAMFILEMANAGER_H_INCLUDED
#define MEGAMOLCON_UTILITY_PARAMFILEMANAGER_H_INCLUDED
#pragma once

#include "vislib/String.h"

namespace megamol {
namespace console {
namespace utility {

    class ParamFileManager {
    public:
        static ParamFileManager& Instance();

        vislib::TString filename;
        void* hCore;

        void Load();
        void Save();

    private:
        ParamFileManager();
        ~ParamFileManager();
    };

} /* end namespace utility */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_UTILITY_PARAMFILEMANAGER_H_INCLUDED */
