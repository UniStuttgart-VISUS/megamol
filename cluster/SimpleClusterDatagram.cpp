/*
 * SimpleClusterDatagram.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/SimpleClusterDatagram.h"
#include "vislib/CharTraits.h"

using namespace megamol::core;


/*
 * cluster::GetDatagramPort
 */
unsigned int cluster::GetDatagramPort(const utility::Configuration *cfg) {
    if (cfg != NULL) {
        if (cfg->IsConfigValueSet("scudp")) {
            try {
                int v = vislib::CharTraitsW::ParseInt(cfg->ConfigValue("scudp"));
                if (v < 1) v = 1;
                if (v > 65535) v = 65535;
                return static_cast<unsigned int>(v);
            } catch(...) {
            }
        }
    }
    return 30201;
}
