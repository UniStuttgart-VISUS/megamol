/*
 * SyncedMMPLDProvider.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "SyncedMMPLDProvider.h"
#include "stdafx.h"


namespace megamol {
namespace datatools {


SyncedMMPLDProvider::SyncedMMPLDProvider()
        : StaticMMPLDProvider()
        , getSyncSlot("getSync", "Slot for synchronization messages.") {

    this->getSyncSlot.SetCallback(core::cluster::SyncDataSourcesCall::ClassName(),
        core::cluster::SyncDataSourcesCall::FunctionName(0), &SyncedMMPLDProvider::checkDirtyCallback);
    this->getSyncSlot.SetCallback(core::cluster::SyncDataSourcesCall::ClassName(),
        core::cluster::SyncDataSourcesCall::FunctionName(1), &SyncedMMPLDProvider::setDirtyCallback);
    MakeSlotAvailable(&getSyncSlot);
}

bool SyncedMMPLDProvider::setDirtyCallback(core::Call& c) {
    this->filenamesSlot.ForceSetDirty();

    return true;
}


bool SyncedMMPLDProvider::checkDirtyCallback(core::Call& c) {
    auto ss = dynamic_cast<core::cluster::SyncDataSourcesCall*>(&c);
    if (ss == nullptr)
        return false;

    if (this->filenamesSlot.IsDirty()) {
        ss->setFilenameDirty();
    }

    this->filenamesSlot.ResetDirty();

    return true;
}

} // namespace datatools
} // namespace megamol
