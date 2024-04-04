/*
 * VolumetricDataSource.h
 *
 * Copyright (C) 2024 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "datraw/raw_reader.h"

#include "geometry_calls/VolumetricDataCall.h"

#include "mmcore/param/ParamSlot.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "vislib/PtrArray.h"
#include "vislib/RawStorage.h"
#include "vislib/sys/Event.h"
#include "vislib/sys/Thread.h"

namespace megamol::volume {
class NewDatRawReader : public core::Module {

public:
    static inline const char* ClassName() {
        return "NewDatRawReader";
    }
    static inline const char* Description() {
        return "Data source for DatRaw volumetric data.";
    }
    static inline bool IsAvailable() {
        return true;
    }
    NewDatRawReader();
    ~NewDatRawReader() override;

protected:
    bool create() override;
    void release() override;

private:
    core::CalleeSlot slotGetData;
    core::param::ParamSlot paramFileName;
    datraw::info<char> datInfo;
    bool onGetData(core::Call& call);
    bool onGetMetadata(core::Call& call);
    bool onGetExtents(core::Call& call);
    bool onStartAsync(core::Call& call);
    bool onStopAsync(core::Call& call);
    bool onTryGetData(core::Call& call);
    bool onFileNameChange(core::param::ParamSlot& slot);
};
} // namespace megamol::volume
