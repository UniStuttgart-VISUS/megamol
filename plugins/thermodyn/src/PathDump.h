#pragma once

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

namespace megamol::thermodyn {
class PathDump : public core::AbstractDataWriter {
public:
    static const char* ClassName(void) {
        return "PathDump";
    }
    static const char* Description(void) {
        return "PathDump";
    }
    static bool IsAvailable(void) {
        return true;
    }

    PathDump();

    virtual ~PathDump();

protected:
    bool run() override;

    bool getCapabilities(core::DataWriterCtrlCall& call) override;

    bool create() override;

    void release() override;

private:
    core::CallerSlot data_in_slot_;

    core::param::ParamSlot filename_slot_;
};
} // namespace megamol::thermodyn
