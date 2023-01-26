/*
 * NullParticleWriter.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmstd/data/AbstractDataWriter.h"

namespace megamol {
namespace datatools {


/**
 * NullWrite behavies like a file writer, i.e. pokes each frame and terminates afterwards
 */
class NullParticleWriter : public core::AbstractDataWriter {
public:
    static const char* ClassName(void) {
        return "NullParticleWriter";
    }
    static const char* Description(void) {
        return "NullWrite behavies like a file writer, i.e. pokes each frame and terminates afterwards";
    }
    static bool IsAvailable(void) {
        return true;
    }

    NullParticleWriter(void);
    ~NullParticleWriter(void) override;

protected:
    bool create(void) override;
    void release(void) override;
    bool run(void) override;
    bool getCapabilities(core::DataWriterCtrlCall& call) override;

private:
    core::CallerSlot dataSlot;
};

} // namespace datatools
} /* end namespace megamol */
