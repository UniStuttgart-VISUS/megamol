/*
 * NullParticleWriter.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmstd/data/AbstractDataWriter.h"

namespace megamol::datatools {


/**
 * NullWrite behavies like a file writer, i.e. pokes each frame and terminates afterwards
 */
class NullParticleWriter : public core::AbstractDataWriter {
public:
    static const char* ClassName() {
        return "NullParticleWriter";
    }
    static const char* Description() {
        return "NullWrite behavies like a file writer, i.e. pokes each frame and terminates afterwards";
    }
    static bool IsAvailable() {
        return true;
    }

    NullParticleWriter();
    ~NullParticleWriter() override;

protected:
    bool create() override;
    void release() override;
    bool run() override;

private:
    core::CallerSlot dataSlot;
};

} // namespace megamol::datatools
