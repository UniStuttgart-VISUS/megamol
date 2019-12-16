/*
 * TestArrowsDataSource.h
 *
 * Copyright (C) 2019 by MegaMol Team. Alle Rechte vorbehalten.
 */
// Make crappy clang-format f*** off:
// clang-format off

#pragma once

#include <array>
#include <vector>

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * Creates particle data with directional information.
     */
    class TestArrowsDataSource : public Module {

    public:

        static const char *ClassName(void) {
            return "TestArrowsDataSource";
        }

        static const char *Description(void) {
            return "Test data source providing particles with directions.";
        }

        static bool IsAvailable(void) {
            return true;
        }

        TestArrowsDataSource(void);

        virtual ~TestArrowsDataSource(void);

    protected:

        virtual bool create(void);

        bool onGetData(Call& caller);

        bool onGetExtents(Call& caller);

        virtual void release(void);

    private:

        struct Particle {
            float x, y, z;
            float l;
            float vx, vy, vz;
            float r, g, b;
        };

        std::vector<Particle> data;
        std::array<float, 6> extents;
        CalleeSlot slotGetData;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */
