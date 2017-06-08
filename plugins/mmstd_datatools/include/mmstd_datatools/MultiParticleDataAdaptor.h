/*
 * AbstractParticleManipulator.h
 *
 * Copyright (C) 2016 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_MULTIPARTICLEDATAADAPTOR_H_INCLUDED
#define MEGAMOL_DATATOOLS_MULTIPARTICLEDATAADAPTOR_H_INCLUDED
#pragma once

#include "mmstd_datatools/mmstd_datatools.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Simple adapter helper class
     *
     * Grants pseudo-linear access to all particles of all lists of the call
     *
     * @remarks:
     *   - Only works with float data for position and color!
     *   - All list should use the same data types (but can use different memory layouts).
     */
    class MMSTD_DATATOOLS_API MultiParticleDataAdaptor {
    private:

        typedef struct _list_data_t {
            size_t count;
            const uint8_t *pos_data;
            const uint8_t *col_data;
            size_t pos_data_step;
            size_t col_data_step;
            struct _list_data_t *next;
        } list_data;

    public:
        MultiParticleDataAdaptor(core::moldyn::MultiParticleDataCall& data);
        ~MultiParticleDataAdaptor();

        inline size_t get_count() const {
            return count;
        }
        inline float const* get_position(size_t idx) const {
            for (list_data *l = list; l != nullptr; l = l->next) {
                if (idx < l->count) {
                    const uint8_t *p = l->pos_data;
                    if (p == nullptr) return nullptr;
                    return reinterpret_cast<float const*>(p + idx * l->pos_data_step);
                } else { 
                    idx -= l->count;
                }
            }
            return nullptr;
        }
        inline float const* get_color(size_t idx) const {
            for (list_data *l = list; l != nullptr; l = l->next) {
                if (idx < l->count) {
                    const uint8_t *p = l->col_data;
                    if (p == nullptr) return nullptr;
                    return reinterpret_cast<float const*>(p + idx * l->col_data_step);
                } else {
                    idx -= l->count;
                }
            }
            return nullptr;
        }

    protected:

        const core::moldyn::MultiParticleDataCall& data;

    private:

        size_t count;
        list_data *list;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_MULTIPARTICLEDATAADAPTOR_H_INCLUDED */
