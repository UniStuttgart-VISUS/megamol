/*
 * AsyncResult.h
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#ifndef MEGAMOL_MMSTD_VOLUME_ASYNCRESULT_H_INCLUDED
#define MEGAMOL_MMSTD_VOLUME_ASYNCRESULT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace megamol {
namespace stdplugin {
namespace volume {

    /**
     * Provides means for tracking the progress of asynchronous loading
     * operations.
     */
    class AsyncResult {

    public:

        AsyncResult(void);

        ~AsyncResult(void);
    };

} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_MMSTD_VOLUME_ASYNCRESULT_H_INCLUDED */
