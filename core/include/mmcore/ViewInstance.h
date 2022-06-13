/*
 * ViewInstance.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEWINSTANCE_H_INCLUDED
#define MEGAMOLCORE_VIEWINSTANCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/ModuleNamespace.h"
#include "mmcore/view/AbstractViewInterface.h"
#include "vislib/assert.h"
#include "vislib/forceinline.h"

/**
 * Function pointer type for view close requests.
 * TODO Moved here from deleted mmcore/api/MegaMolCore.h as only used here
 *
 * @param data The user specified pointer.
 */
typedef void (*mmcViewCloseRequestFunction)(void* data);


namespace megamol {
namespace core {


/**
 * class of view instances.
 */
class ViewInstance : public ModuleNamespace {
public:
    /**
     * Ctor.
     */
    ViewInstance(void);

    /**
     * Dtor.
     */
    virtual ~ViewInstance(void);

    /**
     * Initializes the view instance.
     *
     * @param ns The namespace object to be replaced.
     * @param view The view module to be used.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool Initialize(ModuleNamespace::ptr_type ns, view::AbstractViewInterface* view);

    /**
     * Gets the view object encapsuled by this instance.
     *
     * @return The view object.
     */
    VISLIB_FORCEINLINE view::AbstractViewInterface* View(void) {
        return this->view;
    }

    /**
     * Signals the view that it should be terminated as soon as possible
     * The module must not be immediatly removed from the module graph.
     */
    void Terminate(void) {
        // Informs frontend that the window should be closed as sfx
        // mmcIsViewRunning will return false now!
        this->view = NULL;
    }

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    virtual void ClearCleanupMark(void);

    /**
     * Performs the cleanup operation by removing and deleteing of all
     * marked objects.
     */
    virtual void PerformCleanup(void);

    /**
     * Sets the close request callback.
     *
     * @param func The callback function
     * @param data The user data pointer for the callback function
     */
    inline void SetCloseRequestCallback(mmcViewCloseRequestFunction func, void* data) {
        ASSERT(this->closeRequestCallback == NULL);
        this->closeRequestCallback = func;
        this->closeRequestData = data;
    }

    /**
     * Requests the frontend to close this view.
     */
    void RequestClose(void);

private:
    /** The view module */
    view::AbstractViewInterface* view;

    /** The close request callback function pointer */
    mmcViewCloseRequestFunction closeRequestCallback;

    /** The user data pointer for the close request callback */
    void* closeRequestData;
};


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEWINSTANCE_H_INCLUDED */
