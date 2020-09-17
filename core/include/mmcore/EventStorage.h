/*
 * EventStorage.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_EVENT_STORAGE_H_INCLUDED
#define MEGAMOL_EVENT_STORAGE_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/EventCollection.h"
#include "mmcore/Module.h"

namespace megamol {
namespace core {

class MEGAMOLCORE_API EventStorage : public core::Module {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "EventStorage"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module representing a storage for generic events for event-based communication between modules";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }


    /** Ctor. */
    EventStorage();

    /** Dtor. */
    ~EventStorage();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /**
     * Access the events provided by the EventStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readDataCallback(core::Call& caller);

    /**
     * Write/update the flags provided by the EventStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeDataCallback(core::Call& caller);

    /**
     * Access the metadata provided by the EventStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readMetaDataCallback(core::Call& caller);

    /**
     * Write/update the metadata provided by the EventStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeMetaDataCallback(core::Call& caller);

    /** The slot for reading the data */
    core::CalleeSlot readEventsSlot;

    /** The slot for writing the data */
    core::CalleeSlot writeEventsSlot;

    std::array<std::shared_ptr<EventCollection>,2> m_events;
    int m_read_idx;

    uint32_t version = 0;
};

}
}

#endif // !MEGAMOL_EVENT_STORAGE_H_INCLUDED
