/*
 * EventStorage.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_EVENT_STORAGE_H_INCLUDED
#define MEGAMOL_EVENT_STORAGE_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmstd/event/DoubleBufferedEventCollection.h"

namespace megamol {
namespace core {

class EventStorage : public core::Module {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = Module::requested_lifetime_resources();
        resources.emplace_back("FrameStatistics");
        return resources;
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "EventStorage";
    }

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
    static bool IsAvailable(void) {
        return true;
    }


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
    bool dataCallback(core::Call& caller);

    /**
     * Access the metadata provided by the EventStorage
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool metaDataCallback(core::Call& caller);


    /** The slot for accessomg the event data */
    core::CalleeSlot m_events_slot;

    std::shared_ptr<DoubleBufferedEventCollection> m_events;

    uint32_t m_version = 0;
};

} // namespace core
} // namespace megamol

#endif // !MEGAMOL_EVENT_STORAGE_H_INCLUDED
