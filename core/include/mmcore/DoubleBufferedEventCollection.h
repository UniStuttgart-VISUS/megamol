/*
 * DoubleBufferedEventCollection.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef DOUBLE_BUFFERED_EVENT_COLLECTION_H_INCLUDED
#define DOUBLE_BUFFERED_EVENT_COLLECTION_H_INCLUDED

#include <array>
#include <memory>

#include "EventCollection.h"

namespace megamol {
namespace core {

class DoubleBufferedEventCollection {
public:
    DoubleBufferedEventCollection();
    ~DoubleBufferedEventCollection() = default;

    /**
     *
     */
    template<typename EventType>
    std::vector<EventType> get() const;

    /**
     *
     */
    template<typename EventType>
    void add(std::unique_ptr<EventType>&& event);

    /**
     *
     */
    template<typename EventType>
    std::vector<EventType> consume();

    /**
     *
     */
    void swap();

private:
    std::array<EventCollection, 2> m_event_collections;

    int m_read_idx;
};

inline DoubleBufferedEventCollection::DoubleBufferedEventCollection() : m_event_collections(), m_read_idx(0) {}

inline void DoubleBufferedEventCollection::swap() {
    m_event_collections[m_read_idx].clear();
    m_read_idx = m_read_idx == 0 ? 1 : 0;
}

template<typename EventType>
inline std::vector<EventType> DoubleBufferedEventCollection::get() const {
    return m_event_collections[m_read_idx].get<EventType>();
}

template<typename EventType>
inline void DoubleBufferedEventCollection::add(std::unique_ptr<EventType>&& event) {
    m_event_collections[m_read_idx == 0 ? 1 : 0].add<EventType>(std::forward<std::unique_ptr<EventType>>(event));
}

template<typename EventType>
inline std::vector<EventType> DoubleBufferedEventCollection::consume() {
    static_assert(EventType::is_consumable_t::value);

    auto retval = m_event_collections[m_read_idx].get<EventType>();
    m_event_collections[m_read_idx].remove<EventType>();

    return retval;
}

} // namespace core
} // namespace megamol


#endif // !DOUBLE_BUFFERED_EVENT_COLLECTION_H_INCLUDED
