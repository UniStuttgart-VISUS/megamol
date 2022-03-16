#pragma once

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include <memory>
#include <unordered_map>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

template<class Container>
class LRUCache {
public:

    // TODO thread safety

    using Value = typename Container::FrameType;
    using Key = typename Container::FrameType::FrameIndexType;

    LRUCache() = default;

    void clear() {
        entries.clear();
        accessCount = 0;
        totalByteCount = 0;
    }

    std::shared_ptr<const Value> get(const Key& key) const {
        auto result = entries.find(key);
        if (result != entries.end()) {
            result->second.lastAccess = accessCount++;
            return result->second.value;
        } else {
            return nullptr;
        }
    }

    std::shared_ptr<const Value> operator[](const Key& key) const {
        return get(key);
    }

    std::shared_ptr<const Value> findOrCreate(const Key& key, Container& container) {
        if (maximumSize == 0) {
            // I do not keep dibs, I have no space
            return std::move(container.ReadFrame(key));
        }

        auto result = entries.find(key);
        if (result != entries.end()) {
            result->second.lastAccess = accessCount++;
            return result->second.value;
        } else {
            Entry entry;
            entry.lastAccess = accessCount++;
            entry.value = std::move(container.ReadFrame(key));
            entry.byteCount = entry.value != nullptr ? entry.GetSize() : 0;

            totalByteCount += entry.byteCount;
            entries.insert(std::make_pair(key, entry));

            cleanUp();

            return entry.value;
        }
    }

    void setMaximumSize(std::size_t maximumSize) {
        if (this->maximumSize != maximumSize) {
            this->maximumSize = maximumSize;
            cleanUp();
        }
    }

    std::size_t getMaximumSize() const {
        return maximumSize;
    }

private:
    struct Entry {
        // I own the data and keep dibs so it does not disappear when unused but we still have space
        std::shared_ptr<const Value> value;
        mutable std::size_t lastAccess = 0;
        std::size_t byteCount = 0;
    };

    void cleanUp() {
        if (maximumSize == 0) {
            entries.clear();
        } else if (totalByteCount > maximumSize) {
            // Obtain list of entries
            std::vector<std::pair<Key, Value>> entryList;
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                entryList.push_back(*it);
            }

            // Sort by last access time (oldest entries last)
            std::sort(entryList.begin(), entryList.end(),
                [](const auto& a, const auto& b) { return a.lastAccess > b.lastAccess; });

            // Clean up until total memory usage is below threshold
            std::size_t cleanupThreshold = maximumSize * cleanupFactor;
            while (totalByteCount > cleanupThreshold && !entryList.empty()) {
                totalByteCount -= entryList.back().value.byteCount;
                entries.erase(entryList.back().first);
                entryList.pop_back();
            }
        }
    }

    std::unordered_map<Key, Entry> entries;
    mutable std::size_t accessCount = 0;
    std::size_t totalByteCount = 0;
    std::size_t maximumSize = 0;
    float cleanupFactor = 0.9;
};

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
