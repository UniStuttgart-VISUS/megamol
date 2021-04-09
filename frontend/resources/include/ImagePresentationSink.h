/*
 * ImagePresentationSink.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ImageWrapper.h"

#include <functional>
#include <utility>
#include <list>

namespace megamol {
namespace frontend_resources {

using ImageEntry = std::pair<std::string, ImageWrapper>;
using ImagePresentationSink = std::function<void(std::list<ImageEntry> const&)>;

struct ImagePresentationSinkRegistry {
    std::function<void(std::string const& /*sink name*/, ImagePresentationSink const& /*sink handler*/)> add;
    std::function<void(std::string const& /*sink name*/)> remove;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
