/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

// TODO: search and replace Template_Resource with the name you want to give your resource..
// delete the documentation comments after you read and understood them.

// README
// What do resources do? What are they for?
// Resources are a centralized mechanism in the frontend to share data or callbacks among frontend services.
// Each service may provide resources of his own to other services, or access resources shared by others.
// Resources may also be used by certrain graph modules (e.g. views) for module creation, release or rendering - but this mechanism is not yet fully developed.
// The resource object provided by a service will not be copied (intentionally).
// Services get const references to the resources they request and they usually should treat resources as read-only.
//
// Think of a resource as a struct that is filled with data by its owning service.
// A resource should manage only plain data (plain old data, POD) and
// ideally not introduce unnecessary library headers or other special dependencies in its own header file.
// Move implementation defails and library use into the .cpp implementation and keep the resource header file clean from pollution
// by, for example, using the pointer to implementation pattern (PIMPL).
// A simple modern C++ PIMPL pattern may be
//
// in the resource header:
// struct PimplData; // forward declaration in header
// std::unique_ptr<PimplData, std::function<void(PimplData*)>> m_pimpl; // pimpl declaration with incomplete type PimplData and custom deleter
//
// in the resource cpp/implementation:
// allocate PimplData object and also pass a deleter function to unique_ptr so it knows how to call the PimplData destructor
// m_pimpl = std::unique_ptr<PimplData, std::function<void(PimplData*)>>(new PimplData, [](PimplData* ptr) { delete ptr; });
//
// The important thing is to keep the resources you make minimal so other services
// can access only the subset of data or functionality they require without dealing with bloatet objects and complicated header includes.
// Services using your resource should only need to include one header file and ideally not need to link against any libraries
// (it is your job to isolate and hide library usage).
//
// That way the set of resources provided by frontend services is a basic common set of features that is extensible and can be mixed and matched
// by services to implement functionality they want.

namespace megamol {
namespace frontend_resources {

struct Template_Resource {};

} /* end namespace frontend_resources */
} /* end namespace megamol */
