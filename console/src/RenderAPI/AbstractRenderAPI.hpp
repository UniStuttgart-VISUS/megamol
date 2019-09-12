/*
 * AbstractRenderAPI.hpp
 *
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include <string>

namespace megamol {
namespace console {

class AbstractRenderAPI {
private:
	// creation/deletion of render APIs may depend on other RAPI instances, since they may share data or context state.
	// using individual priorities, one can schedule RAPI instances of lower priorities
	// to be initialized after and deleted earlier than their dependencies
	// alternative approach: explicit dependency managment between RAPIs in a graph-like structure, like the MegaMol graph
    int m_processingPriority = 0;
    bool m_shouldShutdown = false;

public:
    int setPriority(const int p) {
        const auto old = m_processingPriority; 
        m_processingPriority = p;
        return old;
    }
    int getPriority() const { return m_processingPriority; }

	bool shouldShutdown() const { return m_shouldShutdown; }
	void setShutdown(const bool s = true) { m_shouldShutdown = s; }

public:
    virtual ~AbstractRenderAPI() {}

	// TODO: need to look up API Name+Version without instantiating subclass object? do with statics somehow?
	virtual std::string getAPIName() const = 0;

	struct RenderAPIVersion {
        int major = 0, minor = 0;
	};
	virtual RenderAPIVersion getAPIVersion() const = 0; // TODO: think about support for different versions / compatability

	virtual bool initAPI(void* configPtr) = 0; // init API, e.g. init GLFW with OpenGL and open window with certain decorations/hints
	virtual void closeAPI() = 0;

	virtual const void* getAPISharedDataPtr() const { return nullptr; } // ptr non-owning, share data should be only borrowed

	virtual void preViewRender() = 0; // prepare rendering with API, e.g. set OpenGL context, frame-timers, etc
	virtual void postViewRender() = 0; // clean up after rendering, e.g. stop and show frame-timers in GLFW window

	// TODO: how to register passing user inputs? via existing callbacks?

	// TODO: how to force/allow subclasses to interop with other APIs?

	// TODO: API-specific options in subclasses?
};


} // namespace console
} // namespace megamol
