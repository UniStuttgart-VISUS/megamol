/**
* Helpers for KHR_debug.
*
* @author Alexandros Panagiotidis <alexandros@panagiotidis.eu>
* @license MIT
*/

#pragma once

#include <functional>
#include <iomanip>
#include <ostream>
#include <vector>
#include "vislib/sys/Log.h"

namespace zen {
namespace gl {

using debug_action = std::function < void(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const * message, GLvoid const * userParam) >;

static std::vector< debug_action > debug_actions;

/**
* Translates the userParam to a (hopefully human-readable/-understandable) string.
* The default returns nothing.
*/
static
std::function< std::string(void const *) > debug_message_user_param_handler = [](void const * userParam)
{
	return std::string();
};

/**
* Queries the context flags and checks whether GL_CONTEXT_FLAG_DEBUG_BIT is set.
* @return true, if the contexts debug flag was set, false otherwise. 
*/
inline
bool is_debug_context()
{
	GLint contextFlags;
	glGetIntegerv(GL_CONTEXT_FLAGS, &contextFlags);
	return (contextFlags & GL_CONTEXT_FLAG_DEBUG_BIT) == GL_CONTEXT_FLAG_DEBUG_BIT;
}

namespace {

inline
std::string debugSourceToString(GLenum source, std::string defaultText)
{
#define _GL_ENUM_TO_STRING_HELPER(theGLEnum__) \
	case theGLEnum__: return #theGLEnum__

	switch (source)
	{
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SOURCE_API);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SOURCE_WINDOW_SYSTEM);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SOURCE_SHADER_COMPILER);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SOURCE_THIRD_PARTY);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SOURCE_APPLICATION);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SOURCE_OTHER);
	}
#undef _GL_ENUM_TO_STRING_HELPER

	return defaultText + " (" + std::to_string(source) + ")";
}

inline
std::string debugTypeToString(GLenum type, std::string defaultText)
{
#define _GL_ENUM_TO_STRING_HELPER(theGLEnum__) \
	case theGLEnum__: return #theGLEnum__

	switch (type)
	{
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_ERROR);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_PORTABILITY);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_PERFORMANCE);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_MARKER);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_PUSH_GROUP);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_POP_GROUP);

	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_TYPE_OTHER);
	}
#undef _GL_ENUM_TO_STRING_HELPER

	return defaultText + " (" + std::to_string(type) + ")";
}

inline
std::string debugSeverityToString(GLenum severity, std::string defaultText)
{
#define _GL_ENUM_TO_STRING_HELPER(theGLEnum__) \
	case theGLEnum__: return #theGLEnum__

	switch (severity)
	{
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SEVERITY_HIGH);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SEVERITY_MEDIUM);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SEVERITY_LOW);
	_GL_ENUM_TO_STRING_HELPER(GL_DEBUG_SEVERITY_NOTIFICATION);
	}
#undef _GL_ENUM_TO_STRING_HELPER

	return defaultText + " (" + std::to_string(severity) + ")";;
}

} // anonymous namespace

inline
debug_action make_debug_action_ostream(std::ostream & stream)
{
	return [&](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const * message, GLvoid const * userParam)
	{
		static std::string const separator_(78, '-');

		stream
			<< separator_ << std::endl
			<< std::setw(12) << "Source: " << debugSourceToString(source, "unknown") << std::endl
			<< std::setw(12) << "Type: " << debugTypeToString(type, "unknown") << std::endl
			<< std::setw(12) << "ID: " << id << std::endl
			<< std::setw(12) << "Severity: " << debugSeverityToString(severity, "unknown") << std::endl
			<< std::setw(12) << "Message: " << std::string(message, length) << std::endl
			<< std::setw(12) << "UserParam: " << debug_message_user_param_handler(userParam) << std::endl
			;
	};
}

inline
debug_action make_debug_action_Log(vislib::sys::Log& l) {
	return [&](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const * message, GLvoid const * userParam) {
		static const vislib::StringA separator("----------- KHR_DEBUG Callback ---------------------------------------------------------");

		auto m = (void (vislib::sys::Log::*)(const char *, ...))&vislib::sys::Log::WriteInfo;
		switch (type) {
			case GL_DEBUG_TYPE_ERROR:
			case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
				m = (void (vislib::sys::Log::*)(const char *, ...))&vislib::sys::Log::WriteError;
				break;
			case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			case GL_DEBUG_TYPE_PORTABILITY:
			case GL_DEBUG_TYPE_PERFORMANCE:
			case GL_DEBUG_TYPE_OTHER:
				m = (void (vislib::sys::Log::*)(const char *, ...))&vislib::sys::Log::WriteWarn;
				break;
			case GL_DEBUG_TYPE_MARKER:
			case GL_DEBUG_TYPE_POP_GROUP:
			case GL_DEBUG_TYPE_PUSH_GROUP:
				m = (void (vislib::sys::Log::*)(const char *, ...))&vislib::sys::Log::WriteInfo;
				break;
		}
		(l.*m)("%s", separator);
		(l.*m)("Source: %s", debugSourceToString(source, "unknown").c_str());
		(l.*m)("Type: %s", debugTypeToString(type, "unknown").c_str());
		(l.*m)("ID: %u", id);
		(l.*m)("Severity: %s", debugSeverityToString(severity, "unknown").c_str());
		(l.*m)("Message: %s", std::string(message, length).c_str());
		(l.*m)("UserParam: ", debug_message_user_param_handler(userParam).c_str());
	};
}

inline
void debug_action_break(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const * message, GLvoid const * userParam)
{
#ifdef _MSC_VER
	__debugbreak();
#else // _MSC_VER
#error Unsupported environment
#endif
}

inline
void debug_action_throw(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const * message, GLvoid const * userParam)
{
	throw std::exception();
}

inline
void debug_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const * message, GLvoid const * userParam)
{
	for (auto & action : debug_actions)
	{
		action(source, type, id, severity, length, message, userParam);
	}
}

template < typename Iterator >
inline void enable_debug_callback(void * userParam, bool sync, Iterator firstAction, Iterator lastAction)
{
	glEnable(GL_DEBUG_OUTPUT);

	if (sync)
	{
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	}

	debug_actions.assign(firstAction, lastAction);

	glDebugMessageCallback(debug_message_callback, userParam);
}

inline
void enable_all_debug_messages()
{
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, 0, GL_TRUE);
}

inline
void ignore_debug_message(GLenum source, GLenum type, GLuint id)
{
	glDebugMessageControl(source, type, GL_DONT_CARE, 1, &id, GL_FALSE);
}

template < typename Iterator >
void ignore_debug_messages(Iterator first, Iterator last, GLenum source, GLenum type)
{
	std::vector< GLuint > ids(first, last);

	glDebugMessageControl(source, type, GL_DONT_CARE, ids.size(), &ids[0], GL_FALSE);
}

using debug_message_spec = std::tuple< GLenum /*source*/, GLenum /*type*/, GLuint /*id*/ >;

inline
void ignore_debug_messages(std::initializer_list< debug_message_spec > ignores)
{
	for (auto & ignore : ignores)
	{
		auto & id = std::get<2>(ignore);
		glDebugMessageControl(std::get<0>(ignore), std::get<1>(ignore), GL_DONT_CARE, 1, &id, GL_FALSE);
	}
}

inline
void object_label(GLenum identifier, GLuint name, std::string label)
{
	glObjectLabel(identifier, name, static_cast<GLsizei>(label.size() * sizeof(label.front())), label.data());
}

struct debug_group
{
	debug_group(std::string name, GLuint id)
	{
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, id, static_cast<GLsizei>(name.size()), &name[0]);
	}

	debug_group(std::string name, GLuint id, GLenum source)
	{
		glPushDebugGroup(source, id, static_cast<GLsizei>(name.size()), &name[0]);
	}

	~debug_group()
	{
		glPopDebugGroup();
	}
};

} // namespace gl
} // namespace zen
