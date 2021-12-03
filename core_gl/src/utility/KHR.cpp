/*
 * KHR.cpp
 *
 * Copyright (C) 2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore_gl/utility/KHR.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"


using namespace megamol::core::utility;


int KHR::startDebug() {
#if (!defined(_WIN32))
    megamol::core::utility::log::Log::DefaultLog.WriteError("KHR debug is disabled, use new frontend!.");
    return 1;
}
#else
    GLint numExt;
    glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);
    for (int x = 0; x < numExt; x++) {
        const GLubyte* str = glGetStringi(GL_EXTENSIONS, x);
        if (strcmp(reinterpret_cast<const char*>(str), "GL_KHR_debug") == 0) {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback((GLDEBUGPROC)DebugCallback, NULL);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
            GLuint ignorethis;
            ignorethis = 131185;
            glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
            ignorethis = 131184;
            glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
            ignorethis = 131204;
            glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
        }
    }
    return 0;
}


void KHR::DebugCallback(unsigned int source, unsigned int type, unsigned int id, unsigned int severity, int length,
    const char* message, void* userParam) {
    const char *sourceText, *typeText, *severityText;
    switch (source) {
    case GL_DEBUG_SOURCE_API:
        sourceText = "API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        sourceText = "Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        sourceText = "Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        sourceText = "Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        sourceText = "Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        sourceText = "Other";
        break;
    default:
        sourceText = "Unknown";
        break;
    }
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        typeText = "Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        typeText = "Deprecated Behavior";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        typeText = "Undefined Behavior";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        typeText = "Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        typeText = "Performance";
        break;
    case GL_DEBUG_TYPE_OTHER:
        typeText = "Other";
        break;
    case GL_DEBUG_TYPE_MARKER:
        typeText = "Marker";
        break;
    default:
        typeText = "Unknown";
        break;
    }
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        severityText = "High";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        severityText = "Medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        severityText = "Low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        severityText = "Notification";
        break;
    default:
        severityText = "Unknown";
        break;
    }
    const int outputlength = 8192;
    static char outputstring[outputlength];
    std::string stack = getStack();
#ifdef _WIN32
    sprintf_s(outputstring, outputlength, "[%s %s] (%s %u) %s\nstack trace:\n%s\n", sourceText, severityText, typeText,
        id, message, stack.c_str());
    OutputDebugStringA(outputstring);
#else
    sprintf(outputstring, "[%s %s] (%s %u) %s\nstack trace:\n%s\n", sourceText, severityText, typeText, id, message,
        stack.c_str());
#endif
    if (type == GL_DEBUG_TYPE_ERROR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("%s", outputstring);
    } else if (type == GL_DEBUG_TYPE_OTHER || type == GL_DEBUG_TYPE_MARKER) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("%s", outputstring);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("%s", outputstring);
    }
}

#ifdef _WIN32
std::string KHR::getStack() {
    unsigned int i;
    void* stack[100];
    unsigned short frames;
    SYMBOL_INFO* symbol;
    HANDLE process;
    std::stringstream output;

    process = GetCurrentProcess();

    SymSetOptions(SYMOPT_LOAD_LINES);

    SymInitialize(process, NULL, TRUE);

    frames = CaptureStackBackTrace(0, 200, stack, NULL);
    symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (i = 0; i < frames; i++) {
        SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
        DWORD dwDisplacement;
        IMAGEHLP_LINE64 line;

        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        if (!strstr(symbol->Name, "khr::getStack") && !strstr(symbol->Name, "khr::DebugCallback") &&
            SymGetLineFromAddr64(process, (DWORD64)(stack[i]), &dwDisplacement, &line)) {

            output << "function: " << symbol->Name << " - line: " << line.LineNumber << "\n";
        }
        if (0 == strcmp(symbol->Name, "main"))
            break;
    }

    free(symbol);
    return output.str();
}
#endif
#endif
