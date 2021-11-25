#include <stdarg.h>

#include "datRaw.h"
#include "datRaw_log.h"

int datRaw_LogLevel = DR_LOG_ERROR;

void datRaw_logError(char *format, ...)
{
    if (datRaw_LogLevel >= DR_LOG_ERROR) {
        va_list vals;
        va_start (vals, format);
        vfprintf(stderr, format, vals);
        va_end (vals);
    }
}

void datRaw_logWarning(char *format, ...)
{
    if (datRaw_LogLevel >= DR_LOG_WARNING) {
        va_list vals;
        va_start (vals, format);
        vfprintf(stderr, format, vals);
        va_end (vals);
    }
}

void datRaw_logInfo(char *format, ...)
{
    if (datRaw_LogLevel >= DR_LOG_INFO) {
        va_list vals;
        va_start (vals, format);
        vfprintf(stderr, format, vals);
        va_end (vals);
    }
}


