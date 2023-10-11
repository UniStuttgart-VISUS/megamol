namespace megamol::core::utility::log {
template<typename... Args>
void Log::writeMessage(log_level level, std::string const& msg, Args&&... args) {
    auto logger = spdlog::get(logger_name);
    auto echo_logger = spdlog::get(echo_logger_name);
    if (!logger)
        return;
    if (msg.empty())
        return;
    // remove newline at end because spdlog and other log targets already add newlines
    if (msg.back() == '\n') {
        this->writeMessage(level, msg.substr(0, msg.size() - 1), std::forward<Args>(args)...);
        return;
    }

    auto fmsg = msg;
    if constexpr (sizeof...(Args) > 0) {
        if (msg.find("%") == std::string::npos) {
            fmsg = fmt::format(msg, std::forward<Args>(args)...);
        } else {
            fmsg = fmt::sprintf(msg, std::forward<Args>(args)...);
        }
    }

    switch (level) {
    case log_level::error: {
        logger->error(fmsg);
        (echo_logger ? echo_logger->error(fmsg) : (void)(0));
    } break;
    case log_level::warn: {
        logger->warn(fmsg);
        (echo_logger ? echo_logger->warn(fmsg) : (void)(0));
    } break;
    case log_level::info:
    default: {
        logger->info(fmsg);
        (echo_logger ? echo_logger->info(fmsg) : (void)(0));
    }
    }
}
} // namespace megamol::core::utility::log
