template <typename T>
 std::size_t Log::AddEchoTarget(std::shared_ptr<T> target) {
    auto logger = spdlog::get(echo_logger_name);
    if (logger) {
        logger->sinks().push_back(target);
        return logger->sinks().size() - 1;
    } else {
        logger = std::make_shared<spdlog::logger>(echo_logger_name, target);
        spdlog::register_logger(logger);
        return 0;
    }
}
