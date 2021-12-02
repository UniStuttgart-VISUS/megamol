#pragma once
#include "Connection.h"

void printGreeting();
void printHelp();
bool execCommand(Connection& conn, const std::string& command, int index = -1);
void runScript(Connection& conn, const std::string& scriptfile, const bool singleSend = false, int index = -1);
void interactiveConsole(Connection& conn);
