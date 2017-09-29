#pragma once
#include "Connection.h"

void printGreeting();
void printHelp();
void runScript(Connection& conn, const std::string& scriptfile);
void interactiveConsole(Connection &conn);
