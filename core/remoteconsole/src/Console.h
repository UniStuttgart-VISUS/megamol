#pragma once
#include "Connection.h"

void printGreeting();
void printHelp();
bool execCommand(Connection& conn, std::string command);
void runScript(Connection& conn, const std::string& scriptfile);
void interactiveConsole(Connection &conn);
