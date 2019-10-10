#include "mmcore\MegaMolGraph.h"

int main() {

	factories::ModuleDescriptionManager moduleProvider;
	factories::CallDescriptionManager callProvider;

	megamol::core::MegaMolGraph graph(moduleProvider, callProvider);

	return 0;
}