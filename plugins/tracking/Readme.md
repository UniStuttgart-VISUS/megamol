# Tracking plugin
The Tracking plugin is the interface that allows MegaMol plugins to communicate with the tracking software Motive. Since the required library NatNet is not available for Linux and the plguin is only usable in combination with the Powerwall it only works with Windows.
This plugin can be connected with the Powerwall demo plugins to allow interaction with the Stick.

## Building
[NatNet](http://optitrack.com/products/natnet-sdk/) is included in this package and it is required for this plugin to build. If it is not included download the SDK and copy the lib and include folder into the ...\plugins\tracking\natnet\ folder. If NatNet is not automatically found set the appropriate `NATNET_GENERIC_LIBRARY`.

## Modules

The Tracing plugin has two moduls: `VrpnTracker` and `NatNetTracker`. The `VrpnTracker` is the vrpn-client for the Tracking plugin, it recieves updates from the vrpn-server. The `NatNetTracker` handles the connection via NatNet to the Motive software that streams the data from the tracking cameras.
