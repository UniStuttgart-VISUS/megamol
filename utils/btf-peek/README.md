# btf-peek README

Allows to jump quickly between btfs and the actual shader code.

## Features

Should make [F12] / [ALT]-[F12] or 'Peek Definition' available on paths and snippet references inside a btf file.

## Requirements

You need to open the megamol folder for snippet navigation to work: this extension relies on the workspace files for collecting all available btfs.

## Extension Settings

nothing yet.

## Known Issues

nothing yet.

## Release Notes

### 0.0.2

* Added snippet reference support
* provideDefinition is now asynchronous

### 0.0.1

Initial release of btf-peek


## How to install

You need node.js installed.

Add the vsce package:

    npm install -g vsce

Add all possible dependencies of this package (paranoid):

    npm install

generate the vsix:

    vsce package

install from that.