// this whole file is a very slight modification of
// https://github.com/abierbaum/vscode-file-peek
// (licensed under MIT license, it seems)

'use strict';
// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import * as path from 'path';
//import * as console from 'console';

var oc: vscode.OutputChannel;

// this method is called when your extension is activated
// your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

    // Use the console to output diagnostic information (console.log) and errors (console.error)
    // This line of code will only be executed once when your extension is activated
    //console.log('Congratulations, your extension "btf-peek" is now active!');

    var active_languages = ["xml"];
    var extensions = [".btf"];
    const peek_filter: vscode.DocumentFilter[] = active_languages.map((language) => {
        return {
            language: language,
            scheme: 'file'
        };
    });

    oc = vscode.window.createOutputChannel("btf-peek");
    oc.show(true);
    oc.appendLine("btf-peek 0.0.4 activated.");

    // Register the definition provider
    context.subscriptions.push(
        vscode.languages.registerDefinitionProvider(peek_filter,
            new PeekFileDefinitionProvider(extensions))
    );
}

// this method is called when your extension is deactivated
export function deactivate() {
}

/**
 * Provide the lookup so we can peek into the files.
 */
class PeekFileDefinitionProvider implements vscode.DefinitionProvider {
    protected fileSearchExtensions: string[] = [];

    constructor(fileSearchExtensions: string[] = []) {
        this.fileSearchExtensions = fileSearchExtensions;
    }

    /**
     * Return list of potential paths to check
     * based upon file search extensions for a potential lookup.
     */
    getPotentialPaths(lookupPath: string): string[] {
        let potential_paths: string[] = [lookupPath];

        // Add on list where we just add the file extension directly
        this.fileSearchExtensions.forEach((extStr) => {
            potential_paths.push(lookupPath + extStr);
        });

        // if we have an extension, then try replacing it.
        let parsed_path = path.parse(lookupPath);
        if (parsed_path.ext !== "") {
            this.fileSearchExtensions.forEach((extStr) => {
                const new_path = path.format({
                    base: parsed_path.name + extStr,
                    dir: parsed_path.dir,
                    ext: extStr,
                    name: parsed_path.name,
                    root: parsed_path.root
                });
                potential_paths.push(new_path);
            });
        }

        return potential_paths;
    }

    provideDefinition(document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken): Thenable<vscode.Location> {

        return new Promise((resolve, reject) => {
            var working_dir = path.dirname(document.fileName);
            var word = document.getText(document.getWordRangeAtPosition(position));
            var line = document.lineAt(position);

            // We are looking for strings with filenames
            // - simple hack for now we look for the string with our current word in it on our line
            //   and where our cursor position is inside the string
            //let re_str = `\"(.*?${word}.*?)\"|\'(.*?${word}.*?)\'`;
            var re_str = `>(.*?${word}.*?)<`;
            var match = line.text.match(re_str);

            if (null !== match) {
                var potential_fname = match[1]; // || match[2];
                var match_start = match.index;
                var match_end = match.index + potential_fname.length;

                // Verify the match string is at same location as cursor
                if ((position.character >= match_start) &&
                    (position.character <= match_end)) {
                    oc.appendLine("Fname: " + potential_fname);
                    oc.appendLine("working_dir: " + working_dir);

                    // Find all potential paths to check
                    vscode.workspace.findFiles("**/" + potential_fname, "**/share/**").then(
                        files => {
                            //var arr: vscode.Location[];
                            files.forEach(element => {
                                oc.appendLine("found " + element);
                                //arr.push(new vscode.Location(element, new vscode.Position(0, 0)));
                                resolve(new vscode.Location(element, new vscode.Position(0, 0)));
                            });
                            // no idea how to resolve multiple :(
                            //resolve(arr[0]);
                            return;
                        },
                        err => {
                            // I'm not sure I care
                            oc.appendLine("error: " + err);
                        }
                    );

                }
            } else {
                var re_snippet = `<snippet.*?name="([^"]+)"`;
                match = line.text.match(re_snippet);

                //console.log('re_snippet: ' + re_snippet);
                //console.log("   Match: ", match);

                if (null !== match) {
                    var full_name = match[1];
                    // leading "::" need to be removed
                    full_name = full_name.replace(/^::/, '');
                    var pieces = full_name.split("::");
                    if (pieces.length > 1) {
                        var filename = pieces[0];
                        //var snippet = pieces[pieces.length - 1];
                        //console.log(filename);

                        vscode.workspace.findFiles("**/" + filename + ".btf", "**/share/**").then(
                            btfs => {
                                vscode.workspace.openTextDocument(btfs[0]).then(val => {
                                    //console.log("opening " + btfs[0]);
                                    var t = val.getText();
                                    //var parseString = require('xml2js').parseString;
                                    var DOMParser = require('xmldom').DOMParser;
                                    var doc = new DOMParser().parseFromString(t);
                                    console.log(doc);
                                    var btfElem = doc.documentElement;
                                    if (btfElem.tagName === "btf" && btfElem.getAttribute("namespace") === filename) {
                                        // we should be in the right place
                                        var temp = btfElem;
                                        for (var i = 1; i < pieces.length; i++) {
                                            for (var j = 0; 0 < temp.childNodes.length; j++) {
                                                var c = temp.childNodes[j];
                                                if (c.nodeType === 1) {
                                                    // Element
                                                    if (c.tagName === "snippet" && c.getAttribute("name") === pieces[i]) {
                                                        resolve(new vscode.Location(btfs[0], new vscode.Position(c.lineNumber - 1, c.columnNumber - 1)));
                                                        return;
                                                    } else if (c.tagName === "namespace" && c.getAttribute("name") === pieces[i]) {
                                                        // descend
                                                        temp = c;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }, err => { });
                                //resolve(new vscode.Location(val[0], new vscode.Position(0, 1)));
                                //return;
                            },
                            err => {
                                // I'm not sure I care
                                //handleError(err);
                            }
                        );
                        //let snipname = pieces[pieces.length - 1];
                        //console.log(filename + snipname);
                    }
                }
            }
        });
    }
}