# MegaMol Coding Styleguide

## General Remarks

Many of our styleguide rules are defined in the `.clang-format` file within the source repository.
In addition, a GitHub Actions job and the script `.ci/check_format.sh` will check (and fix) some of our rules.
Therefore, in case of conflicts with this document these technical definitions are to be preferred.
An exception is made to third-party code within the repository (always placed within `3rd` directories), there the original formatting should be kept.

## Basic Formatting Rules

- Whitespace: Always use spaces.
- Indentation: 4 spaces.
- Line length: 120 chars.
- File encoding: UTF-8 (without BOM).
- Always add a line break at the end of a file[^1].
- Always use `LF` line breaks within the repo. (Git `core.autocrlf=true` recommended for Windows checkout.)

## Code Style

(See `.clang-format` for full details.)

- Space only after a pointer or reference operator (i.e. `int* foobar`, not ~~`int *foobar`~~)[^2].
- Brace style "Attach".
- Use "east const" (i.e. `int const& foobar`n not ~~`const int& foobar`~~)[^3].
- Initializer lists must have the line break before the comma, indentation with 8 spaces[^4].
- Use C++17 style namespaces. No namespace indentation. 
- Use `nullptr` over `NULL` or `0`.
- Prefer `enum class` over `enum` wherever possible.
- May or may not add a comment to a closing preprocessor #endif with the content of the if condition. (This rule is more just a suggestion until we may can add this to automated testing.)

## Object Naming

- Class names must be formatted as `CamelCase`, and may use a suffix `GL` (without `_`).
- Namespaces must be all lowercase (and preferably be just a single word). May the suffix `_gl` is appended.
- Public member function must be formatted as `CamelCase`.
- Private or protected member functions must be formatted as `camelBack`.
- Member variables should be formatted as `lower_snake_case_` and have a final `_`. Other markings like a `m_` prefix are to be avoided.
- Do not use a `this->` prefix, unless absolutely technically required[^5].
- Do not prefix function calls with absolute addressing `::function()`, unless absolutely technically required.

## File Organization

- Use `#pragma once` and do not use include guards. 
- Split includes into 4 groups, each group is ordered alphabetically, empty line between groups:
  1. Corresponding Header to the current .cpp file (with `""`).
  2. System Includes (with `<>`)
  3. External Library Includes (with `<>`[^6])
  4. MegaMol internal includes (with `""`)
- Header files use the extension `.h`, source files `.cpp`
- Files names must match the corresponding class name (also in `UpperCamelCase`).
- Directory names must match the corresponding namespace (at least for top level categorisation, additional subdirectories may or may not introduce an additional namespace level).
  Directory names are always lowercase, should preferably only be a single word, but may use the `_gl` or `_cuda` suffix.

## Best practices

- Prefer const correctness
- Prefer use of auto
- Avoid macros

## Footnotes - Reasoning

[^1]: Cleaner `git diff`.  
[^2]: Easier and more C++-like to think of a pointer or reference as a type, even if it is technically not exact.  
[^3]: Only the single rule `const` always refers to what is on its left.  
[^4]: Allows for `#ifdef` or commenting without affecting other entries.  
[^5]: We use a `_` suffix to mark member variables and IDE highlighting should make this unnecessary.  
[^6]: The msvc compiler flag `/external:anglebrackets` allows us to suppress warnings from external code. Therefore, all external code must use `<>` and all internal code `""`.  
