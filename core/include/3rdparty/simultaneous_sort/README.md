# Simultaneous Sort
Small header-only library for providing the possibility to simultaneously sort multiple arrays, based on a comparator for the first one.

Being in the situation of having multiple arrays, you might want to sort one of them using a comparator, and have the others mimic its reordering. To prevent needless copying into one array composed of tuples, which hold the values from each input array, this small library implements a proxy object and uses value-wrapping. This way, only the value of the first array is exposed and no copying needs to be performed.

As sorting algorithm, std::sort is used.

## Usage
The public interface of the library is very simple. Here, you have the choice between two functions: one for sorting in ascending order using the default *less* operator, and one where you can specify a custom predicate for sorting.

### Sorting in ascending order:
```c++
std::vector<int> a, b, c;
std::vector<double> d;
// ...
sort(a, b, c, d);
```

### Sorting using a custom predicate:
```c++
std::vector<int> a, b, c;
std::vector<double> d;
// ...
auto my_predicate = [](const int lhs, const int rhs) { return lhs > rhs; };
sort_with(my_predicate, a, b, c, d);
```

The sorting is not restricted to std::vector, but to all containers implementing a random access iterator.

## Integration into your code
Basically, there are two possibilities to use this library:
  - copy the files into your own project, or
  - use the install mechanism and use CMake to include the library *simultaneous_sort*.

## Limitation
This library needs C++17, as folding expressions are used. Furthermore, so far it has been only tested on Windows 10 with Visual Studio 2017 and Ubuntu Linux with gcc 8.2.
