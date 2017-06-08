Debugger Expressions for vislib data types
------------------------------------------
Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
Alle Rechte vorbehalten.


As of Visual Studio 2008, this is the only official way of extending the
capabilities of the watch window for native code. Managed code can add real
Microsoft.VisualStudio.DebuggerVisualizers for arbitrary data types, but native
cannot.

The expressions are basically watch macros that reformat the display to pull out
attributes or re-interpret them. They are defined in 
"$(VSInstallDir)Common7\Packages\Debugger\autoexp.dat"
(you need admin privileges to edit that file, sorry). There seems to be no
additional definition file for you to place in "My Documents" or similar.

There are two parts in this file:
  [AutoExpand]
  Here you can define for some attributes to be pulled up to the "Value" column
  in the watch window, without conversions or casting.
  
  [Visualizer]
  Here the interesting stuff happens. You can define previews, the string
  visualizer (the magnifying glass), work with flow control and collection
  types. You can also reference a DLL that will provide complex extraction
  (search for "EEAddIn" in this file).
  
The vislib visualizers (see below) go into the second category, namely between
its header "[Visualizer]" and the warning we are boldly going to ignore:
"; This section contains visualizers for STL and ATL containers"
"; DO NOT MODIFY".

The visualizer definitions follow:

----- snip -----

[Visualizer]

; vislib
vislib::Exception{
	preview (
		#if ($e.isMsgUnicode) (
			#("msg = ", 
				[$e.msg,su]
			)
		) #else (
			#("msg = ",
				[$e.msg,s]
			)
		)
	)
	stringview (
		#if ($e.isMsgUnicode) (
			[$e.msg,sub]
		) #else (
			[$e.msg,sb]
		)
	)
}
vislib::Array<*,*,*>{
	preview (
		#("capacity = ", $e.capacity, " count = ", $e.count)
	)
	children (
		#if ($e.count == 0) (
			#(
				; We make empty match_results appear to have no children.
				#([actual members] : [$e,!]),
				#array(expr: 0, size: 0)
			)
		) #else (
			#(
				#([actual members] : [$e,!]),
				#array(expr: $e.elements[$i], size: $e.count)
			)
		)
	)
}
vislib::SingleLinkedList<*,*>{
	children(
		#(
			#([actual members] : [$e,!]),
			#list(
				head: $e.first,
				next: next,
			) : $e.item
		)
	)
}

; This section contains visualizers for STL and ATL containers
; DO NOT MODIFY

----- snip -----