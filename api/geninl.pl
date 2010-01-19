#
# geninl.pl
#
# Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
# Alle Rechte vorbehalten.
#
$#ARGV == 1 || die "Syntax: geninl.pl <infile> <outfile>";

my $inFile = shift;
my $outFile = shift;
my @allFuncNames = ();

open (INFILE, $inFile) || die "Cannot open input file \"$inFile\"";
while (<INFILE>) {
    chomp;
    # MEGAMOLVIEWER_API unsigned int MEGAMOLVIEWER_CALL(mmvGetHandleSize)(void) MEGAMOLVIEWER_INIT;
    # if (/^\s*MEGAMOLVIEWER_API\s+(.+)\s+MEGAMOLVIEWER_CALL\((\w+)\)\s*(\([^;]+);/) {
    if (/MEGAMOLCORE_CALL\s+([^\(]*)\(/) {
        push @allFuncNames, $1;
    }
}
close (INFILE);

open (OUTFILE, ">$outFile") || die "Cannot write output file \"$outFile\"";
print OUTFILE "//\n";
print OUTFILE "// MegaMolCore.inl\n";
print OUTFILE "//\n";
print OUTFILE "// Copyright (C) 2009 by Universitaet Stuttgart (VISUS).\n";
print OUTFILE "// Alle Rechte vorbehalten.\n";
print OUTFILE "//\n";
print OUTFILE "// GENERATED FILE! DO NOT EDIT!\n";
print OUTFILE "//\n";
print OUTFILE "\n";
print OUTFILE "#define __ALL_API_FUNCS \\\n";
foreach $func (@allFuncNames) {
    print OUTFILE "function_cast<void*>($func),\\\n";
}
print OUTFILE "NULL";
print OUTFILE "\n";
close (OUTFILE);
