use warnings 'all';
use strict;
use List::MoreUtils qw/uniq/;

my %defines;
my $ignoreCount = 0;
undef $/;
my $handle;
my $fileName;
#$fileName = "C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Include\\gl\\GLU.h";
$fileName = "T:\\home\\reina\\src\\vislib2010\\vislib.net\\scripts\\MegaMol05API.h";
open($handle, "<", $fileName)
    or croak("cannot open GL.h");
my $glh = <$handle>;
close $handle;

# weed out comments
$glh =~ s|/\*.*?\*/||sg;
$glh =~ s|//.*?$||g;

# un-crlfs are evil
$glh =~ s|\\\s*$||g;

# ifdefs
my @ganzvielefetzen;
my @fetzen = split /(#\s*if.*?\R)/, $glh;
foreach my $f (@fetzen) {
    push @ganzvielefetzen, (split /(#\s*endif.*?\R)/, $f);
}
@fetzen = ();
foreach my $f (@ganzvielefetzen) {
    push @fetzen, (split /(#\s*else.*?\R)/, $f);
}

@ganzvielefetzen = ();
foreach my $f (@fetzen) {
    push @ganzvielefetzen, (split /(#\s*define.*?\R)/, $f);
}

@fetzen = ();
foreach my $f (@ganzvielefetzen) {
    push @fetzen, (split /(#\s*undef.*?\R)/, $f);
}


$glh eq join "", @fetzen or croak ("We broke the file by replacing #if...");

#foreach my $f (@fetzen) {
#    print "fetzen: $f\n";
#    #if ($f =~ /#\s*if/) {
#    #    print "$f\n";
#    #}
#}



foreach my $f (@fetzen) {
    if ($f =~ /#\s*(if\w*)(.*?)\R/) {
        my $a = $1;
        my $b = $2;
        if ($a =~ /^ifdef/) {
            $b = "defined $b";
        }
        if ($a =~ /^ifndef/) {
            $b = "!defined $b";
        }
        print "$b:\n";
        my @tokens = $b =~ /(\w+)/g;
        @tokens = uniq sort {(length $a) < (length $b)} @tokens;
        foreach my $x (@tokens) {
            if ($x eq "defined") {
                next;
            }
            if ($x =~ /^\d/) {
                next;
            }
            #print "replacing $x\n";
            my $c = "";
            foreach my $y (split /(".*?")/, $b) {
                if ($y =~ /^"/) {
                    
                } else {
                    $y =~ s/$x/\$defines{"$x"}/g;
                }
                $c .= $y;
            }
            #$b =~ s/$x/\$defines{"$x"}/g;
            #print "result: $c\n";
            $b = $c;
        }
        print "\tafter: $b\n";
    }
    if ($f =~ /#\s*define\s+(\S+)(?:\s+(.*)\s*)?\R/) {
        if (defined $2) {
            $defines{$1} = $2;
        } else {
            $defines{$1} = '';
        }
    }
    if ($f =~ /#\s*undef\s+(\S+)\s*\R/) {
        delete $defines{$1};
    }
}

#foreach my $h (keys %defines) {
#    print "$h -> $defines{$h}\n";
#}

open ($handle, ">", "t:\\home\\reina\\src\\ogldd\\demp.dxd") || croak ("da derf i net neischreiba");
print $handle $glh;
close $handle;

#print $glh;