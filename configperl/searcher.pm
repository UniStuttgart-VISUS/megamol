package VISUS::configperl::searcher;
#
# searcher.inc
# Copyright (c) 2008-2011, VISUS
# (Visualization Research Center, Universitaet Stuttgart)
# All rights reserved.
#
# See: LICENCE.TXT or
# https://svn.vis.uni-stuttgart.de/utilities/configperl/LICENCE.TXT
#
use strict;
use warnings 'all';
use File::Spec::Functions;
use Cwd qw{abs_path cwd};
use Time::HiRes qw(usleep);
use threads;
use threads::shared;

use configperl qw{%headers};

my $t;
my %foundLibs :shared;
my $configperlkey;
my %configperlDBM;
my $configperlDBMavailable = 0;
my $input :shared;

my @searchedPaths :shared;

my $searchDone = 0;
#our @searchFolders :shared = ("..", "../..");
#my @searchFolders :shared = initSearchFolders();
my @searchFolders :shared = ("..", "../..");
my $currentFolder :shared = "";
my $folderBeingRecursed :shared = "";

#our %headers;
#$headers{"GL/glut.h\$"} = "..";
#$headers{"vislib/assert.h\$"} = "../../..";

sub initSearchFolders {
    my @res;
    #my $c = cwd();
    my $count = () = cwd() =~ /[\/\\]/g;
    my $s = "..";
    for (my $x = 0; $x < $count; $x++) {
        push @res, $s;
        $s .= "/..";
    }
    return @res;
}

sub addSearchFolder {
    my $d = shift;
    my $v = shift;
    print "pushing $d onto searchFolders.\n" if $v;
    push @searchFolders, $d;
    return;
}

sub appendLib {
    if ($#_ == 1) {
        my ($h, $w) = @_;
        lock %foundLibs;
        if (defined $foundLibs{$h}) {
            # do not push duplicates!
            if (! grep(/^${w}$/i, @{$foundLibs{$h}})) {
                #print "appending $w to $h\n";
                push @{$foundLibs{$h}}, $w;
            }
        } else {
            #print "new array for $h in $w\n";
            $foundLibs{$h} = &share([]);
            push @{$foundLibs{$h}}, $w;
            #$foundLibs{$h} = \@array;
        }
    }
    return;
}

sub getValFromDB {
    my $k = shift;
    my $val;
    if ($configperlkey) {
        $val = $configperlkey->GetValue($k);
    } else {
        if ($configperlDBMavailable) {
            if ($configperlDBM{$k}) {
                #whatever?
                $val = [split /\0/, $configperlDBM{$k}];
            } else {            
                $val = [];
            }
        }
    }
    return $val;
}

sub setValInDB {
    my $k = shift;
    my $val = shift;
    if ($configperlkey) {
        $configperlkey->SetValue($k, $val, "REG_MULTI_SZ");
    } else {
        if ($configperlDBMavailable) {
            $configperlDBM{$k} = join "\0", @{$val};
        }
    }    
}

sub appendValToDB {
    my $k = shift;
    my $libpath = shift;
    my $val = getValFromDB($k);
    if ($val) {
        if (! grep(/^${libpath}$/i, @{$val})) {
            push $val, $libpath;
            setValInDB($k, $val);
        }
    } else {
        setValInDB($k, [$libpath]);
    }
}

sub findHeaders {
    my @stack;
    my $currdir;
    my $nextdir;
    my $libpath;
    my $fullauto = shift;
    my %foundHeaders;

    #print "entering findHeaders\n";        
    $SIG{'KILL'} = sub { print "ack! header hunter has been killed!\n"; threads->exit(); };
    
    my $Registry;
    eval qq{use Win32::TieRegistry (Delimiter => qq{/}, SplitMultis=>1, ArrayValues => 0,
        qw( REG_SZ REG_EXPAND_SZ REG_DWORD REG_BINARY REG_MULTI_SZ
            KEY_READ KEY_WRITE KEY_ALL_ACCESS ));};
    my $softkey;
    eval q{$softkey = new Win32::TieRegistry "CUser/Software/", {Access=>KEY_READ(), Delimiter=>"/"}};
    my $visuskey;
    if ($softkey) {
        #print "I can access the registry!\n";
        $visuskey = $softkey->Open("VISUS", {Access=>KEY_ALL_ACCESS()});
        if (! $visuskey) {
            my $tmpkey = new Win32::TieRegistry "CUser/Software/", {Access=>KEY_WRITE(), Delimiter=>"/"};
            $visuskey = $tmpkey->CreateKey("VISUS", {Access=>KEY_ALL_ACCESS()});
        }
        $configperlkey = $visuskey->Open("configperl", {Access=>KEY_ALL_ACCESS()});
        if (! $configperlkey) {
            $configperlkey = $visuskey->CreateKey("configperl", {Access=>KEY_ALL_ACCESS()});
        }
    }

    #always use DB instead of registry:
    #undef $configperlkey;
    if ($configperlkey) {
        $configperlkey->SplitMultis(1);
    } else {
        # pre-load stuff from home
        use Fcntl;
        eval "use AnyDBM_File; use File::HomeDir";
        if (my $h = File::HomeDir->my_home) {
            if (tie(%configperlDBM, 'AnyDBM_File', "$h/configperl.libcache.dbmx", O_RDWR|O_CREAT, 0640)) {
                $configperlDBMavailable = 1;
            }
        }
    }
 
    # pre-load stuff from database, removing those that do not exist anymore!
    if ($configperlkey || $configperlDBMavailable) {
        foreach my $k (keys(%VISUS::configperl::headers)) {
            my $val = getValFromDB($k); #$configperlkey->GetValue($k);
            my $rewrite = 0;
            my @newvals;
            foreach my $inst (@{$val}) {
                if (! -e "$inst/$k") {
                    $rewrite = 1;
                } else {
                    appendLib($k, $inst);
                    $foundHeaders{$k} = 1;
                    push @newvals, $inst;
                }
            }
            if ($rewrite) {
                #$configperlkey->SetValue($k, \@newvals, "REG_MULTI_SZ");
                setValInDB($k, \@newvals);
            }
        }
    }
   

    while(!$fullauto || ($fullauto && $#searchFolders > -1)) { # when in fullauto, only as long as there are searchfolders left!!!!
        if ($#searchFolders > -1) {
            my $sp = abs_path(canonpath(shift(@searchFolders)));
            push @stack, $sp;
            push @searchedPaths, $sp;
            $folderBeingRecursed = $sp;
    
            while ($#stack > -1) {
                $currdir = pop @stack;
                $currentFolder = $currdir;
                #print "searching $currdir\n";
                opendir (my $CURRDIR, $currdir) || next; #die "wtf is the dir $currdir";
                while ($nextdir = readdir $CURRDIR) {
                    if (-d "$currdir/$nextdir" && !($nextdir =~ /^\./)) {
                        my $d;
                        my $found = 0;
                        foreach my $d (@searchedPaths) {
                            #print "checking $d against $currdir/$nextdir\n";
                            if ($d eq "$currdir/$nextdir") {
                                $found = 1;
                                #print "already visited $d\n";
                                last;
                            }
                        }
                        if ($found == 0) {
                                #print "pushing $currdir/$nextdir";
                                push @stack, "$currdir/$nextdir";
                        }
                    } else {
                        if (-f "$currdir/$nextdir") {
                            my $test = "$currdir/$nextdir";
                            foreach my $k (keys(%VISUS::configperl::headers)) {
                                if ($test =~ /$k$/) {
                                    #print "found $k in $test\n";
                                    #print "\a"; # terminal bell
                                    $libpath = abs_path(canonpath("$currdir/$VISUS::configperl::headers{$k}"));
                                    appendLib($k, $libpath);
                                    $foundHeaders{$k} = 1;
                                    appendValToDB($k, $libpath);
                                }
                            }
                            if ($fullauto) {
                                if (scalar(keys(%VISUS::configperl::headers)) == scalar(keys(%foundHeaders))) {
                                    closedir $CURRDIR;
                                    return;
                                }
                            }
                        }
                    }
                }
                closedir $CURRDIR;
                $currentFolder = "";
            }
            $folderBeingRecursed = "";
            #print qq{done searching "$k".\n};
        }
        usleep(1);
    }
    #print "exiting searcher\n";
    return;
}

sub waitForInput {
    chomp($input = <STDIN>);
    return;
}

sub dumpLibs {
    lock %foundLibs;
    print "\n";
    foreach my $b (keys(%foundLibs)) {
        print "$b:\n";
        print "-" x (length($b) + 1) . "\n";
        foreach my $a (@{$foundLibs{$b}}) {
            print "$a\n";
        }
        print "\n";
    }
    return;
}

sub printResults {
    my $libname = shift;
    lock %foundLibs;
    my $num = 0;
    if (defined $foundLibs{$libname}) {
        foreach my $lib (@{$foundLibs{$libname}}) {
            if ($num > 0) {
                print " " x 4;
            }
            print "[$num] $lib\n";
            $num++;
        }
    } else {
        print "no suggestions so far.\n";
    }
    return;
}

sub printSearchFolders {
    print "Folders already searched:\n";
    foreach my $sp (@searchedPaths) {
        if ($sp ne $folderBeingRecursed) {
            print "$sp\n";
        }
    }
    print "Currently searching: " . (($currentFolder ne "") ? $currentFolder : "-") . "\n";
    print "Folders still to search:\n";
    foreach my $sp (@searchFolders) {
        print "$sp\n";
    }
    print "\n";
    return;
}

sub getResult {
    my $libname = shift;
    my $num = shift;
    if ($num > -1 && $num <= $#{$foundLibs{$libname}}) {
        return (@{$foundLibs{$libname}})[$num];
    } else {
        print "$num is no valid result!\n";
        return "";
    }
}

sub startSearch {
    my $fullauto = shift;
    #$t = threads->new(\&findHeaders, $searchFolders[$searchDone]);
    $t = threads->new(\&findHeaders, $fullauto);
    return;
}

sub stopSearch {
    if (defined $t) {
        $t->detach();
    }
    return;
}

sub waitForSearch {
    $t->join();
    undef $t;
}

return 1;