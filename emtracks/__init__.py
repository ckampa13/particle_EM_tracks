import os

if ("EMTRACKS_DDIR" not in os.environ) or ("EMTRACKS_PDIR" not in os.environ):
    print("ERROR! Please set $EMTRACKS_DDIR and $EMTRACKS_PDIR. Setting defaults (current directory)")
    emtracks_ddir = os.getcwd()+'/'
    emtracks_pdir = os.getcwd()+'/'
else:
    emtracks_ddir = os.environ["EMTRACKS_DDIR"]
    emtracks_pdir = os.environ["EMTRACKS_PDIR"]
