Copyright (c) 2020. 
Originally written by Paolo Parotto, University of Wuppertal, 
Department of Physics, Wuppertal D-42119, Germany. 

This is a readme file for the hadronic particle and decay lists 
provided in this webpage (http://nsmn1.uh.edu/cratti/decays.html)

The hadronic lists provided are:
- PDG2005
- PDG2016
- PDG2016Plus
- QM2016Plus

For more info on how the particle lists and the decay lists were created,
please consult Phys. Rev. D 96, 034517 (2017), Phys. Rev. C 98, 034909 (2018) 
and Phys. Rev. C 101, 054905 (2020).

----------------------------------------------------------------------------

All lists are given in two different formats.

For the particle lists (e.g., PDG2005.dat and PDG2005_ThFIST.dat), one has:

- Format 1 (e.g., PDG2005.dat) 
  Both particles and anti-particles are included. The photon is
  always included. There are 12 columns, containing:
	
  1.    ID 
  2.    Name 
  3.    Mass(GeV) 
  4.    Width(GeV) 
  5.    Degeneracy(from spin) 
  6.    Baryon no. 
  7.    Strangeness no. 
  8.    Charm no. 
  9.    Bottom no.
  10.   Isospin
  11.   Electric charge
  12.   No. of decay channels

  Example: Delta(1232)++
  ID   Name          M(GeV) W(GeV) Deg B(aryon) S C B(ottom) I   Q  Dec.No
  -------------------------------------------------------------------------
  2224 Delta(1232)++ 1.232  0.12   4   1        0 0 0        3/2 2  1
 

- Format 2 (for use in Thermal FIST, e.g., PDG2005_ThFIST.dat)
  Anti-particles are not included. The photon is always included.
  There are 14 columns, containing:

  1.    ID 
  2.    Name 
  3.    Stable? (1=yes, 0=no)
  4.    Mass(GeV) 
  5.    Degeneracy (from spin) 
  6.    Statistics (0=boson, 1=fermion) 
  7.    Baryon no. 
  8.    Electric charge 
  9.    Strangeness no.
  10.   Charm no. 
  11.   Absolute strangeness
  12.   Asbolute 'charmness'
  13.   Width(GeV)
  14.   Threshold(GeV)

  Example: Delta(1232)++
  ID   Name          Stable M(GeV) Deg Stat B Q S C |S| |C| W(GeV) Thre(GeV)
  --------------------------------------------------------------------------
  2224 Delta(1232)++ 0      1.232  4   1    1 2 0 0  0   0  0.12   0	


------------------------------------------------------------------------------

For the decay lists (e.g., decays_PDG2005.dat and decays_PDG2005_ThFIST.dat), 
one has:

- Format 1 (e.g., decays_PDG2005.dat) 
  Both particles and anti-particles are included. The photon is
  always included. Each particle appears in the following way. A 12-column
  line repeating the entry from the particle list, followed by a number of
  8-column lines, each representing one decay mode.

	The number of 8-column lines is the same as the 12th entry in the 12-column
  line.
	
  Each 8-column line is structured as follows:
  1.    ID (mother particle)
  2.    No. of daughter particles
  3.    Branching ratio
  4.-8. ID (daughter particle, 0 if none)

  Example: Delta(1232)++
  ID   Name          M(GeV) W(GeV) Deg B(aryon) S C B(ottom) I   Q  Dec.No
  -------------------------------------------------------------------------
  2224 Delta(1232)++ 1.232  0.12   4   1        0 0 0        3/2 2  1

  ID (mother)  No. daughters  BR  ID#1  ID#2  ID#3  ID#4  ID#5
  ------------------------------------------------------------------------
  2224	       2	            1	  2212	211	  0	    0	    0				

- Format 2 (for use in Thermal FIST, e.g., decays_PDG2005_ThFIST.dat)
  Anti-particles are not included. Stable particles are also not included.
  Each particle appears in the following way. A single column line with the 
  particle's ID. A single column line with the number of decay modes. For each 
  decay mode, a line containing:
  1.           Branching ratio
  from 2. on   ID of daughter(s) 

  Example: Delta(1232)++
  2224                                 # Delta(1232)++
  1                                    # 1 decay channel
  1               2212 211             # Delta(1232)++ -> p + pi+


-------------------------------------------------------------------------------

  CONTACT 
  For problems or questions, contact Paolo Parotto at parotto@uni-wuppertal.de
  
  WHEN USING THESE LISTS
  When using these lists, please refer to the following works:
  - Phys.Rev. D96 (2017) no.3, 034517 (https://inspirehep.net/record/1512119)
  - Phys.Rev. C98 (2018) no.3, 034909 (https://inspirehep.net/record/1636208)
  - Phys.Rev. C101 (2020) no.5, 054905 (https://inspirehep.net/literature/1782970)
