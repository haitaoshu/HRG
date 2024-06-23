import numpy as np
from scipy.special import kn
import sys

def meson_chi(Q, S, m_by_T, muQ_by_T, muS_by_T, j, k, l):
    z = np.exp(-1*(Q*muQ_by_T+S*muS_by_T)) #the minus sign is added to make direct comparison to lattice calculations where the chemical potential is added positively to the original action 
    return z**l*(l*Q)**j*(l*S)**k/l**2*kn(2, l*m_by_T)

def baryon_chi_approx(B, Q, S, m_by_T, muB_by_T, muQ_by_T, muS_by_T, i, j, k):
    z = np.exp(-1*(B*muB_by_T+Q*muQ_by_T+S*muS_by_T))
    return z*B**i*Q**j*S**k*kn(2, m_by_T)

def baryon_chi(B, Q, S, m_by_T, muB_by_T, muQ_by_T, muS_by_T, i, j, k, l):
    z = np.exp(-1*(B*muB_by_T+Q*muQ_by_T+S*muS_by_T))
    return z**l*(l*B)**i*(l*Q)**j*(l*S)**k/l**2*kn(2, l*m_by_T)

#calculate generalized susceptibilities via modified Bessel functions of the second kind with approximation based on the fact that  the large mass of even the lightest baryon compared to the temperature scale, contributions with l > 1 can be neglected in the baryon term.
def chi_ijk_BQS(i,j,k,B,Q,S, d, m_by_T, muB_by_T, muQ_by_T, muS_by_T): 
    ls = np.arange(1, 10)
    if  B == 0: #for meson, there is no baryonic charge contribution
        if abs(m_by_T)<1e-4: #to skip photon
            return 0
        else:
            if i == 0:
                mesons = meson_chi(Q, S, m_by_T, muQ_by_T, muS_by_T, j, k, ls)
                chi = d*(m_by_T)**2/2/np.pi**2*np.sum(mesons)
                return chi
            else:
                return 0
    else:
        baryons = baryon_chi_approx(B, Q, S, m_by_T, muB_by_T, muQ_by_T, muS_by_T, i, j, k)
        chi = d*(m_by_T)**2/2/np.pi**2*baryons
        #baryons = baryon_chi(B, Q, S, m_by_T, muB_by_T, muQ_by_T, muS_by_T, i, j, k, ls)
        #chi = d*(m_by_T)**2/2/np.pi**2*np.sum(baryons)
        return chi

#calculate generalized susceptibilities exactly in the Boltzmann approximation
def chi_exact(i,j,k,B,Q,S, d, m_by_T, muB_by_T, muQ_by_T, muS_by_T):#m, flag_mb, g, T, V, mu):
    #ks = np.arange(1, 1000)
    if  B == 0: #for meson
        if abs(m_by_T)<1e-4: #to skip photon
            return 0
        else:
            if i == 0:
                integrand = lambda p: -p**2 * np.log(1-Q**j*S**k*np.exp(-(np.sqrt(p**2 + m_by_T**2) + muQ_by_T*Q + muS_by_T*S)))#bose
            else:
                return 0 
    else:
        integrand = lambda p: p**2 * np.log(1+B**i*Q**j*S**k*np.exp(-(np.sqrt(p**2 + m_by_T**2) + muB_by_T*B + muQ_by_T*Q + muS_by_T*S)))#fermi-dirac
    p_vals = np.linspace(0, 100*m_by_T, 1000)
    chi = d/2/np.pi**2 * np.trapz(integrand(p_vals), p_vals)
    return chi 

def convert_value(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            return value

keys = ['ID', 'Name', 'mass', 'width', 'degeneracy', 'baryon', 'strangeness', 'charm', 'bottom', 'isospin', 'charge', 'decay-channel']
hadron_data = []
with open('./hadron_lists_Sep2021/PDG2016Plus_massorder.dat', 'r') as file:
#with open('./hadron_lists_Sep2021/QM2016Plus_massorder.dat.new', 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace characters (like newline)
        line = line.strip()
        # Split the line into key-value pairs
        values = line.split('\t')
        # Initialize an empty dictionary for the current record
        record = {}
        # Process each key-value pair
        for ind, value in enumerate(values):
            record[keys[ind]] = convert_value(value)
        # Add the record to the list
        hadron_data.append(record)
# Print the resulting list of dictionaries
#print(hadron_data)

T = 0.107712 #GeV
#mu_B = 0
mu_Q = 0
mu_S = 0
#for i in range(2,3):
for i in range(1,2):
    for j in range(1):
        for k in range(1):
            for ii_muB_by_T in range(101):
                imuB_by_T = 1j*ii_muB_by_T/100*np.pi 
                imuQ_by_T = 0
                imuS_by_T = 0
                #imuSum_by_T = imuB_by_T + imuQ_by_T + imuS_by_T
                chi_sum = 0
                for hadron in hadron_data:
                    m_by_T = hadron["mass"]/T
                    #muB_by_T = hadron["baryon"] * mu_B / T
                    #muQ_by_T = hadron["charge"] * mu_Q / T
                    #muS_by_T = hadron["strangeness"] * mu_S / T
                    #muSum_by_T = imuB_by_T+muQ_by_T+muS_by_T
                    #chi_tmp = chi_ijk_BQS(i,j,k,hadron["baryon"], hadron["charge"], hadron["strangeness"], hadron["degeneracy"], m_by_T, imuB_by_T, imuQ_by_T, imuS_by_T)
                    chi_tmp = chi_exact(i,j,k,hadron["baryon"], hadron["charge"], hadron["strangeness"], hadron["degeneracy"], m_by_T, imuB_by_T, imuQ_by_T, imuS_by_T)
                    chi_sum += chi_tmp
                #print("muB_by_T:", muB_by_T, " chi_{",i,j,k,"}:", chi_sum)
                print(ii_muB_by_T/100*np.pi, chi_sum.real, chi_sum.imag) #
                #sys.exit(1)
