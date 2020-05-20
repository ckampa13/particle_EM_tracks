from scipy.constants import c, elementary_charge

one_gev_c2_to_kg = 1.7826619e-27
one_kgm_s_to_mev_c = (1.7826619e-30 * c)**(-1)

q_factor = elementary_charge / one_gev_c2_to_kg / c

# p = 10 kg m / s
# p_MeV_c = p * one_kgm_s_to_mev_c
# p_kgm_s = p_MeV_c / one_kgm_s_to_mev_c
