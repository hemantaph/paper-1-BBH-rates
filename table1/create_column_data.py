# Import LeR
from ler.rates import LeR
from ler.utils import get_param_from_json
# other necessary imports
from astropy.cosmology import LambdaCDM
# Import helper modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])

if len(sys.argv) > 1:
    rate_configuration = sys.argv[1]
else:
    rate_configuration = 'baseline'

npool=32

if rate_configuration == 'baseline':
    # LeR configurations:
    merger_rate_density = {'R0': 2.39e-08, 'b2': 1.6, 'b3': 2.0, 'b4': 30 }
    merger_rate_density_name = 'merger_rate_density_bbh_popI_II_oguri2018'
    source_frame_masses = {
            'mminbh': 4.98, 'mmaxbh': 112.5, 'alpha': 3.78, 'mu_g': 32.27, 'sigma_g': 3.88, 'lambda_peak': 0.03, 'delta_m': 4.8, 'beta': 0.81
            }
    source_frame_masses_name = 'binary_masses_BBH_popI_II_powerlaw_gaussian'
    lens_type = 'epl_galaxy'
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    ler_directory="./ler_data"+"_baseline"
    interpolator_directory="./interpolator_pickle"+"_baseline"
    psds = {'L1':'aLIGO_O4_high_asd.txt','H1':'aLIGO_O4_high_asd.txt', 'V1':'AdV_asd.txt', 'K1':'KAGRA_design_asd.txt'}
    ifos = ['L1', 'H1', 'V1']
    print("Chose baseline name")
else:
    raise ValueError("Invalid configuration name")

# Load LeR and compute all the SNRs etc
ler = LeR(
    npool=npool, # number of processors to use
    z_min=0.0, # minimum redshift
    z_max=10.0, # maximum redshift
    event_type='BBH', # event type
    size=100000, # number of events to simulate
    batch_size=50000, # batch size
    cosmology=cosmo, # cosmology
    snr_finder=None, # snr calculator from 'gwsnr' package will be used
    pdet_finder=None,  # will not be consider unless specified
    list_of_detectors=None, # list of detectors that will be considered when calculating snr or pdet for lensed events. if None, all the detectors from 'gwsnr' will be considered
    json_file_names=dict(
        ler_params="ler_params.json", # to store initialization parameters and important results
        unlensed_param="unlensed_param.json", # to store all unlensed events
        unlensed_param_detectable="unlensed_param_detectable.json", # to store only detectable unlensed events
        lensed_param="lensed_param.json", # to store all lensed events 
        lensed_param_detectable="lensed_param_detectable.json"), # to store only detectable lensed events
    interpolator_directory=interpolator_directory, # directory to store the interpolator pickle files. 'ler' uses interpolation to get values of various functions to speed up the calculations (relying on numba njit).
    create_new_interpolator = False, # if True, will create new interpolator files
    ler_directory=ler_directory, # directory to store all the outputs
    verbose=False, # if True, will print all information at initialization

    # CBCSourceParameterDistribution class arguments
    source_priors= {
        'merger_rate_density': merger_rate_density_name, 
        'source_frame_masses': source_frame_masses_name, 
        'zs': 'sample_source_redshift', 
        'geocent_time': 'sampler_uniform', 
        'ra': 'sampler_uniform', 
        'dec': 'sampler_cosine', 
        'phase': 'sampler_uniform', 
        'psi': 'sampler_uniform', 
        'theta_jn': 'sampler_sine'
        },
    source_priors_params= {
        'merger_rate_density': merger_rate_density, 
        'source_frame_masses': source_frame_masses, 
        'zs': None, 
        'geocent_time': {
            'min_': 1238166018, 'max_': 1269702018
            }, 
        'ra': {
            'min_': 0.0, 'max_': 6.283185307179586
            }, 
        'dec': None, 
        'phase': {
            'min_': 0.0, 'max_': 6.283185307179586
            }, 
        'psi': {
            'min_': 0.0, 'max_': 3.141592653589793
            }, 
        'theta_jn': None
        },
    spin_zero= True, # if True, spins will be set to zero
    spin_precession= False, # if True, spins will be precessing

    # LensGalaxyParameterDistribution class arguments
    lens_type = lens_type,
    lens_functions =  {
        'strong_lensing_condition': 'rjs_with_cross_section_SIE',
        'optical_depth': 'optical_depth_SIE_hemanta',
        'param_sampler_type': 'sample_all_routine'
        },
    lens_priors =  {
        'source_redshift_sl': 'strongly_lensed_source_redshifts', 
        'lens_redshift': 'lens_redshift_SDSS_catalogue', 
        'velocity_dispersion': 'velocity_dispersion_ewoud', 
        'axis_ratio': 'axis_ratio_rayleigh', 
        'axis_rotation_angle': 'axis_rotation_angle_uniform', 
        'shear': 'shear_norm', 
        'mass_density_spectral_index': 'mass_density_spectral_index_normal', 'source_parameters': 'sample_gw_parameters'
        },
    lens_priors_params =  {
        'source_redshift_sl': None, 
        'lens_redshift': None, 
        'velocity_dispersion': {
            'vd_min':10., 'vd_max':350.
            }, 
        'axis_ratio': {
            'q_min': 0.2, 'q_max': 1.0
            }, 
        'axis_rotation_angle': {
            'phi_min': 0.0, 'phi_max': 6.283185307179586
            }, 
        'shear': {
            'scale': 0.05
            }, 
        'mass_density_spectral_index': {
            'mean': 2.0, 'std': 0.2
            }, 
        'source_parameters': None
        },

    # ImageProperties class arguments
    n_min_images = 2,
    n_max_images = 4,
    geocent_time_min = 1238166018,
    geocent_time_max = 1269702018,
    lens_model_list = ['EPL_NUMBA', 'SHEAR'],

    # gwsnr package arguments
    mtot_min = 2.0,
    mtot_max = 184.98599853446768,
    ratio_min = 0.1,
    ratio_max = 1.0,
    # mtot_resolution = 500,
    # ratio_resolution = 50,
    sampling_frequency = 2048.0,
    waveform_approximant = 'IMRPhenomD',
    minimum_frequency = 20.0,
    snr_type = 'interpolation',
    psds = psds,
    ifos = ifos,
    interpolator_dir = interpolator_directory,
    # gwsnr_verbose = True,
    # multiprocessing_verbose = True,
    mtot_cut = True,
)

# Generate unlensed event parameters
output_jsonfile_unlensed = 'n_unlensed_param_detectable.json'
meta_data_file_unlensed='meta_unlensed.json'
ler.selecting_n_unlensed_detectable_events(
    size=200000,
    batch_size=250000,
    snr_threshold=8.0,
    resume=True,
    output_jsonfile=output_jsonfile_unlensed,
    meta_data_file=meta_data_file_unlensed,
    detectability_condition='step_function',
    trim_to_size=False,
)

# getting data from json
meta_data_unlensed= get_param_from_json("./"+ler_directory+"/"+meta_data_file_unlensed)

# plot the rate vs sampling size for the sake of 
plt.figure(figsize=(6,4))
plt.plot(meta_data_unlensed['events_total'], meta_data_unlensed['total_rate'], 'o-')
plt.xlabel(r"Sampling size")
plt.ylabel(r"Rate (per year)")
plt.title(r"Rate vs Sampling size")
plt.grid(alpha=0.4)
plt.savefig("./"+ler_directory+"/diagnosis01_rate_convergence.pdf", bbox_inches='tight')
plt.close()

# Add the first column (ULR/yr)
table_column_data = {}
table_column_data['unlensed_rate_per_year'] = meta_data_unlensed['total_rate'][-1]

# Get the lensed events per year next
output_jsonfile_lensed='n_lensed_param_detectable.json'
meta_data_file_lensed = 'meta_lensed_O4.json'
ler.selecting_n_lensed_detectable_events(
    size=20000,
    batch_size=500000,
    snr_threshold=[8.0, 8.0],
    num_img=[1, 1],
    resume=True,
    detectability_condition='step_function',
    output_jsonfile=output_jsonfile_lensed,
    meta_data_file=meta_data_file_lensed,
    trim_to_size=False,
    nan_to_num=False,
);

# Plot the diagnostics for the lensed events


























# # npz file to be created
# filename_out = sys.argv[2]

# x = np.arange(10)

# # Save the data
# np.savez(filename_out, x)

