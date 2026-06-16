import os
import sys

# ncpus = "7"
# ngpus = "1"
# mem = "48gb"
ncpus = "14"
ngpus = "2"
mem = "96gb"
io = "0.1"
gputype = "A6000"

### TRAINING
# config = "configs/evt_atlas.yaml"
# config = "configs/part_atlas.yaml"
# ckpt = None

### EVALUATION
### Hamza's training (mc23e)
# config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_hamza/part_atlas.yaml"
# ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_hamza/ckpts/atlas_part_nsteps25_large_20250928-T082955-epoch=54-val_loss_avg=1.6199.ckpt"
# evt_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_hamza/evt_atlas.yaml"
# evt_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_pow_evt_hamza/ckpts/atlas_evt_nsteps25_large_20250922-T120051-epoch=497-val_loss_avg=0.6183.ckpt"

### My training (mc23a + mc23e)
# config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_mc23ae_JZ1-9_nsteps25_20260409-T231642/part_atlas.yaml"
# ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_mc23ae_JZ1-9_nsteps25_20260409-T231642/ckpts/rcfm_atlas_part_mc23ae_JZ1-9_nsteps25_20260409-T231642-epoch=75-val_loss_avg=1.4101.ckpt"
# evt_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260409-T143251/evt_atlas.yaml"
# evt_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260409-T143251/ckpts/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260409-T143251-epoch=85-val_loss_avg=0.5647.ckpt"

### My new training (mc23a + mc23e)
config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_20260415-T212223/part_atlas.yaml"
ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_20260415-T212223/ckpts/rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_20260415-T212223-epoch=76-val_loss_avg=1.3879.ckpt"
evt_config="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260415-T174030/evt_atlas.yaml"
evt_ckpt="/storage/agrp/dreyet/f_delphes/cms-flow-evt/saved_models/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260415-T174030/ckpts/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260415-T174030-epoch=144-val_loss_avg=0.5564.ckpt"

test_samples = {
    "mc23e_test_WprimeWZ_00000x_FullSim": "/storage/agrp/dreyet/f_delphes/data/Wprime_ATLAS_mc16e_FullSim/user.edreyer.mc23_13p6TeV_wprimewz_truth_pflow_fullsim.v4_EXT0/user.edreyer.49778284.EXT0._00000*.root",
    # "mc23e_test_JZ1-9_FullSim.root": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_FullSim.root",
    # "mc23e_test_JZ1-9_hamza_new": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/hamza_samples/test_JZ1-9.root",
    # "mc23e_test_JZ1-9_hamza": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_hamza.root",
    # "JZ1_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz1_truth_pflow_fullsim.v4_EXT0/merged_jz1.root",
    # "JZ2_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz2_truth_pflow_fullsim.v4_EXT0/merged_jz2.root",
    # "JZ3_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz3_truth_pflow_fullsim.v4_EXT0/merged_jz3.root",
    # "JZ4_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz4_truth_pflow_fullsim.v4_EXT0/merged_jz4.root",
    # "JZ5_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz5_truth_pflow_fullsim.v4_EXT0/merged_jz5.root",
    # "JZ6_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz6_truth_pflow_fullsim.v4_EXT0/merged_jz6.root",
    # "JZ7_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz7_truth_pflow_fullsim.v4_EXT0/merged_jz7.root",
    # "JZ8_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz8_truth_pflow_fullsim.v4_EXT0/merged_jz8.root",
    # "JZ9incl_FullSim": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz9incl_truth_pflow_fullsim.v4_EXT0/merged_jz9incl.root",
    # "mc23e_test_WprimeWZ_0000x_AF3": "/storage/agrp/dreyet/f_delphes/data/Wprime_ATLAS_mc16e_AF3/user.edreyer.mc23_13p6TeV_wprimewz_truth_pflow_af3.v4_EXT0/user.edreyer.50091438.EXT0._0000*.root",
    # "JZ1_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz1_truth_pflow.v4_EXT0",
    # "JZ2_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz2_truth_pflow.v4_EXT0",
    # "JZ3_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz3_truth_pflow.v4_EXT0",
    # "JZ4_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz4_truth_pflow.v4_EXT0/merged_jz4.root",
    # "JZ5_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz5_truth_pflow.v4_EXT0",
    # "JZ6_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz6_truth_pflow.v4_EXT0",
    # "JZ7_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz7_truth_pflow.v4_EXT0",
    # "JZ8_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz8_truth_pflow.v4_EXT0",
    # "JZ9incl_AF3": "/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/user.edreyer.mc23_13p6TeV_jz9incl_truth_pflow.v4_EXT0",
}

if len(sys.argv) > 1:
    run_eval = sys.argv[1] == "eval"
else:
    run_eval = False

if run_eval:
    walltime = "32:00:00"
else:
    walltime = "72:00:00"

command = f"qsub -o /storage/agrp/dreyet/f_delphes/cms-flow-evt/logs/output.log"
command += f" -e /storage/agrp/dreyet/f_delphes/cms-flow-evt/logs/error.log"
command += f" -q N -N flow_atlas_fevt -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io={io},gputype={gputype}"
command += f" -v CONFIG={config}"
if ckpt is not None:
    command += f",CKPT={ckpt}"

if run_eval:
    for tag, test_path in test_samples.items():
        sub_command = command
        sub_command += f",ECONFIG={evt_config},ECKPT={evt_ckpt}"
        sub_command += f",TEST_PATH={test_path},BS=1024,NUM_EVENTS=-1"
        sub_command += f",PREFIX={tag}_50steps_"
        sub_command += f" {os.getcwd()}/eval_on_node.sh"
        print(sub_command)
        os.system(sub_command)
else:
    command += f" {os.getcwd()}/run_on_node.sh"
    print(command)
    os.system(command)
