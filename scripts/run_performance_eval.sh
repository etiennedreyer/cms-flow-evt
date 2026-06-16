# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ3456_84_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ3-6.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ3456_84_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZall_65_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ1-8.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZall_65_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ1-2_65_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ1-2.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ1-2_65_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ3-6_65_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS/test_JZ3-6.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_rcfm_atlas_part_JZ3-6_65_25"

# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/rcfm_atlas_part_JZ7-8_65_25.root"

### Parnassus
# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/JZ4_FullSim_rcfm_atlas_part_JZall_54_25.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/user.edreyer.mc23_13p6TeV_jz4_truth_pflow_fullsim.v4_EXT0/merged_jz4.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_JZ4_FullSim_rcfm_atlas_part_JZall_54_25.root"

### Hamza's training (mc23e)
# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/mc23e_test_JZ1-9_hamza_atlas_part_nsteps25_large_54_25.root"
# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/mc23e_test_JZ1-9_hamza_atlas_part_nsteps50_large_54_50.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_hamza.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_mc23e_test_JZ1-9_hamza_atlas_part_nsteps25_large_54_25.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_mc23e_test_JZ1-9_hamza_atlas_part_nsteps50_large_54_50.root"

### My training (mc23a + mc23e)
# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/mc23e_test_JZ1-9_FullSim_50steps_rcfm_atlas_part_mc23ae_JZ1-9_nsteps25_75_50.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_FullSim.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_mc23e_test_JZ1-9_FullSim_50steps_rcfm_atlas_part_mc23ae_JZ1-9_nsteps25_75_50.root"

### My new training (mc23a + mc23e), eval on JZ1-9
# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/mc23e_test_JZ1-9_FullSim.root_50steps_rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_76_50.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_FullSim.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_mc23e_test_JZ1-9_FullSim_50steps_rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_76_50.root"

### My new training (mc23a + mc23e), eval on WprimeWZ
# eval_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/mc23e_test_WprimeWZ_00000x_FullSim_50steps_rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_76_50.root"
# test_file="/storage/agrp/dreyet/f_delphes/data/Wprime_ATLAS_mc16e_FullSim/user.edreyer.mc23_13p6TeV_wprimewz_truth_pflow_fullsim.v4_EXT0_user.edreyer.49778284.EXT0._00000x.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_mc23e_test_WprimeWZ_00000x_FullSim_50steps_rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_76_50.root"

# python evaluation/preprocess_eval_fast.py \
#   -e ${eval_file} \
#   -d "${test_file}" \
#   -o ${out_file} \
#   -n -1 \
#   -dr 0.4 \
#   --eta 2.7

### AF3 (i.e. "Delphes")
# test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/test_JZ1-9_hamza_AF3.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_test_JZ1-9_hamza_AF3.root"
# eval_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/test_JZ1-9_AF3.root"
eval_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_AF3/test_JZ1-9_AF3_bugfix.root"
test_file="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_FullSim.root"
out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_test_JZ1-9_AF3_bugfix.root"

# test_file="/storage/agrp/dreyet/f_delphes/data/Wprime_ATLAS_mc16e_AF3/user.edreyer.mc23_13p6TeV_wprimewz_truth_pflow_af3.v4_EXT0_merged.root"
# out_file="/storage/agrp/dreyet/f_delphes/cms-flow-evt/evals/eval_test_WprimeWZ_AF3.root"

python evaluation/preprocess_eval_fast.py \
  -e ${eval_file} \
  -d ${test_file} \
  -o ${out_file} \
  -n -1 \
  -dr 0.4 \
  --eta 2.7 \
  -dl
