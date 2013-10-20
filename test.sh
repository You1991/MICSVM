export KMP_AFFINITY='compact'
echo $KMP_AFFINITY
export MIC_ENV_PREFIX=PHI
export PHI_KMP_AFFINITY='balanced'
export OFFLOAD_REPORT=2
export PHI_USE_2MB_BUFFERS=16K
echo $PHI_KMP_AFFINITY
#export OMP_NUM_THREADS=16
icpc -o right_row0 -openmp -simd -vec-report1 sparse_caching.cpp
./right_row0 -c 1.0 -g 0.031250 -o forest12000.mdl ./dataset/real_sim
./libsvm/svm-predict ./dataset/real_sim1000.t forest12000.mdl libsvm_out
