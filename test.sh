export KMP_AFFINITY='compact'
echo $KMP_AFFINITY
export MIC_ENV_PREFIX=PHI
export PHI_KMP_AFFINITY='balanced'
export OFFLOAD_REPORT=2
export PHI_USE_2MB_BUFFERS=16K
echo $PHI_KMP_AFFINITY
#export OMP_NUM_THREADS=16
icpc -o mic_svm -openmp -simd -vec-report1 sparse_caching.cpp
./mic_svm -c 0.1  -g 0.0625 -o adult.mdl ./adult

