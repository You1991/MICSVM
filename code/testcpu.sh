rm adult.mdl 
rm cpu_csr.optrpt 
rm libsvm_out 
rm sparse_time 
rm svm-train
export KMP_AFFINITY='compact'
echo $KMP_AFFINITY
export MIC_ENV_PREFIX=PHI
export PHI_KMP_AFFINITY='balanced'
export OFFLOAD_REPORT=2
export PHI_USE_2MB_BUFFERS=16K
echo $PHI_KMP_AFFINITY
export OMP_NUM_THREADS=24
icpc -o svm-train -openmp -simd -vec-report1 cpu_csr.cpp
./svm-train -c 0.1 -g 0.0625 -o adult.mdl ../dataset/adult
../libsvm/svm-predict ../dataset/adult.t adult.mdl libsvm_out
