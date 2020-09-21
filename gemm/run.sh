export CUOPT_GEMM_N=256
export CUOPT_GEMM_M=256
export CUOPT_GEMM_K=256
nvprof --log-file res/NNN256.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=512
export CUOPT_GEMM_M=512
export CUOPT_GEMM_K=512
nvprof --log-file res/NNN512.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=1024
export CUOPT_GEMM_M=1024
export CUOPT_GEMM_K=1024
nvprof --log-file res/NNN1024.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=2048
export CUOPT_GEMM_M=2048
export CUOPT_GEMM_K=2048
nvprof --log-file res/NNN2048.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=4096
export CUOPT_GEMM_M=4096
export CUOPT_GEMM_K=4096
nvprof --log-file res/NNN4096.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=8192
export CUOPT_GEMM_M=8192
export CUOPT_GEMM_K=8192
nvprof --log-file res/NNN8192.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=16384
export CUOPT_GEMM_M=16384
export CUOPT_GEMM_K=16384
nvprof --log-file res/NNN16384.gemm_metric_tc -s -m all ./gemm
export CUOPT_GEMM_N=32768
export CUOPT_GEMM_M=32768
export CUOPT_GEMM_K=32768
nvprof --log-file res/NNN32768.gemm_metric_tc -s -m all ./gemm
