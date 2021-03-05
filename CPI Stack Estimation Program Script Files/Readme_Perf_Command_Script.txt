1. Initializing the environment variables:
source shrc

***** Compiling the Benchmark ******

go benchmark_name (e.g. xz_r, leela_r,... )
cd build/build_base_hpca-m64.0000/
make


***** Running the Benchmark ******

1. With "ref" input:

go benchmark_name
cd run/run_base_refrate_hpca-m64.0000/
perf stat -d -I 100 -e branch-load-misses:u,branch-misses:u,L1-dcache-load-misses:u,L1-icache-load-misses:u,dTLB-load-misses:u,iTLB-load-misses:u,cache-misses:u,cycles:u,instructions:u bash run.sh 2>data_ref_name.txt



2. With "train" input:

go benchmark_name
cd run/run_base_train_hpca-m64.0000/
perf stat -d -I 100 -e branch-load-misses:u,branch-misses:u,L1-dcache-load-misses:u,L1-icache-load-misses:u,dTLB-load-misses:u,iTLB-load-misses:u,cache-misses:u,cycles:u,instructions:u bash run.sh 2>data_train_name.txt

3. With "test" input:

go benchmark_name
cd run/run_base_test_hpca-m64.0000/
perf stat -d -I 100 -e branch-load-misses:u,branch-misses:u,L1-dcache-load-misses:u,L1-icache-load-misses:u,dTLB-load-misses:u,iTLB-load-misses:u,cache-misses:u,cycles:u,instructions:u bash run.sh 2>data_test_name.txt


